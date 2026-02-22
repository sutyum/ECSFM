from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import matplotlib
import numpy as np

from ecsfm.fm.eval_classical import _resolve_model_geometry
from ecsfm.fm.model import VectorFieldNet
from ecsfm.fm.posterior import (
    CEMPosteriorConfig,
    PosteriorInferenceConfig,
    infer_parameter_posterior,
)
from ecsfm.fm.train import (
    MODEL_META_FILENAME,
    NORMALIZERS_FILENAME,
    load_model_metadata,
    load_saved_normalizers,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer posterior over unknown conditioning parameters from partial current observations."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained surrogate_model.eqx")
    parser.add_argument(
        "--measurement",
        type=str,
        required=True,
        help="NPZ file with keys: e (applied signal), i_obs (observed current), optional i_mask, known_p_core, known_p_mask",
    )
    parser.add_argument("--output-dir", type=str, default="/tmp/ecsfm/posterior_inference", help="Output directory")
    parser.add_argument("--normalizers", type=str, default=None, help="Path to normalizers NPZ")
    parser.add_argument("--meta", type=str, default=None, help="Path to model metadata JSON")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--task-idx", type=int, default=None, help="Optional task index to fix in conditioning")
    parser.add_argument("--stage-idx", type=int, default=None, help="Optional stage index to fix in conditioning")

    parser.add_argument("--n-particles", type=int, default=96, help="CEM particles per iteration")
    parser.add_argument("--n-iters", type=int, default=6, help="CEM iterations")
    parser.add_argument("--elite-frac", type=float, default=0.25, help="CEM elite fraction")
    parser.add_argument("--n-mc", type=int, default=4, help="Monte Carlo trajectories per particle")
    parser.add_argument("--n-steps", type=int, default=100, help="Flow integration steps")
    parser.add_argument("--obs-noise-std", type=float, default=0.25, help="Observation noise std for likelihood")
    return parser.parse_args()


def _load_measurement(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Measurement file not found: {path}")
    with np.load(path) as data:
        keys = set(data.keys())
        e_key = "e" if "e" in keys else ("E" if "E" in keys else None)
        i_key = "i_obs" if "i_obs" in keys else ("i" if "i" in keys else None)
        if e_key is None or i_key is None:
            raise KeyError(f"Measurement NPZ must contain e/E and i_obs/i keys. Found keys: {sorted(keys)}")

        payload = {
            "e": np.asarray(data[e_key], dtype=np.float32).reshape(-1),
            "i_obs": np.asarray(data[i_key], dtype=np.float32).reshape(-1),
        }
        if "i_mask" in data:
            payload["i_mask"] = np.asarray(data["i_mask"], dtype=np.float32).reshape(-1)
        elif "obs_mask" in data:
            payload["i_mask"] = np.asarray(data["obs_mask"], dtype=np.float32).reshape(-1)

        if "known_p_core" in data:
            payload["known_p_core"] = np.asarray(data["known_p_core"], dtype=np.float32).reshape(-1)
        if "known_p_mask" in data:
            payload["known_p_mask"] = np.asarray(data["known_p_mask"], dtype=np.float32).reshape(-1)
        return payload


def _decode_base_params(base_vec: np.ndarray, max_species: int) -> dict[str, list[float]]:
    m = int(max_species)
    out = {
        "D_ox": np.exp(base_vec[0:m]).tolist(),
        "D_red": np.exp(base_vec[m : 2 * m]).tolist(),
        "C_ox": base_vec[2 * m : 3 * m].tolist(),
        "C_red": base_vec[3 * m : 4 * m].tolist(),
        "E0": base_vec[4 * m : 5 * m].tolist(),
        "k0": np.exp(base_vec[5 * m : 6 * m]).tolist(),
        "alpha": base_vec[6 * m : 7 * m].tolist(),
    }
    return out


def _inject_task_stage_constraints(
    known_p_core: np.ndarray,
    known_p_mask: np.ndarray,
    p_mean: np.ndarray,
    geometry: dict[str, Any],
    task_idx_arg: int | None,
    stage_idx_arg: int | None,
) -> tuple[np.ndarray, np.ndarray, int | None, int | None]:
    phys_dim_base = int(geometry["phys_dim_base"])
    n_tasks = int(geometry["n_tasks"])
    n_stages = int(geometry["n_stages"])

    task_idx = task_idx_arg
    stage_idx = stage_idx_arg

    cursor = phys_dim_base
    if n_tasks > 0:
        if task_idx is None:
            freq = p_mean[cursor : cursor + n_tasks]
            task_idx = int(np.argmax(freq))
        task_idx = int(np.clip(task_idx, 0, n_tasks - 1))
        known_p_core[cursor : cursor + n_tasks] = 0.0
        known_p_core[cursor + task_idx] = 1.0
        known_p_mask[cursor : cursor + n_tasks] = True
        cursor += n_tasks

    if n_stages > 0:
        if stage_idx is None:
            freq = p_mean[cursor : cursor + n_stages]
            stage_idx = int(np.argmax(freq))
        stage_idx = int(np.clip(stage_idx, 0, n_stages - 1))
        known_p_core[cursor : cursor + n_stages] = 0.0
        known_p_core[cursor + stage_idx] = 1.0
        known_p_mask[cursor : cursor + n_stages] = True

    return known_p_core, known_p_mask, task_idx, stage_idx


def _plot_predictive(
    output_path: Path,
    observed_current: np.ndarray,
    observed_mask: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
) -> None:
    t = np.arange(observed_current.shape[0], dtype=float)
    mask = observed_mask.astype(bool)
    lower1 = pred_mean - pred_std
    upper1 = pred_mean + pred_std
    lower2 = pred_mean - 2.0 * pred_std
    upper2 = pred_mean + 2.0 * pred_std

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(t, lower2, upper2, color="lightsteelblue", alpha=0.6, label="pred ±2σ")
    ax.fill_between(t, lower1, upper1, color="cornflowerblue", alpha=0.45, label="pred ±1σ")
    ax.plot(t, pred_mean, color="navy", lw=1.4, label="pred mean")
    ax.plot(t[mask], observed_current[mask], "k.", ms=3, label="observed")
    if np.any(~mask):
        ax.plot(t[~mask], observed_current[~mask], ".", color="gray", ms=2, alpha=0.35, label="unobserved points")
    ax.set_title("Posterior Predictive Current")
    ax.set_xlabel("Trace Index")
    ax.set_ylabel("Current")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizers_path = Path(args.normalizers) if args.normalizers else checkpoint.parent / NORMALIZERS_FILENAME
    meta_path = Path(args.meta) if args.meta else checkpoint.parent / MODEL_META_FILENAME

    normalizers = load_saved_normalizers(normalizers_path)
    model_meta = load_model_metadata(meta_path)
    geometry = _resolve_model_geometry(normalizers, model_meta)

    key = jax.random.PRNGKey(np.uint32(args.seed))
    _, model_key = jax.random.split(key)
    model = VectorFieldNet(
        state_dim=int(geometry["state_dim"]),
        hidden_size=int(model_meta.get("hidden_size", 128)),
        depth=int(model_meta.get("depth", 3)),
        cond_dim=int(model_meta.get("cond_dim", 32)),
        phys_dim=int(geometry["phys_dim"]),
        signal_channels=int(geometry["signal_channels"]),
        key=model_key,
    )
    model = eqx.tree_deserialise_leaves(checkpoint, model)

    measurement = _load_measurement(Path(args.measurement))
    e_signal = measurement["e"]
    i_obs = measurement["i_obs"]
    i_mask = measurement.get("i_mask", None)

    phys_dim_core = int(geometry["phys_dim_core"])
    known_p_core = np.zeros((phys_dim_core,), dtype=np.float32)
    known_p_mask = np.zeros((phys_dim_core,), dtype=bool)
    if "known_p_core" in measurement:
        known = np.asarray(measurement["known_p_core"], dtype=np.float32).reshape(-1)
        if known.shape[0] != phys_dim_core:
            raise ValueError(
                f"known_p_core length mismatch: expected {phys_dim_core}, got {known.shape[0]}"
            )
        known_p_core = known
    if "known_p_mask" in measurement:
        kmask = np.asarray(measurement["known_p_mask"], dtype=np.float32).reshape(-1)
        if kmask.shape[0] != phys_dim_core:
            raise ValueError(
                f"known_p_mask length mismatch: expected {phys_dim_core}, got {kmask.shape[0]}"
            )
        known_p_mask = kmask >= 0.5

    _, _, _, _, p_mean, _ = normalizers
    known_p_core, known_p_mask, task_idx, stage_idx = _inject_task_stage_constraints(
        known_p_core=known_p_core,
        known_p_mask=known_p_mask,
        p_mean=np.asarray(p_mean, dtype=np.float32),
        geometry=geometry,
        task_idx_arg=args.task_idx,
        stage_idx_arg=args.stage_idx,
    )

    posterior_cfg = PosteriorInferenceConfig(
        cem=CEMPosteriorConfig(
            n_particles=args.n_particles,
            n_iterations=args.n_iters,
            elite_fraction=args.elite_frac,
        ),
        n_mc_per_particle=args.n_mc,
        n_integration_steps=args.n_steps,
        obs_noise_std=args.obs_noise_std,
    )

    posterior = infer_parameter_posterior(
        model=model,
        normalizers=normalizers,
        geometry=geometry,
        observed_current=i_obs,
        applied_signal=e_signal,
        known_p_core=known_p_core,
        known_p_mask=known_p_mask,
        obs_mask=i_mask,
        config=posterior_cfg,
        seed=args.seed,
    )

    pred_mean = np.asarray(posterior["predictive_mean_current"], dtype=np.float32)
    pred_std = np.asarray(posterior["predictive_std_current"], dtype=np.float32)
    obs_current = np.asarray(posterior["observed_current"], dtype=np.float32)
    obs_mask = np.asarray(posterior["observed_mask"], dtype=np.float32)

    plot_path = output_dir / "posterior_predictive.png"
    _plot_predictive(plot_path, obs_current, obs_mask, pred_mean, pred_std)

    weights = np.asarray(posterior["posterior_weights"], dtype=np.float32)
    samples_raw = np.asarray(posterior["posterior_samples_raw"], dtype=np.float32)
    top_idx = np.argsort(weights)[-8:][::-1]

    mean_raw = np.asarray(posterior["posterior_mean_raw"], dtype=np.float32)
    std_raw = np.asarray(posterior["posterior_std_raw"], dtype=np.float32)
    max_species = int(geometry["max_species"])
    phys_dim_base = int(geometry["phys_dim_base"])
    base_mean = mean_raw[:phys_dim_base]
    base_std = std_raw[:phys_dim_base]

    summary = {
        "checkpoint": str(checkpoint),
        "measurement": str(Path(args.measurement)),
        "task_idx": task_idx,
        "stage_idx": stage_idx,
        "reliability": posterior["reliability"],
        "posterior_top_weights": weights[top_idx].tolist(),
        "posterior_top_indices": top_idx.tolist(),
        "posterior_top_samples_raw": samples_raw[top_idx].tolist(),
        "posterior_mean_base_params": _decode_base_params(base_mean, max_species=max_species),
        "posterior_std_base_params": _decode_base_params(base_std, max_species=max_species),
        "artifacts": {
            "predictive_plot": str(plot_path),
            "posterior_npz": str(output_dir / "posterior_artifacts.npz"),
        },
    }

    np.savez(
        output_dir / "posterior_artifacts.npz",
        posterior_samples_norm=np.asarray(posterior["posterior_samples_norm"], dtype=np.float32),
        posterior_samples_raw=np.asarray(posterior["posterior_samples_raw"], dtype=np.float32),
        posterior_weights=weights,
        posterior_mean_norm=np.asarray(posterior["posterior_mean_norm"], dtype=np.float32),
        posterior_std_norm=np.asarray(posterior["posterior_std_norm"], dtype=np.float32),
        posterior_mean_raw=mean_raw,
        posterior_std_raw=std_raw,
        predictive_mean_current=pred_mean,
        predictive_std_current=pred_std,
        observed_current=obs_current,
        observed_mask=obs_mask,
        applied_signal=np.asarray(posterior["applied_signal"], dtype=np.float32),
    )

    summary_path = output_dir / "posterior_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
