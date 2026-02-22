import argparse
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ecsfm.data.generate import (
    STAGE_NAMES,
    TASK_NAMES,
    get_ca_waveform,
    get_cv_waveform,
    get_eis_waveform,
    get_swv_waveform,
)
from ecsfm.fm.model import VectorFieldNet
from ecsfm.fm.train import (
    MODEL_META_FILENAME,
    NORMALIZERS_FILENAME,
    _build_param_input,
    _build_signal_input,
    integrate_flow,
    load_dataset,
    load_model_metadata,
    load_saved_normalizers,
)
from ecsfm.sim.experiment import simulate_electrochem

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENARIO_TASK_MAP = {
    "Ferrocene_Reversible_CV": "cv_reversible",
    "Multi_Species_CV": "cv_multispecies",
    "SWV_Ferrocene": "swv_pulse",
    "CA_Step": "ca_step",
    "EIS_Sine": "eis_low_freq",
}

SCENARIO_STAGE_MAP = {
    "Ferrocene_Reversible_CV": "foundation",
    "Multi_Species_CV": "bridge",
    "SWV_Ferrocene": "bridge",
    "CA_Step": "foundation",
    "EIS_Sine": "frontier",
}


def _build_optional_features(extras: dict[str, Any], n_rows: int) -> np.ndarray | None:
    task_id = extras.get("task_id")
    stage_id = extras.get("stage_id")
    task_names = extras.get("task_names")
    stage_names = extras.get("stage_names")

    has_labels = (task_id is not None) or (stage_id is not None)
    if not (has_labels or task_names is not None or stage_names is not None):
        return None

    if task_id is None:
        task_id = np.zeros((n_rows,), dtype=np.int32)
    else:
        task_id = np.asarray(task_id, dtype=np.int32).reshape(-1)

    if stage_id is None:
        stage_id = np.zeros((n_rows,), dtype=np.int32)
    else:
        stage_id = np.asarray(stage_id, dtype=np.int32).reshape(-1)

    n_tasks = max(1, int(np.max(task_id)) + 1)
    n_stages = max(1, int(np.max(stage_id)) + 1)
    if task_names is not None:
        n_tasks = max(n_tasks, len(task_names))
    if stage_names is not None:
        n_stages = max(n_stages, len(stage_names))

    task_oh = np.eye(n_tasks, dtype=np.float32)[task_id]
    stage_oh = np.eye(n_stages, dtype=np.float32)[stage_id]
    return np.concatenate([task_oh, stage_oh], axis=1)


def get_normalizers_from_dataset(data_path: str, seed: int = 42, val_split: float = 0.2):
    c_ox, c_red, curr, sigs, params_base, extras = load_dataset(data_path)
    if c_ox.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute stable normalizers")

    optional_features = _build_optional_features(extras, n_rows=c_ox.shape[0])
    params = np.asarray(params_base, dtype=np.float32)
    if optional_features is not None:
        params = np.concatenate([params, optional_features], axis=1)

    dataset_x = jnp.concatenate([jnp.asarray(c_ox), jnp.asarray(c_red), jnp.asarray(curr)], axis=1)
    sigs = jnp.asarray(sigs)
    params = jnp.asarray(params)

    key = jax.random.PRNGKey(np.uint32(seed))
    _, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, len(dataset_x))

    dataset_x = dataset_x[indices]
    sigs = sigs[indices]
    params = params[indices]

    val_size = int(len(dataset_x) * val_split)
    val_size = min(max(1, val_size), len(dataset_x) - 1)

    train_x = dataset_x[:-val_size]
    train_e = sigs[:-val_size]
    train_p = params[:-val_size]

    x_mean = jnp.mean(train_x, axis=0)
    x_std = jnp.std(train_x, axis=0) + 1e-5
    e_mean = jnp.mean(train_e, axis=0)
    e_std = jnp.std(train_e, axis=0) + 1e-5
    p_mean = jnp.mean(train_p, axis=0)
    p_std = jnp.std(train_p, axis=0) + 1e-5

    return x_mean, x_std, e_mean, e_std, p_mean, p_std


def calculate_metrics(gt_trace, gen_trace, gtox, genox, gtred, genred):
    gt_trace_comp = np.array(gt_trace)
    gen_trace_comp = np.array(gen_trace)

    current_mse = float(mean_squared_error(gt_trace_comp, gen_trace_comp))
    current_mae = float(mean_absolute_error(gt_trace_comp, gen_trace_comp))
    current_r2 = float(r2_score(gt_trace_comp, gen_trace_comp))
    peak_pos_err = float(abs(np.max(gt_trace_comp) - np.max(gen_trace_comp)))
    peak_neg_err = float(abs(np.min(gt_trace_comp) - np.min(gen_trace_comp)))

    rmse = float(np.sqrt(current_mse))
    gt_range = float(np.ptp(gt_trace_comp))
    gt_rms = float(np.sqrt(np.mean(np.square(gt_trace_comp))))
    eps = 1e-8

    nrmse_range = rmse / max(gt_range, eps)
    nmae_rms = current_mae / max(gt_rms, eps)
    peak_err_rel = 0.5 * (peak_pos_err + peak_neg_err) / max(gt_range, eps)

    gt_std = float(np.std(gt_trace_comp))
    pred_std = float(np.std(gen_trace_comp))
    if gt_std < eps or pred_std < eps:
        corr = 0.0
    else:
        corr = float(np.corrcoef(gt_trace_comp, gen_trace_comp)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    corr_term = 0.5 * (np.clip(corr, -1.0, 1.0) + 1.0)

    rmse_term = float(np.exp(-nrmse_range))
    mae_term = float(np.exp(-nmae_rms))
    peak_term = float(np.exp(-peak_err_rel))
    scenario_score = float(
        100.0 * (0.40 * rmse_term + 0.30 * mae_term + 0.20 * peak_term + 0.10 * corr_term)
    )
    scenario_score = float(np.clip(scenario_score, 0.0, 100.0))

    return {
        "current_mse": current_mse,
        "current_mae": current_mae,
        "current_r2": current_r2,
        "ox_mse": float(mean_squared_error(np.array(gtox).flatten(), np.array(genox).flatten())),
        "red_mse": float(mean_squared_error(np.array(gtred).flatten(), np.array(genred).flatten())),
        "peak_pos_err": peak_pos_err,
        "peak_neg_err": peak_neg_err,
        "nrmse_range": nrmse_range,
        "nmae_rms": nmae_rms,
        "peak_err_rel": peak_err_rel,
        "corr": corr,
        "scenario_score": scenario_score,
    }


def _pad_param(values, fill_value: float, max_species: int) -> np.ndarray:
    out = np.full(max_species, fill_value, dtype=float)
    values = np.asarray(np.atleast_1d(values), dtype=float)
    n = min(max_species, values.shape[0])
    out[:n] = values[:n]
    return out


def _clip_index(index: int, size: int) -> int:
    if size <= 0:
        return 0
    return int(np.clip(index, 0, size - 1))


def _find_label_index(
    target: str,
    names: list[str],
    size: int,
    fallback_names: list[str],
    default_index: int,
) -> int:
    if size <= 0:
        return 0

    trimmed = names[:size]
    exact = {name: idx for idx, name in enumerate(trimmed)}
    if target in exact:
        return exact[target]

    lowered = {name.lower(): idx for idx, name in enumerate(trimmed)}
    if target.lower() in lowered:
        return lowered[target.lower()]

    for idx, name in enumerate(fallback_names):
        if name == target and idx < size:
            return idx

    return _clip_index(default_index, size)


def _build_conditioning_vector(
    flat_base_params: np.ndarray,
    n_tasks: int,
    n_stages: int,
    task_idx: int,
    stage_idx: int,
) -> np.ndarray:
    blocks = [flat_base_params]

    if n_tasks > 0:
        task_oh = np.zeros((n_tasks,), dtype=np.float32)
        task_oh[_clip_index(task_idx, n_tasks)] = 1.0
        blocks.append(task_oh)

    if n_stages > 0:
        stage_oh = np.zeros((n_stages,), dtype=np.float32)
        stage_oh[_clip_index(stage_idx, n_stages)] = 1.0
        blocks.append(stage_oh)

    return np.concatenate(blocks).astype(np.float32)


def run_classical_eval(
    name: str,
    score_key: str,
    E_t: np.ndarray,
    t_max: float,
    params: tuple[np.ndarray, ...],
    model: VectorFieldNet,
    norm: tuple[jax.Array, ...],
    key: jax.Array,
    nx: int,
    max_species: int,
    target_len: int,
    n_tasks: int,
    n_stages: int,
    phys_dim_core: int,
    param_mask_features: bool,
    signal_channels: int,
    task_names: list[str],
    stage_names: list[str],
    output_dir: Path,
):
    D_ox, D_red, C_ox, C_red, E0, k0, alpha = params
    x_mean, x_std, e_mean, e_std, p_mean, p_std = norm

    _, C_ox_hist, C_red_hist, _, _, E_hist_vis, I_hist_vis = simulate_electrochem(
        E_array=E_t,
        t_max=t_max,
        D_ox=D_ox,
        D_red=D_red,
        C_bulk_ox=C_ox,
        C_bulk_red=C_red,
        E0=E0,
        k0=k0,
        alpha=alpha,
        nx=nx,
        save_every=0,
    )

    gt_ox = C_ox_hist[-1].flatten()
    gt_red = C_red_hist[-1].flatten()

    orig_indices = np.linspace(0.0, 1.0, len(E_hist_vis))
    target_indices = np.linspace(0.0, 1.0, target_len)
    e_signal = np.interp(target_indices, orig_indices, E_hist_vis)
    gt_i = np.interp(target_indices, orig_indices, I_hist_vis)

    D_ox_pad = _pad_param(D_ox, 1e-5, max_species)
    D_red_pad = _pad_param(D_red, 1e-5, max_species)
    C_ox_pad = _pad_param(C_ox, 0.0, max_species)
    C_red_pad = _pad_param(C_red, 0.0, max_species)
    E0_pad = _pad_param(E0, 0.0, max_species)
    k0_pad = _pad_param(k0, 0.01, max_species)
    alpha_pad = _pad_param(alpha, 0.5, max_species)

    flat_base_params = np.concatenate(
        [
            np.log(D_ox_pad),
            np.log(D_red_pad),
            C_ox_pad,
            C_red_pad,
            E0_pad,
            np.log(k0_pad),
            alpha_pad,
        ]
    )

    task_target = SCENARIO_TASK_MAP.get(score_key, TASK_NAMES[0])
    stage_target = SCENARIO_STAGE_MAP.get(score_key, STAGE_NAMES[-1])
    task_idx = _find_label_index(task_target, task_names, n_tasks, TASK_NAMES, default_index=0)
    stage_idx = _find_label_index(
        stage_target,
        stage_names,
        n_stages,
        STAGE_NAMES,
        default_index=max(0, n_stages - 1),
    )

    full_params_core = _build_conditioning_vector(
        flat_base_params=flat_base_params,
        n_tasks=n_tasks,
        n_stages=n_stages,
        task_idx=task_idx,
        stage_idx=stage_idx,
    )

    if full_params_core.shape[0] != int(p_mean.shape[0]) or full_params_core.shape[0] != phys_dim_core:
        raise ValueError(
            "Conditioning dim mismatch during eval: "
            f"constructed={full_params_core.shape[0]}, expected_core={phys_dim_core}, "
            f"normalizer={int(p_mean.shape[0])}."
        )

    e_normalized_base = (jnp.array([e_signal]) - e_mean) / e_std
    p_normalized_core = (jnp.array([full_params_core]) - p_mean) / p_std
    e_normalized = _build_signal_input(
        signal_norm=e_normalized_base,
        signal_mask=None,
        signal_channels=signal_channels,
    )
    p_normalized = _build_param_input(
        params_norm=p_normalized_core,
        param_mask=None,
        append_mask_features=param_mask_features,
    )

    state_dim = int(x_mean.shape[0])
    x0 = jax.random.normal(key, (1, state_dim))
    x_generated = integrate_flow(model, x0, e_normalized, p_normalized, n_steps=100)

    gen_x = np.array((x_generated[0] * x_std) + x_mean)

    species_state_dim = nx * max_species
    gen_ox = gen_x[:species_state_dim].reshape(max_species, nx)
    gen_red = gen_x[species_state_dim : 2 * species_state_dim].reshape(max_species, nx)
    gen_current = gen_x[2 * species_state_dim :]

    gt_ox_rs = gt_ox.reshape(max_species, nx)
    gt_red_rs = gt_red.reshape(max_species, nx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Surrogate Evaluation: {name}", fontsize=14)

    ax_states = axes[0]
    ax_states.plot(gen_ox[0], label="Pred Ox", color="blue")
    ax_states.plot(gen_red[0], label="Pred Red", color="red")
    ax_states.plot(gt_ox_rs[0], label="GT Ox", color="darkblue", linestyle="--")
    ax_states.plot(gt_red_rs[0], label="GT Red", color="darkred", linestyle="--")

    if max_species > 1 and C_ox_pad[1] > 0.0:
        ax_states.plot(gen_ox[1], label="Pred Ox (sp2)", color="cyan")
        ax_states.plot(gen_red[1], label="Pred Red (sp2)", color="magenta")
        ax_states.plot(gt_ox_rs[1], label="GT Ox (sp2)", color="darkcyan", linestyle="--")
        ax_states.plot(gt_red_rs[1], label="GT Red (sp2)", color="darkmagenta", linestyle="--")

    ax_states.set_title("Concentration Profiles")
    ax_states.legend()

    ax_iv = axes[1]
    ax_iv.plot(e_signal, gen_current, label="Pred", color="green")
    ax_iv.plot(e_signal, gt_i, label="GT", color="green", linestyle="--")
    ax_iv.set_xlabel("Potential (V)")
    ax_iv.set_ylabel("Current")
    ax_iv.set_title("I-V Curve")
    ax_iv.legend()

    ax_it = axes[2]
    times = np.linspace(0.0, t_max, target_len)
    ax_it.plot(times, gen_current, label="Pred", color="green")
    ax_it.plot(times, gt_i, label="GT", color="green", linestyle="--")
    ax_it.set_xlabel("Time (s)")
    ax_it.set_ylabel("Current")
    ax_it.set_title("I-t Curve")
    ax_it.legend()

    plt.tight_layout()
    out_png = output_dir / f"eval_{name.replace(' ', '_').lower()}.png"
    plt.savefig(out_png)
    plt.close()

    metrics = calculate_metrics(gt_i, gen_current, gt_ox_rs, gen_ox, gt_red_rs, gen_red)
    print(f"[{name}] R2={metrics['current_r2']:.4f} MSE={metrics['current_mse']:.3e}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained flow-matching surrogate on classical scenarios")
    parser.add_argument("--checkpoint", type=str, default="/tmp/ecsfm/surrogate_model.eqx", help="Path to .eqx model checkpoint")
    parser.add_argument("--dataset", "--data-path", dest="data_path", type=str, default=None, help="Dataset path used to rebuild normalizers if artifact normalizers are missing")
    parser.add_argument("--normalizers", type=str, default=None, help="Path to saved normalizers NPZ (defaults to checkpoint dir)")
    parser.add_argument("--meta", type=str, default=None, help="Path to model metadata JSON (defaults to checkpoint dir)")
    parser.add_argument("--output-dir", type=str, default="/tmp/ecsfm", help="Directory for plots and scorecard")
    parser.add_argument("--hidden-size", type=int, default=None, help="Model hidden size (defaults to metadata or 128)")
    parser.add_argument("--depth", type=int, default=None, help="Model MLP depth (defaults to metadata or 3)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split for dataset-derived normalizers")
    return parser.parse_args()


def _normalize_label_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _resolve_model_geometry(norm: tuple[jax.Array, ...], model_meta: dict[str, Any]) -> dict[str, int | list[str]]:
    x_mean, _, e_mean, _, p_mean, _ = norm

    state_dim = int(x_mean.shape[0])
    target_len = int(e_mean.shape[0])
    phys_norm_dim = int(p_mean.shape[0])
    signal_channels = int(model_meta.get("signal_channels", 1)) if model_meta else 1
    if signal_channels not in (1, 2):
        raise ValueError(f"Unsupported signal_channels={signal_channels}; expected 1 or 2.")

    max_species = int(model_meta.get("max_species", 0)) if model_meta else 0
    phys_dim_base = int(model_meta.get("phys_dim_base", 0)) if model_meta else 0
    phys_dim_core = int(model_meta.get("phys_dim_core", phys_norm_dim)) if model_meta else phys_norm_dim
    param_mask_features = bool(model_meta.get("param_mask_features", False)) if model_meta else False
    phys_dim = int(model_meta.get("phys_dim", phys_dim_core * (2 if param_mask_features else 1))) if model_meta else phys_dim_core

    if max_species <= 0:
        if phys_dim_base > 0:
            if phys_dim_base % 7 != 0:
                raise ValueError(f"Invalid phys_dim_base={phys_dim_base}; expected divisibility by 7")
            max_species = phys_dim_base // 7
        elif phys_dim_core % 7 == 0:
            max_species = phys_dim_core // 7
        else:
            raise ValueError(
                "Unable to infer max_species from normalizers. "
                "Provide --meta with max_species/phys_dim_base."
            )

    if phys_dim_base <= 0:
        phys_dim_base = 7 * max_species

    if phys_dim_base != 7 * max_species:
        raise ValueError(
            f"Inconsistent metadata: phys_dim_base={phys_dim_base}, max_species={max_species}"
        )
    if phys_dim_base > phys_dim_core:
        raise ValueError(
            f"Invalid dimensions: phys_dim_base={phys_dim_base} exceeds phys_dim_core={phys_dim_core}"
        )

    if phys_norm_dim != phys_dim_core:
        raise ValueError(
            f"Normalizer dimension mismatch: expected phys_dim_core={phys_dim_core}, got {phys_norm_dim}"
        )

    expected_phys_dim = phys_dim_core * (2 if param_mask_features else 1)
    if expected_phys_dim != phys_dim:
        raise ValueError(
            f"Conditioning dim mismatch: expected phys_dim={expected_phys_dim} from metadata, got {phys_dim}"
        )

    extra_dim = phys_dim_core - phys_dim_base
    n_tasks = int(model_meta.get("n_tasks", 0)) if model_meta else 0
    n_stages = int(model_meta.get("n_stages", 0)) if model_meta else 0

    if n_tasks < 0 or n_stages < 0:
        raise ValueError("n_tasks and n_stages must be non-negative")

    if n_tasks == 0 and n_stages == 0:
        if extra_dim == 0:
            pass
        elif extra_dim == 1:
            n_tasks = 1
        else:
            n_stages = 1
            n_tasks = extra_dim - 1
    elif n_tasks == 0:
        n_tasks = extra_dim - n_stages
    elif n_stages == 0:
        n_stages = extra_dim - n_tasks

    if n_tasks < 0 or n_stages < 0 or (n_tasks + n_stages) != extra_dim:
        raise ValueError(
            "Conditioning metadata mismatch: "
            f"phys_dim={phys_dim}, phys_dim_base={phys_dim_base}, "
            f"n_tasks={n_tasks}, n_stages={n_stages}."
        )

    task_names = _normalize_label_list(model_meta.get("task_names")) if model_meta else []
    stage_names = _normalize_label_list(model_meta.get("stage_names")) if model_meta else []

    if n_tasks == 0:
        task_names = []
    else:
        task_names = task_names[:n_tasks]
        for fallback in TASK_NAMES:
            if len(task_names) >= n_tasks:
                break
            if fallback not in task_names:
                task_names.append(fallback)
        if len(task_names) < n_tasks:
            task_names.extend(f"task_{i}" for i in range(len(task_names), n_tasks))

    if n_stages == 0:
        stage_names = []
    else:
        stage_names = stage_names[:n_stages]
        for fallback in STAGE_NAMES:
            if len(stage_names) >= n_stages:
                break
            if fallback not in stage_names:
                stage_names.append(fallback)
        if len(stage_names) < n_stages:
            stage_names.extend(f"stage_{i}" for i in range(len(stage_names), n_stages))

    spatial_total = state_dim - target_len
    denom = 2 * max_species
    if spatial_total <= 0 or spatial_total % denom != 0:
        raise ValueError(
            f"Cannot infer nx from state_dim={state_dim}, target_len={target_len}, max_species={max_species}"
        )

    nx = spatial_total // denom
    return {
        "state_dim": state_dim,
        "target_len": target_len,
        "phys_dim": phys_dim,
        "phys_dim_core": phys_dim_core,
        "phys_dim_base": phys_dim_base,
        "max_species": max_species,
        "nx": nx,
        "n_tasks": n_tasks,
        "n_stages": n_stages,
        "param_mask_features": param_mask_features,
        "signal_channels": signal_channels,
        "task_names": task_names,
        "stage_names": stage_names,
    }


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    default_norm_path = checkpoint_path.parent / NORMALIZERS_FILENAME
    default_meta_path = checkpoint_path.parent / MODEL_META_FILENAME
    normalizers_path = Path(args.normalizers) if args.normalizers else default_norm_path
    meta_path = Path(args.meta) if args.meta else default_meta_path

    norm = None
    model_meta: dict[str, Any] = {}

    if normalizers_path.exists():
        norm = load_saved_normalizers(normalizers_path)
        print(f"Loaded normalizers from {normalizers_path}")

    if meta_path.exists():
        model_meta = load_model_metadata(meta_path)
        print(f"Loaded model metadata from {meta_path}")

    if norm is None:
        if args.data_path is None:
            raise ValueError(
                "No saved normalizers found and no --dataset provided. "
                "Provide a dataset path or pass --normalizers."
            )
        print("Recomputing normalizers from dataset...")
        norm = get_normalizers_from_dataset(args.data_path, seed=args.seed, val_split=args.val_split)

    geometry = _resolve_model_geometry(norm, model_meta)

    hidden_size = args.hidden_size if args.hidden_size is not None else int(model_meta.get("hidden_size", 128))
    depth = args.depth if args.depth is not None else int(model_meta.get("depth", 3))
    cond_dim = int(model_meta.get("cond_dim", 32))

    key = jax.random.PRNGKey(args.seed)
    _, subkey = jax.random.split(key)

    model = VectorFieldNet(
        state_dim=int(geometry["state_dim"]),
        hidden_size=hidden_size,
        depth=depth,
        cond_dim=cond_dim,
        phys_dim=int(geometry["phys_dim"]),
        signal_channels=int(geometry["signal_channels"]),
        key=subkey,
    )
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)

    print(
        "Loaded model with dimensions: "
        f"state_dim={geometry['state_dim']}, max_species={geometry['max_species']}, "
        f"nx={geometry['nx']}, target_len={geometry['target_len']}, phys_dim={geometry['phys_dim']}, "
        f"phys_dim_core={geometry['phys_dim_core']}, phys_dim_base={geometry['phys_dim_base']}, "
        f"signal_channels={geometry['signal_channels']}, param_mask_features={geometry['param_mask_features']}, "
        f"tasks={geometry['n_tasks']}, "
        f"stages={geometry['n_stages']}, hidden_size={hidden_size}, depth={depth}"
    )

    def get_base_params() -> tuple[np.ndarray, ...]:
        max_species = int(geometry["max_species"])
        D_ox = np.ones(max_species) * 1e-5
        D_red = np.ones(max_species) * 1e-5
        C_ox = np.zeros(max_species)
        C_red = np.zeros(max_species)
        E0 = np.zeros(max_species)
        k0 = np.ones(max_species) * 0.01
        alpha = np.ones(max_species) * 0.5
        return D_ox, D_red, C_ox, C_red, E0, k0, alpha

    results_scorecard = {}

    D_ox, D_red, C_ox, C_red, E0, k0, alpha = get_base_params()
    C_ox[0] = 1.0
    E0[0] = 0.0
    k0[0] = 0.1
    params_ferro = (D_ox, D_red, C_ox, C_red, E0, k0, alpha)
    E_cv, t_cv = get_cv_waveform(0.5, -0.5, 0.1)

    key, subkey = jax.random.split(key)
    results_scorecard["Ferrocene_Reversible_CV"] = run_classical_eval(
        "Ferrocene Reversible CV",
        "Ferrocene_Reversible_CV",
        E_cv,
        t_cv,
        params_ferro,
        model,
        norm,
        subkey,
        int(geometry["nx"]),
        int(geometry["max_species"]),
        int(geometry["target_len"]),
        int(geometry["n_tasks"]),
        int(geometry["n_stages"]),
        int(geometry["phys_dim_core"]),
        bool(geometry["param_mask_features"]),
        int(geometry["signal_channels"]),
        list(geometry["task_names"]),
        list(geometry["stage_names"]),
        output_dir,
    )

    D_ox, D_red, C_ox, C_red, E0, k0, alpha = get_base_params()
    C_ox[0] = 1.0
    E0[0] = 0.3
    k0[0] = 0.01
    if int(geometry["max_species"]) > 1:
        C_ox[1] = 0.8
        E0[1] = -0.2
        k0[1] = 0.01
    params_multi = (D_ox, D_red, C_ox, C_red, E0, k0, alpha)
    E_cv2, t_cv2 = get_cv_waveform(0.6, -0.6, 0.1)

    key, subkey = jax.random.split(key)
    results_scorecard["Multi_Species_CV"] = run_classical_eval(
        "Multi Species CV",
        "Multi_Species_CV",
        E_cv2,
        t_cv2,
        params_multi,
        model,
        norm,
        subkey,
        int(geometry["nx"]),
        int(geometry["max_species"]),
        int(geometry["target_len"]),
        int(geometry["n_tasks"]),
        int(geometry["n_stages"]),
        int(geometry["phys_dim_core"]),
        bool(geometry["param_mask_features"]),
        int(geometry["signal_channels"]),
        list(geometry["task_names"]),
        list(geometry["stage_names"]),
        output_dir,
    )

    E_swv, t_swv = get_swv_waveform(0.5, -0.5, 0.01, 0.05, 15.0)
    key, subkey = jax.random.split(key)
    results_scorecard["SWV_Ferrocene"] = run_classical_eval(
        "SWV Ferrocene",
        "SWV_Ferrocene",
        E_swv,
        t_swv,
        params_ferro,
        model,
        norm,
        subkey,
        int(geometry["nx"]),
        int(geometry["max_species"]),
        int(geometry["target_len"]),
        int(geometry["n_tasks"]),
        int(geometry["n_stages"]),
        int(geometry["phys_dim_core"]),
        bool(geometry["param_mask_features"]),
        int(geometry["signal_channels"]),
        list(geometry["task_names"]),
        list(geometry["stage_names"]),
        output_dir,
    )

    D_ox, D_red, C_ox, C_red, E0, k0, alpha = get_base_params()
    C_ox[0] = 1.0
    E0[0] = 0.1
    k0[0] = 0.005
    params_slow = (D_ox, D_red, C_ox, C_red, E0, k0, alpha)

    E_ca, t_ca = get_ca_waveform(0.5, -0.2, 2.0, 0.5)
    key, subkey = jax.random.split(key)
    results_scorecard["CA_Step"] = run_classical_eval(
        "CA Step",
        "CA_Step",
        E_ca,
        t_ca,
        params_slow,
        model,
        norm,
        subkey,
        int(geometry["nx"]),
        int(geometry["max_species"]),
        int(geometry["target_len"]),
        int(geometry["n_tasks"]),
        int(geometry["n_stages"]),
        int(geometry["phys_dim_core"]),
        bool(geometry["param_mask_features"]),
        int(geometry["signal_channels"]),
        list(geometry["task_names"]),
        list(geometry["stage_names"]),
        output_dir,
    )

    E_eis, t_eis = get_eis_waveform(0.0, 0.05, 10.0, 2.0)
    key, subkey = jax.random.split(key)
    results_scorecard["EIS_Sine"] = run_classical_eval(
        "EIS Sine",
        "EIS_Sine",
        E_eis,
        t_eis,
        params_slow,
        model,
        norm,
        subkey,
        int(geometry["nx"]),
        int(geometry["max_species"]),
        int(geometry["target_len"]),
        int(geometry["n_tasks"]),
        int(geometry["n_stages"]),
        int(geometry["phys_dim_core"]),
        bool(geometry["param_mask_features"]),
        int(geometry["signal_channels"]),
        list(geometry["task_names"]),
        list(geometry["stage_names"]),
        output_dir,
    )

    scenario_scores = [res["scenario_score"] for res in results_scorecard.values()]
    final_score = float(np.mean(scenario_scores))
    final_score = max(0.0, min(100.0, final_score))

    avg_r2 = float(np.mean([res["current_r2"] for res in results_scorecard.values()]))
    legacy_score = max(0.0, min(100.0, avg_r2 * 100.0))

    results_scorecard["Final_Score_Out_Of_100"] = round(final_score, 2)
    results_scorecard["Legacy_R2_Score_Out_Of_100"] = round(legacy_score, 2)
    results_scorecard["Scoring_Method"] = (
        "Final score is the mean of per-scenario robust scores "
        "(weights: nRMSE-range 0.40, nMAE-rms 0.30, peak-relative 0.20, correlation 0.10)."
    )

    scorecard_path = output_dir / "evaluation_scorecard.json"
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(results_scorecard, f, indent=4)

    print("\n" + "=" * 50)
    print(f"FINAL EVALUATION SCORECARD ({results_scorecard['Final_Score_Out_Of_100']}/100)")
    print("=" * 50)
    print(json.dumps(results_scorecard, indent=4))
    print(f"Saved scorecard to {scorecard_path}")


if __name__ == "__main__":
    main()
