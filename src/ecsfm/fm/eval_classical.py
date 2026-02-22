import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ecsfm.data.generate import get_ca_waveform, get_cv_waveform, get_eis_waveform, get_swv_waveform
from ecsfm.fm.model import VectorFieldNet
from ecsfm.fm.train import integrate_flow, load_dataset
from ecsfm.sim.experiment import simulate_electrochem

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_normalizers(data_path: str, seed: int = 42, val_split: float = 0.2):
    c_ox, c_red, curr, sigs, params = load_dataset(data_path)
    if c_ox.shape[0] < 2:
        raise ValueError("Need at least 2 samples to compute stable normalizers")

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

    metrics = {
        "current_mse": float(mean_squared_error(gt_trace_comp, gen_trace_comp)),
        "current_mae": float(mean_absolute_error(gt_trace_comp, gen_trace_comp)),
        "current_r2": float(r2_score(gt_trace_comp, gen_trace_comp)),
        "ox_mse": float(mean_squared_error(np.array(gtox).flatten(), np.array(genox).flatten())),
        "red_mse": float(mean_squared_error(np.array(gtred).flatten(), np.array(genred).flatten())),
        "peak_pos_err": float(abs(np.max(gt_trace_comp) - np.max(gen_trace_comp))),
        "peak_neg_err": float(abs(np.min(gt_trace_comp) - np.min(gen_trace_comp))),
    }
    return metrics


def _pad_param(values, fill_value: float, max_species: int) -> np.ndarray:
    out = np.full(max_species, fill_value, dtype=float)
    values = np.asarray(np.atleast_1d(values), dtype=float)
    n = min(max_species, values.shape[0])
    out[:n] = values[:n]
    return out


def run_classical_eval(
    name: str,
    E_t: np.ndarray,
    t_max: float,
    params: tuple[np.ndarray, ...],
    model: VectorFieldNet,
    norm: tuple[jax.Array, ...],
    key: jax.Array,
    nx: int,
    max_species: int,
    target_len: int,
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

    flat_params = np.concatenate(
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

    e_normalized = (jnp.array([e_signal]) - e_mean) / e_std
    p_normalized = (jnp.array([flat_params]) - p_mean) / p_std

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
    parser.add_argument("--dataset", "--data-path", dest="data_path", type=str, default="/tmp/ecsfm/dataset_massive", help="Training dataset path used for normalizers")
    parser.add_argument("--output-dir", type=str, default="/tmp/ecsfm", help="Directory for plots and scorecard")
    parser.add_argument("--hidden-size", type=int, default=128, help="Model hidden size")
    parser.add_argument("--depth", type=int, default=3, help="Model MLP depth")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split used for normalizer recreation")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print("Loading normalizers...")
    norm = get_normalizers(args.data_path, seed=args.seed, val_split=args.val_split)
    x_mean, _, e_mean, _, p_mean, _ = norm

    phys_dim = int(p_mean.shape[0])
    if phys_dim % 7 != 0:
        raise ValueError(f"Invalid phys_dim={phys_dim}; must be divisible by 7")
    max_species = phys_dim // 7

    state_dim = int(x_mean.shape[0])
    target_len = int(e_mean.shape[0])
    spatial_total = state_dim - target_len
    if spatial_total <= 0:
        raise ValueError(
            f"Invalid dimensions: state_dim={state_dim}, target_len={target_len}"
        )

    denom = 2 * max_species
    if spatial_total % denom != 0:
        raise ValueError(
            f"Cannot infer nx: spatial_total={spatial_total} not divisible by {denom}"
        )
    nx = spatial_total // denom

    key = jax.random.PRNGKey(args.seed)
    _, subkey = jax.random.split(key)

    model = VectorFieldNet(
        state_dim=state_dim,
        hidden_size=args.hidden_size,
        depth=args.depth,
        cond_dim=32,
        phys_dim=phys_dim,
        key=subkey,
    )
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)

    print(
        f"Loaded model. state_dim={state_dim}, max_species={max_species}, nx={nx}, target_len={target_len}"
    )

    def get_base_params() -> tuple[np.ndarray, ...]:
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
        E_cv,
        t_cv,
        params_ferro,
        model,
        norm,
        subkey,
        nx,
        max_species,
        target_len,
        output_dir,
    )

    D_ox, D_red, C_ox, C_red, E0, k0, alpha = get_base_params()
    C_ox[0] = 1.0
    E0[0] = 0.3
    k0[0] = 0.01
    if max_species > 1:
        C_ox[1] = 0.8
        E0[1] = -0.2
        k0[1] = 0.01
    params_multi = (D_ox, D_red, C_ox, C_red, E0, k0, alpha)
    E_cv2, t_cv2 = get_cv_waveform(0.6, -0.6, 0.1)

    key, subkey = jax.random.split(key)
    results_scorecard["Multi_Species_CV"] = run_classical_eval(
        "Multi Species CV",
        E_cv2,
        t_cv2,
        params_multi,
        model,
        norm,
        subkey,
        nx,
        max_species,
        target_len,
        output_dir,
    )

    E_swv, t_swv = get_swv_waveform(0.5, -0.5, 0.01, 0.05, 15.0)
    key, subkey = jax.random.split(key)
    results_scorecard["SWV_Ferrocene"] = run_classical_eval(
        "SWV Ferrocene",
        E_swv,
        t_swv,
        params_ferro,
        model,
        norm,
        subkey,
        nx,
        max_species,
        target_len,
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
        E_ca,
        t_ca,
        params_slow,
        model,
        norm,
        subkey,
        nx,
        max_species,
        target_len,
        output_dir,
    )

    E_eis, t_eis = get_eis_waveform(0.0, 0.05, 10.0, 2.0)
    key, subkey = jax.random.split(key)
    results_scorecard["EIS_Sine"] = run_classical_eval(
        "EIS Sine",
        E_eis,
        t_eis,
        params_slow,
        model,
        norm,
        subkey,
        nx,
        max_species,
        target_len,
        output_dir,
    )

    avg_r2 = float(np.mean([res["current_r2"] for res in results_scorecard.values()]))
    final_score = max(0.0, min(100.0, avg_r2 * 100.0))
    results_scorecard["Final_Score_Out_Of_100"] = round(final_score, 2)

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
