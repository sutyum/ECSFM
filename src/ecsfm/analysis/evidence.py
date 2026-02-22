from __future__ import annotations

from math import log
from typing import Any

import jax
import numpy as np

from ecsfm.data.generate import (
    AUGMENTATION_NAMES,
    STAGE_NAMES,
    TASK_NAMES,
    generate_multi_species_dataset,
)
from ecsfm.sim.cv import simulate_cv


def _safe_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)))


def _cv_peak_metrics(E_hist: np.ndarray, I_hist: np.ndarray) -> dict[str, float]:
    E_hist = np.asarray(E_hist, dtype=float)
    I_hist = np.asarray(I_hist, dtype=float)
    if E_hist.shape != I_hist.shape:
        raise ValueError(f"E_hist and I_hist must share shape, got {E_hist.shape} and {I_hist.shape}")
    if E_hist.ndim != 1:
        raise ValueError(f"E_hist and I_hist must be 1D, got {E_hist.shape}")
    if len(E_hist) < 4:
        raise ValueError("Need at least 4 time points to extract CV peaks")

    mid = len(E_hist) // 2
    idx_c = int(np.argmin(I_hist[:mid]))
    idx_a = int(mid + np.argmax(I_hist[mid:]))
    ep_c = float(E_hist[idx_c])
    ep_a = float(E_hist[idx_a])
    ip_c = float(-I_hist[idx_c])
    ip_a = float(I_hist[idx_a])
    return {
        "ep_c": ep_c,
        "ep_a": ep_a,
        "delta_ep": ep_a - ep_c,
        "ip_c": ip_c,
        "ip_a": ip_a,
    }


def cv_trace_sweep_by_k0(
    k0_values: np.ndarray,
    *,
    D: float = 1e-5,
    concentration: float = 1.0,
    scan_rate: float = 1.0,
    nx: int = 36,
    E0: float = 0.0,
    alpha: float = 0.5,
    E_start: float = 0.5,
    E_vertex: float = -0.5,
) -> dict[str, Any]:
    k0_values = np.asarray(k0_values, dtype=float)
    traces: list[dict[str, Any]] = []
    summary_rows: list[dict[str, float]] = []

    for k0 in k0_values:
        _, _, _, E_hist, I_hist, _ = simulate_cv(
            D_ox=D,
            D_red=D,
            C_bulk_ox=concentration,
            C_bulk_red=0.0,
            E0=E0,
            k0=float(k0),
            alpha=alpha,
            scan_rate=scan_rate,
            E_start=E_start,
            E_vertex=E_vertex,
            nx=nx,
            save_every=0,
        )
        metrics = _cv_peak_metrics(E_hist, I_hist)
        traces.append({"k0": float(k0), "E_hist": np.asarray(E_hist), "I_hist": np.asarray(I_hist), **metrics})
        summary_rows.append({"k0": float(k0), **metrics})

    return {"k0_values": k0_values, "traces": traces, "summary": summary_rows}


def cv_trace_sweep_by_scan_rate(
    scan_rates: np.ndarray,
    *,
    D: float = 1e-5,
    concentration: float = 1.0,
    k0: float = 1e-3,
    nx: int = 36,
    E0: float = 0.0,
    alpha: float = 0.5,
    E_start: float = 0.5,
    E_vertex: float = -0.5,
) -> dict[str, Any]:
    scan_rates = np.asarray(scan_rates, dtype=float)
    traces: list[dict[str, Any]] = []
    summary_rows: list[dict[str, float]] = []

    for scan_rate in scan_rates:
        _, _, _, E_hist, I_hist, _ = simulate_cv(
            D_ox=D,
            D_red=D,
            C_bulk_ox=concentration,
            C_bulk_red=0.0,
            E0=E0,
            k0=k0,
            alpha=alpha,
            scan_rate=float(scan_rate),
            E_start=E_start,
            E_vertex=E_vertex,
            nx=nx,
            save_every=0,
        )
        metrics = _cv_peak_metrics(E_hist, I_hist)
        traces.append(
            {"scan_rate": float(scan_rate), "E_hist": np.asarray(E_hist), "I_hist": np.asarray(I_hist), **metrics}
        )
        summary_rows.append({"scan_rate": float(scan_rate), **metrics})

    return {"scan_rates": scan_rates, "traces": traces, "summary": summary_rows}


def simulator_convergence_study(
    nx_values: np.ndarray,
    *,
    reference_nx: int = 72,
    D: float = 1e-5,
    concentration: float = 1.0,
    k0: float = 1e-2,
    scan_rate: float = 1.0,
    E0: float = 0.0,
    alpha: float = 0.5,
    E_start: float = 0.5,
    E_vertex: float = -0.5,
) -> dict[str, Any]:
    nx_values = np.asarray(nx_values, dtype=int)
    if np.any(nx_values < 2):
        raise ValueError("All nx values must be >= 2")
    if reference_nx < 2:
        raise ValueError("reference_nx must be >= 2")

    _, _, _, ref_E, ref_I, _ = simulate_cv(
        D_ox=D,
        D_red=D,
        C_bulk_ox=concentration,
        C_bulk_red=0.0,
        E0=E0,
        k0=k0,
        alpha=alpha,
        scan_rate=scan_rate,
        E_start=E_start,
        E_vertex=E_vertex,
        nx=reference_nx,
        save_every=0,
    )
    ref_E = np.asarray(ref_E, dtype=float)
    ref_I = np.asarray(ref_I, dtype=float)
    ref_metrics = _cv_peak_metrics(ref_E, ref_I)
    ref_peak = float(np.max(np.abs(ref_I)))
    ref_t = np.linspace(0.0, 1.0, len(ref_I))

    rows: list[dict[str, float]] = []
    for nx in nx_values:
        _, _, _, E_hist, I_hist, _ = simulate_cv(
            D_ox=D,
            D_red=D,
            C_bulk_ox=concentration,
            C_bulk_red=0.0,
            E0=E0,
            k0=k0,
            alpha=alpha,
            scan_rate=scan_rate,
            E_start=E_start,
            E_vertex=E_vertex,
            nx=int(nx),
            save_every=0,
        )
        E_hist = np.asarray(E_hist, dtype=float)
        I_hist = np.asarray(I_hist, dtype=float)
        t = np.linspace(0.0, 1.0, len(I_hist))
        I_interp = np.interp(ref_t, t, I_hist)
        rmse = float(np.sqrt(np.mean((I_interp - ref_I) ** 2)))
        nrmse = rmse / (ref_peak + 1e-12)

        metrics = _cv_peak_metrics(E_hist, I_hist)
        peak = float(np.max(np.abs(I_hist)))
        rows.append(
            {
                "nx": int(nx),
                "nrmse_vs_ref": nrmse,
                "delta_ep_abs_error_vs_ref": abs(metrics["delta_ep"] - ref_metrics["delta_ep"]),
                "peak_abs_rel_error_vs_ref": abs(peak - ref_peak) / (ref_peak + 1e-12),
                "delta_ep": metrics["delta_ep"],
                "peak_current": peak,
            }
        )

    return {
        "reference_nx": int(reference_nx),
        "reference_metrics": {
            "delta_ep": ref_metrics["delta_ep"],
            "peak_current": ref_peak,
            "num_points": int(len(ref_I)),
        },
        "rows": rows,
    }


def _infer_aug_pairs(task_id: np.ndarray, stage_id: np.ndarray, aug_id: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for idx in np.where(aug_id > 0)[0]:
        if idx == 0:
            continue
        base_idx = idx - 1
        if aug_id[base_idx] != 0:
            continue
        if task_id[idx] != task_id[base_idx] or stage_id[idx] != stage_id[base_idx]:
            continue
        pairs.append((base_idx, idx))
    return pairs


def _param_ranges(params: np.ndarray) -> dict[str, tuple[float, float]]:
    m = params.shape[1] // 7
    d_ox = np.exp(params[:, 0:m])
    d_red = np.exp(params[:, m : 2 * m])
    c_ox = params[:, 2 * m : 3 * m]
    c_red = params[:, 3 * m : 4 * m]
    e0 = params[:, 4 * m : 5 * m]
    k0 = np.exp(params[:, 5 * m : 6 * m])
    alpha = params[:, 6 * m : 7 * m]

    def rng(arr: np.ndarray) -> tuple[float, float]:
        return float(np.min(arr)), float(np.max(arr))

    return {
        "D_ox": rng(d_ox),
        "D_red": rng(d_red),
        "C_ox": rng(c_ox),
        "C_red": rng(c_red),
        "E0": rng(e0),
        "k0": rng(k0),
        "alpha": rng(alpha),
    }


def dataset_recipe_audit(
    *,
    recipes: list[str] | None = None,
    n_samples: int = 512,
    max_species: int = 5,
    nx: int = 24,
    seed: int = 2026,
    invariant_fraction: float = 0.35,
    stage_proportions: tuple[float, float, float] = (0.4, 0.35, 0.25),
    target_sig_len: int = 200,
    sim_steps: int = 128,
    device_batch_size: int = 64,
) -> dict[str, Any]:
    recipes = recipes or ["baseline_random", "curriculum_multitask", "stress_mixture"]
    audit_rows: list[dict[str, Any]] = []

    for offset, recipe in enumerate(recipes):
        key = jax.random.PRNGKey(seed + offset * 1009)
        c_ox, c_red, currents, signals, params, task_id, stage_id, aug_id = (
            np.asarray(item)
            for item in generate_multi_species_dataset(
                n_samples=n_samples,
                key=key,
                max_species=max_species,
                nx=nx,
                max_workers=1,
                backend="gpu_batch",
                progress=False,
                sim_steps=sim_steps,
                device_batch_size=device_batch_size,
                recipe=recipe,
                stage_proportions=stage_proportions,
                include_invariant_pairs=True,
                invariant_fraction=invariant_fraction,
                target_sig_len=target_sig_len,
            )
        )

        task_counts = np.bincount(task_id, minlength=len(TASK_NAMES)).astype(int)
        stage_counts = np.bincount(stage_id, minlength=len(STAGE_NAMES)).astype(int)
        aug_counts = np.bincount(aug_id, minlength=len(AUGMENTATION_NAMES)).astype(int)

        pairs = _infer_aug_pairs(task_id, stage_id, aug_id)
        permute_success = 0
        permute_total = 0
        scale_total = 0
        scale_log_ratio_errors: list[float] = []

        m = params.shape[1] // 7
        for base_idx, aug_idx in pairs:
            aug_kind = int(aug_id[aug_idx])
            if aug_kind == 1:
                permute_total += 1
                same_current = np.allclose(currents[base_idx], currents[aug_idx], atol=1e-6, rtol=1e-6)
                same_signal = np.allclose(signals[base_idx], signals[aug_idx], atol=1e-6, rtol=1e-6)
                if same_current and same_signal:
                    permute_success += 1
            elif aug_kind == 2:
                scale_total += 1
                base_total_conc = np.sum(params[base_idx, 2 * m : 4 * m]) + 1e-12
                aug_total_conc = np.sum(params[aug_idx, 2 * m : 4 * m]) + 1e-12
                base_curr_norm = np.linalg.norm(currents[base_idx]) + 1e-12
                aug_curr_norm = np.linalg.norm(currents[aug_idx]) + 1e-12
                ratio_conc = aug_total_conc / base_total_conc
                ratio_curr = aug_curr_norm / base_curr_norm
                scale_log_ratio_errors.append(abs(log(ratio_curr / ratio_conc)))

        task_diversity = float(np.count_nonzero(task_counts))
        stage_diversity = float(np.count_nonzero(stage_counts))
        permute_pass_frac = float(permute_success / permute_total) if permute_total > 0 else 1.0
        scale_log_ratio_mae = float(np.mean(scale_log_ratio_errors)) if scale_log_ratio_errors else 0.0
        scale_within_20pct = (
            float(np.mean(np.asarray(scale_log_ratio_errors) < 0.2)) if scale_log_ratio_errors else 1.0
        )

        max_current = np.max(np.abs(currents), axis=1)
        mean_signal_span = float(np.mean(np.ptp(signals, axis=1)))

        audit_rows.append(
            {
                "recipe": recipe,
                "total_rows": int(len(task_id)),
                "augmentation_fraction": float(np.mean(aug_id > 0)),
                "task_counts": task_counts.tolist(),
                "stage_counts": stage_counts.tolist(),
                "augmentation_counts": aug_counts.tolist(),
                "task_entropy": _safe_entropy(task_counts),
                "stage_entropy": _safe_entropy(stage_counts),
                "task_diversity": task_diversity,
                "stage_diversity": stage_diversity,
                "pair_count": int(len(pairs)),
                "permute_pass_fraction": permute_pass_frac,
                "scale_log_ratio_mae": scale_log_ratio_mae,
                "scale_within_20pct_fraction": scale_within_20pct,
                "max_current_mA_mean": float(np.mean(max_current)),
                "max_current_mA_std": float(np.std(max_current)),
                "signal_span_mean": mean_signal_span,
                "parameter_ranges": _param_ranges(params),
            }
        )

    leaderboard = sorted(
        audit_rows,
        key=lambda row: (
            -float(row["task_entropy"]),
            -float(row["stage_entropy"]),
            -float(row["scale_within_20pct_fraction"]),
            -float(row["permute_pass_fraction"]),
        ),
    )

    return {
        "recipes": recipes,
        "task_names": TASK_NAMES,
        "stage_names": STAGE_NAMES,
        "augmentation_names": AUGMENTATION_NAMES,
        "rows": audit_rows,
        "recipe_ranking": [row["recipe"] for row in leaderboard],
    }
