from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ecsfm.sim.multiphysics import (
    MultiPhysicsConfig,
    build_biofouling_protocol,
    estimate_impedance_from_trace,
    simulate_multiphysics_electrochem,
)


@dataclass(frozen=True)
class RandlesFit:
    rs_ohm: float
    rct_ohm: float
    cdl_f: float
    mean_rel_error: float


def _default_species() -> dict[str, jnp.ndarray]:
    return {
        "D_ox": jnp.asarray([1.0e-5], dtype=jnp.float32),
        "D_red": jnp.asarray([1.0e-5], dtype=jnp.float32),
        "C_bulk_ox": jnp.asarray([1.0], dtype=jnp.float32),
        "C_bulk_red": jnp.asarray([0.0], dtype=jnp.float32),
        "E0": jnp.asarray([0.0], dtype=jnp.float32),
        "k0": jnp.asarray([0.02], dtype=jnp.float32),
        "alpha": jnp.asarray([0.5], dtype=jnp.float32),
    }


def _simulate_single_tone(
    frequency_hz: float,
    *,
    initial_theta: float,
    t_window_s: float = 8.0,
    dt_wave_s: float = 1e-3,
    e_dc_v: float = -0.02,
    e_amp_v: float = 0.01,
    nx: int = 28,
    config: MultiPhysicsConfig | None = None,
) -> dict[str, float]:
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")
    if t_window_s <= 0:
        raise ValueError(f"t_window_s must be positive, got {t_window_s}")
    if dt_wave_s <= 0:
        raise ValueError(f"dt_wave_s must be positive, got {dt_wave_s}")

    t_wave = np.arange(0.0, t_window_s, dt_wave_s, dtype=np.float32)
    e_wave = e_dc_v + e_amp_v * np.sin(2.0 * np.pi * frequency_hz * t_wave)

    if config is None:
        cfg = MultiPhysicsConfig(
            initial_theta=initial_theta,
            k_ads=0.0,
            k_des=0.0,
            k_reaction=0.0,
            k_clean=0.0,
            Rfilm_theta_max_ohm=1600.0,
            cdl_theta_fraction=0.7,
            area_floor_fraction=0.1,
            k0_theta_coeff=2.5,
        )
    else:
        cfg = config

    out = simulate_multiphysics_electrochem(
        E_array=jnp.asarray(e_wave),
        t_max=float(t_window_s),
        nx=nx,
        config=cfg,
        **_default_species(),
    )
    e_hist = np.asarray(out[3], dtype=float)
    i_hist = np.asarray(out[4], dtype=float)
    t_hist = np.linspace(0.0, t_window_s, e_hist.shape[0], endpoint=False, dtype=float)
    return estimate_impedance_from_trace(
        t_s=t_hist,
        potential_v=e_hist,
        current_mA=i_hist,
        frequency_hz=frequency_hz,
        discard_fraction=0.5,
    )


def multiphysics_impedance_sweep(
    freq_hz: np.ndarray,
    *,
    initial_theta: float,
    t_window_s: float = 8.0,
    dt_wave_s: float = 1e-3,
    e_dc_v: float = -0.02,
    e_amp_v: float = 0.01,
    nx: int = 28,
    config: MultiPhysicsConfig | None = None,
) -> dict[str, np.ndarray]:
    freq_hz = np.asarray(freq_hz, dtype=float).reshape(-1)
    if freq_hz.size == 0:
        raise ValueError("freq_hz cannot be empty")
    if np.any(freq_hz <= 0):
        raise ValueError("All frequencies must be positive")

    z_real = np.zeros_like(freq_hz)
    z_imag = np.zeros_like(freq_hz)
    z_mag = np.zeros_like(freq_hz)
    z_phase = np.zeros_like(freq_hz)
    i_amp = np.zeros_like(freq_hz)
    e_amp = np.zeros_like(freq_hz)

    for i, freq in enumerate(freq_hz):
        metrics = _simulate_single_tone(
            frequency_hz=float(freq),
            initial_theta=initial_theta,
            t_window_s=t_window_s,
            dt_wave_s=dt_wave_s,
            e_dc_v=e_dc_v,
            e_amp_v=e_amp_v,
            nx=nx,
            config=config,
        )
        z_real[i] = metrics["z_real_ohm"]
        z_imag[i] = metrics["z_imag_ohm"]
        z_mag[i] = metrics["z_mag_ohm"]
        z_phase[i] = metrics["z_phase_rad"]
        i_amp[i] = metrics["amplitude_mA"]
        e_amp[i] = metrics["amplitude_v"]

    return {
        "freq_hz": freq_hz,
        "z_real_ohm": z_real,
        "z_imag_ohm": z_imag,
        "z_mag_ohm": z_mag,
        "z_phase_rad": z_phase,
        "i_amp_mA": i_amp,
        "e_amp_v": e_amp,
    }


def fit_randles_three_element(
    freq_hz: np.ndarray,
    z_complex_ohm: np.ndarray,
    *,
    rs_grid_size: int = 24,
    rct_grid_size: int = 28,
    cdl_grid_size: int = 24,
) -> RandlesFit:
    """Fits Z = Rs + 1 / (1/Rct + j*w*Cdl) with coarse grid search."""
    freq_hz = np.asarray(freq_hz, dtype=float).reshape(-1)
    z_complex = np.asarray(z_complex_ohm, dtype=complex).reshape(-1)
    if not (freq_hz.shape[0] == z_complex.shape[0]):
        raise ValueError("freq_hz and z_complex_ohm must have matching lengths")
    if freq_hz.shape[0] < 3:
        raise ValueError("Need at least 3 frequencies for Randles fit")
    if np.any(freq_hz <= 0):
        raise ValueError("All frequencies must be positive")
    if rs_grid_size < 4 or rct_grid_size < 4 or cdl_grid_size < 4:
        raise ValueError("Grid sizes must be >= 4")

    ws = 2.0 * np.pi * freq_hz
    re = np.real(z_complex)
    rs_min = max(0.0, float(np.min(re) * 0.2))
    rs_max = max(rs_min + 1.0, float(np.max(re) * 1.3))

    rs_grid = np.linspace(rs_min, rs_max, rs_grid_size)
    rct_grid = np.logspace(1.0, 6.0, rct_grid_size)
    cdl_grid = np.logspace(-7.0, -3.0, cdl_grid_size)

    best_err = float("inf")
    best_params = (float(rs_grid[0]), float(rct_grid[0]), float(cdl_grid[0]))
    denom = np.maximum(np.abs(z_complex), 1e-12)

    for rs in rs_grid:
        for rct in rct_grid:
            for cdl in cdl_grid:
                z_model = rs + 1.0 / (1.0 / rct + 1j * ws * cdl)
                rel_err = float(np.mean(np.abs(z_complex - z_model) / denom))
                if rel_err < best_err:
                    best_err = rel_err
                    best_params = (float(rs), float(rct), float(cdl))

    return RandlesFit(
        rs_ohm=best_params[0],
        rct_ohm=best_params[1],
        cdl_f=best_params[2],
        mean_rel_error=best_err,
    )


def fouling_cleaning_cycle_benchmark() -> dict[str, float]:
    protocol = build_biofouling_protocol(
        dt=1e-3,
        n_cycles=3,
        baseline_duration_s=0.4,
        foul_duration_s=6.0,
        probe_duration_s=1.0,
        recovery_duration_s=0.4,
        cleaning_steps=((0.95, 0.8), (1.05, 0.6)),
    )
    e_array = jnp.asarray(protocol["E_array"], dtype=jnp.float32)
    cleaning_mask = jnp.asarray(protocol["cleaning_mask"], dtype=jnp.float32)
    t_max = float(protocol["t_max"])

    cfg = MultiPhysicsConfig(
        k_ads=4500.0,
        k_reaction=0.004,
        k_clean=0.45,
        k_des=1.0e-4,
        Rfilm_theta_max_ohm=1200.0,
        cdl_theta_fraction=0.75,
        area_floor_fraction=0.12,
        k0_theta_coeff=2.2,
        electrode_area_cm2=0.01,
    )

    out_clean = simulate_multiphysics_electrochem(
        E_array=e_array,
        cleaning_mask=cleaning_mask,
        t_max=t_max,
        nx=24,
        config=cfg,
        **_default_species(),
    )
    out_no_clean = simulate_multiphysics_electrochem(
        E_array=e_array,
        cleaning_mask=jnp.zeros_like(cleaning_mask),
        t_max=t_max,
        nx=24,
        config=cfg,
        **_default_species(),
    )

    theta_clean = np.asarray(out_clean[7], dtype=float)
    theta_no_clean = np.asarray(out_no_clean[7], dtype=float)
    rfilm_clean = np.asarray(out_clean[9], dtype=float)
    rfilm_no_clean = np.asarray(out_no_clean[9], dtype=float)

    return {
        "theta_peak_clean": float(np.max(theta_clean)),
        "theta_final_clean": float(theta_clean[-1]),
        "theta_final_no_clean": float(theta_no_clean[-1]),
        "rfilm_final_clean_ohm": float(rfilm_clean[-1]),
        "rfilm_final_no_clean_ohm": float(rfilm_no_clean[-1]),
    }


def randles_misfit_benchmark(
    *,
    initial_theta: float = 0.8,
    frequencies_hz: np.ndarray | None = None,
) -> dict[str, Any]:
    if frequencies_hz is None:
        frequencies_hz = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float)
    else:
        frequencies_hz = np.asarray(frequencies_hz, dtype=float).reshape(-1)

    sweep = multiphysics_impedance_sweep(
        frequencies_hz,
        initial_theta=initial_theta,
        t_window_s=8.0,
        dt_wave_s=1e-3,
        nx=28,
    )
    z_complex = sweep["z_real_ohm"] + 1j * sweep["z_imag_ohm"]
    randles_fit = fit_randles_three_element(sweep["freq_hz"], z_complex)

    ws = 2.0 * np.pi * sweep["freq_hz"]
    z_model = randles_fit.rs_ohm + 1.0 / (1.0 / randles_fit.rct_ohm + 1j * ws * randles_fit.cdl_f)
    rel_err = np.abs(z_complex - z_model) / np.maximum(np.abs(z_complex), 1e-12)

    return {
        "freq_hz": sweep["freq_hz"].tolist(),
        "z_real_ohm": sweep["z_real_ohm"].tolist(),
        "z_imag_ohm": sweep["z_imag_ohm"].tolist(),
        "z_mag_ohm": sweep["z_mag_ohm"].tolist(),
        "fit": {
            "rs_ohm": randles_fit.rs_ohm,
            "rct_ohm": randles_fit.rct_ohm,
            "cdl_f": randles_fit.cdl_f,
            "mean_rel_error": randles_fit.mean_rel_error,
        },
        "fit_rel_error_per_freq": rel_err.tolist(),
    }


def run_multiphysics_benchmarks() -> dict[str, Any]:
    fouling = fouling_cleaning_cycle_benchmark()
    clean = multiphysics_impedance_sweep(np.asarray([3.0], dtype=float), initial_theta=0.0)
    fouled = multiphysics_impedance_sweep(np.asarray([3.0], dtype=float), initial_theta=0.8)
    misfit = randles_misfit_benchmark(initial_theta=0.8)

    checks = {
        "cleaning_reduces_fouling": bool(fouling["theta_final_clean"] < fouling["theta_final_no_clean"]),
        "cleaning_reduces_film_resistance": bool(
            fouling["rfilm_final_clean_ohm"] < fouling["rfilm_final_no_clean_ohm"]
        ),
        "fouling_increases_impedance": bool(fouled["z_mag_ohm"][0] > clean["z_mag_ohm"][0] * 1.5),
        "biofouled_not_well_fit_by_3_element_randles": bool(misfit["fit"]["mean_rel_error"] >= 0.08),
    }
    return {
        "checks": checks,
        "overall_pass": bool(all(checks.values())),
        "fouling_cleaning": fouling,
        "impedance_clean": {
            "freq_hz": clean["freq_hz"].tolist(),
            "z_mag_ohm": clean["z_mag_ohm"].tolist(),
            "z_phase_rad": clean["z_phase_rad"].tolist(),
        },
        "impedance_fouled": {
            "freq_hz": fouled["freq_hz"].tolist(),
            "z_mag_ohm": fouled["z_mag_ohm"].tolist(),
            "z_phase_rad": fouled["z_phase_rad"].tolist(),
        },
        "randles_misfit": misfit,
    }
