from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ecsfm.sim.cv import simulate_cv
from ecsfm.sim.mesh import Mesh1D
from ecsfm.sim.physics import Diffusion1D
from ecsfm.sim.sensor import apply_sensor_model


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    r2: float


@dataclass(frozen=True)
class BenchmarkThresholds:
    cv_k0_delta_ep_low_min: float = 0.30
    cv_k0_delta_ep_high_max: float = 0.08
    cv_scan_delta_ep_gain_min: float = 0.08
    cv_concentration_r2_min: float = 0.999
    cv_concentration_intercept_abs_max: float = 1e-3
    cottrell_rel_err_max: float = 0.02
    cottrell_slope_target: float = -0.5
    cottrell_slope_abs_tol: float = 0.04
    sensor_amp_rel_err_max: float = 0.02
    sensor_phase_err_max_rad: float = 0.03


def fit_linear(x: np.ndarray, y: np.ndarray) -> LinearFit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y length mismatch: {x.shape[0]} != {y.shape[0]}")
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 points for linear fit")

    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return LinearFit(slope=float(slope), intercept=float(intercept), r2=float(r2))


def _cv_peak_metrics(E_hist: np.ndarray, I_hist: np.ndarray) -> dict[str, float]:
    E_hist = np.asarray(E_hist, dtype=float)
    I_hist = np.asarray(I_hist, dtype=float)
    if E_hist.shape != I_hist.shape:
        raise ValueError(f"E_hist and I_hist shape mismatch: {E_hist.shape} != {I_hist.shape}")
    if E_hist.ndim != 1:
        raise ValueError(f"Expected 1D traces, got shape {E_hist.shape}")
    if E_hist.shape[0] < 4:
        raise ValueError("Need at least 4 points to extract CV peaks")

    mid = E_hist.shape[0] // 2
    if mid < 2:
        raise ValueError("Insufficient points in first CV half-cycle")

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


def cv_peak_separation_vs_k0(
    k0_values: np.ndarray,
    *,
    D: float = 1e-5,
    concentration: float = 1.0,
    E0: float = 0.0,
    alpha: float = 0.5,
    scan_rate: float = 1.0,
    E_start: float = 0.5,
    E_vertex: float = -0.5,
    nx: int = 36,
) -> dict[str, np.ndarray]:
    k0_values = np.asarray(k0_values, dtype=float)
    delta_ep = np.zeros_like(k0_values)
    ip_c = np.zeros_like(k0_values)
    ip_a = np.zeros_like(k0_values)

    for i, k0 in enumerate(k0_values):
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
        delta_ep[i] = metrics["delta_ep"]
        ip_c[i] = metrics["ip_c"]
        ip_a[i] = metrics["ip_a"]

    return {
        "k0_values": k0_values,
        "delta_ep": delta_ep,
        "ip_c": ip_c,
        "ip_a": ip_a,
    }


def cv_peak_separation_vs_scan_rate(
    scan_rates: np.ndarray,
    *,
    D: float = 1e-5,
    concentration: float = 1.0,
    E0: float = 0.0,
    alpha: float = 0.5,
    k0: float = 1e-3,
    E_start: float = 0.5,
    E_vertex: float = -0.5,
    nx: int = 36,
) -> dict[str, np.ndarray]:
    scan_rates = np.asarray(scan_rates, dtype=float)
    delta_ep = np.zeros_like(scan_rates)
    ip_c = np.zeros_like(scan_rates)
    ip_a = np.zeros_like(scan_rates)

    for i, scan_rate in enumerate(scan_rates):
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
        delta_ep[i] = metrics["delta_ep"]
        ip_c[i] = metrics["ip_c"]
        ip_a[i] = metrics["ip_a"]

    return {
        "scan_rates": scan_rates,
        "delta_ep": delta_ep,
        "ip_c": ip_c,
        "ip_a": ip_a,
    }


def cv_peak_current_vs_concentration(
    concentrations: np.ndarray,
    *,
    D: float = 1e-5,
    E0: float = 0.0,
    alpha: float = 0.5,
    k0: float = 1e-2,
    scan_rate: float = 1.0,
    E_start: float = 0.5,
    E_vertex: float = -0.5,
    nx: int = 36,
) -> dict[str, np.ndarray | LinearFit]:
    concentrations = np.asarray(concentrations, dtype=float)
    ip_c = np.zeros_like(concentrations)
    ip_a = np.zeros_like(concentrations)

    for i, concentration in enumerate(concentrations):
        _, _, _, E_hist, I_hist, _ = simulate_cv(
            D_ox=D,
            D_red=D,
            C_bulk_ox=float(concentration),
            C_bulk_red=0.0,
            E0=E0,
            k0=k0,
            alpha=alpha,
            scan_rate=scan_rate,
            E_start=E_start,
            E_vertex=E_vertex,
            nx=nx,
            save_every=0,
        )
        metrics = _cv_peak_metrics(E_hist, I_hist)
        ip_c[i] = metrics["ip_c"]
        ip_a[i] = metrics["ip_a"]

    fit = fit_linear(concentrations, ip_c)
    return {
        "concentrations": concentrations,
        "ip_c": ip_c,
        "ip_a": ip_a,
        "fit_ipc_vs_concentration": fit,
    }


def cottrell_flux_time_series(
    times_s: np.ndarray,
    *,
    D: float = 1e-5,
    C_bulk: float = 1.0,
    L: float = 0.05,
    nx: int = 1000,
    dt: float = 2e-5,
) -> dict[str, np.ndarray | LinearFit]:
    times_s = np.asarray(times_s, dtype=float)
    if np.any(times_s <= 0):
        raise ValueError("All times must be positive")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    if nx < 2:
        raise ValueError(f"nx must be >= 2, got {nx}")

    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    mesh = Mesh1D(x_min=0.0, x_max=L, n_points=nx, dtype=dtype)
    diffusion = Diffusion1D(mesh, D)
    c_state = jnp.full((nx,), jnp.asarray(C_bulk, dtype=dtype))
    c_state = c_state.at[0].set(jnp.asarray(0.0, dtype=c_state.dtype))
    dt_arr = jnp.asarray(dt, dtype=c_state.dtype)
    c_bulk_arr = jnp.asarray(C_bulk, dtype=c_state.dtype)

    @jax.jit
    def step(c_current: jax.Array) -> jax.Array:
        rates = diffusion.compute_rates(c_current)
        c_next = c_current + dt_arr * rates
        c_next = c_next.at[0].set(jnp.asarray(0.0, dtype=c_current.dtype))
        c_next = c_next.at[-1].set(c_bulk_arr)
        return c_next

    order = np.argsort(times_s)
    times_sorted = times_s[order]
    flux_sorted = np.zeros_like(times_sorted)

    previous_steps = 0
    state = c_state
    for i, t_eval in enumerate(times_sorted):
        steps = int(t_eval / dt)
        if steps < 1:
            raise ValueError(f"time {t_eval} too small for dt={dt}; increase time or decrease dt")
        delta_steps = steps - previous_steps
        if delta_steps > 0:
            state = jax.lax.fori_loop(0, delta_steps, lambda _, c: step(c), state)
        previous_steps = steps
        flux = D * (state[1] - state[0]) / mesh.dx
        flux_sorted[i] = float(flux)

    flux = np.empty_like(flux_sorted)
    flux[order] = flux_sorted
    analytic_flux = C_bulk * np.sqrt(D / (np.pi * times_s))
    rel_error = np.abs(flux - analytic_flux) / analytic_flux
    slope_fit = fit_linear(np.log(times_s), np.log(flux))

    return {
        "times_s": times_s,
        "flux": flux,
        "analytic_flux": analytic_flux,
        "rel_error": rel_error,
        "loglog_fit": slope_fit,
    }


def _fit_sine_response(t: np.ndarray, y: np.ndarray, freq_hz: float) -> tuple[float, float]:
    omega = 2.0 * np.pi * freq_hz
    X = np.column_stack(
        [
            np.sin(omega * t),
            np.cos(omega * t),
            np.ones_like(t),
        ]
    )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, _ = beta
    amplitude = float(np.hypot(a, b))
    phase_rad = float(np.arctan2(b, a))
    return amplitude, phase_rad


def wrap_phase_error(measured: np.ndarray, expected: np.ndarray) -> np.ndarray:
    measured = np.asarray(measured, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return (measured - expected + np.pi) % (2.0 * np.pi) - np.pi


def sensor_impedance_sweep(
    freq_hz: np.ndarray,
    *,
    amplitude_v: float = 0.01,
    cycles: int = 10,
    points_per_cycle: int = 2000,
    Cdl: float = 1e-5,
    Ru: float = 100.0,
) -> dict[str, np.ndarray]:
    freq_hz = np.asarray(freq_hz, dtype=float)
    if np.any(freq_hz <= 0):
        raise ValueError("All frequencies must be positive")

    amp_measured = np.zeros_like(freq_hz)
    amp_theory = np.zeros_like(freq_hz)
    phase_measured = np.zeros_like(freq_hz)
    phase_theory = np.zeros_like(freq_hz)

    for i, freq in enumerate(freq_hz):
        omega = 2.0 * np.pi * freq
        dt = 1.0 / (points_per_cycle * freq)
        t_max = cycles / freq
        t = np.arange(0.0, t_max, dt, dtype=float)
        E_app = amplitude_v * np.sin(omega * t)
        I_f_mA = np.zeros_like(t)

        _, i_total_mA, _ = apply_sensor_model(
            t=jnp.asarray(t, dtype=jnp.float32),
            E_app=jnp.asarray(E_app, dtype=jnp.float32),
            I_f_mA=jnp.asarray(I_f_mA, dtype=jnp.float32),
            Cdl=Cdl,
            Ru=Ru,
            noise_std_mA=0.0,
        )

        y = np.asarray(i_total_mA, dtype=float)
        steady = t >= (0.5 * t_max)
        amp_fit, phase_fit = _fit_sine_response(t[steady], y[steady], freq_hz=freq)

        Z = Ru - 1j / (omega * Cdl)
        amp_ref = (amplitude_v / np.abs(Z)) * 1000.0
        phase_ref = -np.angle(Z)

        amp_measured[i] = amp_fit
        amp_theory[i] = float(amp_ref)
        phase_measured[i] = phase_fit
        phase_theory[i] = float(phase_ref)

    phase_error = wrap_phase_error(phase_measured, phase_theory)
    amp_rel_error = np.abs(amp_measured - amp_theory) / np.maximum(amp_theory, 1e-12)

    return {
        "freq_hz": freq_hz,
        "amp_measured_mA": amp_measured,
        "amp_theory_mA": amp_theory,
        "amp_rel_error": amp_rel_error,
        "phase_measured_rad": phase_measured,
        "phase_theory_rad": phase_theory,
        "phase_error_rad": phase_error,
    }


def run_canonical_benchmarks(
    thresholds: BenchmarkThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or BenchmarkThresholds()

    k0_sweep = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], dtype=float)
    scan_sweep = np.array([0.5, 1.0, 2.0, 4.0], dtype=float)
    concentration_sweep = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    cottrell_times = np.array([0.02, 0.05, 0.1], dtype=float)
    sensor_freqs = np.array([5.0, 20.0, 80.0, 200.0], dtype=float)

    by_k0 = cv_peak_separation_vs_k0(k0_sweep)
    by_scan = cv_peak_separation_vs_scan_rate(scan_sweep, k0=1e-3)
    by_concentration = cv_peak_current_vs_concentration(concentration_sweep, k0=1e-2)
    cottrell = cottrell_flux_time_series(cottrell_times)
    sensor = sensor_impedance_sweep(sensor_freqs)

    checks = {
        "cv_delta_ep_monotonic_vs_k0": bool(np.all(np.diff(by_k0["delta_ep"]) < 0.0)),
        "cv_delta_ep_range_vs_k0": bool(
            by_k0["delta_ep"][0] > thresholds.cv_k0_delta_ep_low_min
            and by_k0["delta_ep"][-1] < thresholds.cv_k0_delta_ep_high_max
        ),
        "cv_delta_ep_monotonic_vs_scan_rate": bool(np.all(np.diff(by_scan["delta_ep"]) > 0.0)),
        "cv_delta_ep_scan_gain": bool(
            (by_scan["delta_ep"][-1] - by_scan["delta_ep"][0]) > thresholds.cv_scan_delta_ep_gain_min
        ),
        "cv_ipc_linear_vs_concentration": bool(
            by_concentration["fit_ipc_vs_concentration"].r2 >= thresholds.cv_concentration_r2_min
            and abs(by_concentration["fit_ipc_vs_concentration"].intercept)
            <= thresholds.cv_concentration_intercept_abs_max
        ),
        "cottrell_rel_error": bool(
            float(np.max(cottrell["rel_error"])) <= thresholds.cottrell_rel_err_max
        ),
        "cottrell_loglog_slope": bool(
            abs(
                cottrell["loglog_fit"].slope - thresholds.cottrell_slope_target
            )
            <= thresholds.cottrell_slope_abs_tol
        ),
        "sensor_amplitude_error": bool(
            float(np.max(sensor["amp_rel_error"])) <= thresholds.sensor_amp_rel_err_max
        ),
        "sensor_phase_error": bool(
            float(np.max(np.abs(sensor["phase_error_rad"]))) <= thresholds.sensor_phase_err_max_rad
        ),
    }

    summary: dict[str, Any] = {
        "checks": checks,
        "overall_pass": bool(all(checks.values())),
        "cv_vs_k0": {
            "k0_values": by_k0["k0_values"].tolist(),
            "delta_ep": by_k0["delta_ep"].tolist(),
            "ip_c": by_k0["ip_c"].tolist(),
            "ip_a": by_k0["ip_a"].tolist(),
        },
        "cv_vs_scan_rate": {
            "scan_rates": by_scan["scan_rates"].tolist(),
            "delta_ep": by_scan["delta_ep"].tolist(),
            "ip_c": by_scan["ip_c"].tolist(),
            "ip_a": by_scan["ip_a"].tolist(),
        },
        "cv_vs_concentration": {
            "concentrations": by_concentration["concentrations"].tolist(),
            "ip_c": by_concentration["ip_c"].tolist(),
            "ip_a": by_concentration["ip_a"].tolist(),
            "fit": {
                "slope": by_concentration["fit_ipc_vs_concentration"].slope,
                "intercept": by_concentration["fit_ipc_vs_concentration"].intercept,
                "r2": by_concentration["fit_ipc_vs_concentration"].r2,
            },
        },
        "cottrell": {
            "times_s": cottrell["times_s"].tolist(),
            "flux": cottrell["flux"].tolist(),
            "analytic_flux": cottrell["analytic_flux"].tolist(),
            "rel_error": cottrell["rel_error"].tolist(),
            "loglog_fit": {
                "slope": cottrell["loglog_fit"].slope,
                "intercept": cottrell["loglog_fit"].intercept,
                "r2": cottrell["loglog_fit"].r2,
            },
        },
        "sensor_impedance": {
            "freq_hz": sensor["freq_hz"].tolist(),
            "amp_measured_mA": sensor["amp_measured_mA"].tolist(),
            "amp_theory_mA": sensor["amp_theory_mA"].tolist(),
            "amp_rel_error": sensor["amp_rel_error"].tolist(),
            "phase_measured_rad": sensor["phase_measured_rad"].tolist(),
            "phase_theory_rad": sensor["phase_theory_rad"].tolist(),
            "phase_error_rad": sensor["phase_error_rad"].tolist(),
        },
        "thresholds": {
            "cv_k0_delta_ep_low_min": thresholds.cv_k0_delta_ep_low_min,
            "cv_k0_delta_ep_high_max": thresholds.cv_k0_delta_ep_high_max,
            "cv_scan_delta_ep_gain_min": thresholds.cv_scan_delta_ep_gain_min,
            "cv_concentration_r2_min": thresholds.cv_concentration_r2_min,
            "cv_concentration_intercept_abs_max": thresholds.cv_concentration_intercept_abs_max,
            "cottrell_rel_err_max": thresholds.cottrell_rel_err_max,
            "cottrell_slope_target": thresholds.cottrell_slope_target,
            "cottrell_slope_abs_tol": thresholds.cottrell_slope_abs_tol,
            "sensor_amp_rel_err_max": thresholds.sensor_amp_rel_err_max,
            "sensor_phase_err_max_rad": thresholds.sensor_phase_err_max_rad,
        },
    }
    return summary
