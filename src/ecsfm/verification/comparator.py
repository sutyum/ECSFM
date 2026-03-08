"""Quantitative comparison of simulation results against reference data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ecsfm.verification.test_cases import TestCaseSpec


@dataclass
class ComparisonMetrics:
    """Quantitative error metrics between a simulation and a reference trace."""

    l2_norm: float
    l_inf_norm: float
    max_relative_error: float
    peak_position_error: float | None = None
    peak_current_error: float | None = None


def compare_traces(
    t_sim: np.ndarray,
    y_sim: np.ndarray,
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    interpolate: bool = True,
) -> ComparisonMetrics:
    """Compare two time-series traces with optional interpolation.

    If ``interpolate`` is True, the reference trace is interpolated onto the
    simulation timepoints before computing error norms.
    """
    t_sim = np.asarray(t_sim, dtype=float).ravel()
    y_sim = np.asarray(y_sim, dtype=float).ravel()
    t_ref = np.asarray(t_ref, dtype=float).ravel()
    y_ref = np.asarray(y_ref, dtype=float).ravel()

    if interpolate and t_sim.shape != t_ref.shape:
        # Interpolate reference onto simulation time grid
        y_ref_interp = np.interp(t_sim, t_ref, y_ref)
    elif interpolate and not np.allclose(t_sim, t_ref, atol=1e-12):
        y_ref_interp = np.interp(t_sim, t_ref, y_ref)
    else:
        y_ref_interp = y_ref
        if y_sim.shape != y_ref_interp.shape:
            # Truncate to shorter length
            n = min(y_sim.shape[0], y_ref_interp.shape[0])
            y_sim = y_sim[:n]
            y_ref_interp = y_ref_interp[:n]

    diff = y_sim - y_ref_interp

    # L2 norm (RMS-like, normalized by trace length)
    l2 = float(np.sqrt(np.mean(diff ** 2)))

    # L-infinity norm
    l_inf = float(np.max(np.abs(diff)))

    # Max relative error (avoid division by zero)
    denom = np.maximum(np.abs(y_ref_interp), 1e-12)
    max_rel = float(np.max(np.abs(diff) / denom))

    # Peak metrics
    peak_pos_err = None
    peak_curr_err = None

    if y_sim.shape[0] >= 4:
        idx_sim_peak = int(np.argmin(y_sim))
        idx_ref_peak = int(np.argmin(y_ref_interp))
        if t_sim.shape[0] > idx_sim_peak and t_sim.shape[0] > idx_ref_peak:
            peak_pos_err = float(abs(t_sim[idx_sim_peak] - t_sim[idx_ref_peak]))
        peak_sim = float(y_sim[idx_sim_peak])
        peak_ref = float(y_ref_interp[idx_ref_peak])
        if abs(peak_ref) > 1e-12:
            peak_curr_err = float(abs(peak_sim - peak_ref) / abs(peak_ref))

    return ComparisonMetrics(
        l2_norm=l2,
        l_inf_norm=l_inf,
        max_relative_error=max_rel,
        peak_position_error=peak_pos_err,
        peak_current_error=peak_curr_err,
    )


def _extract_cv_peaks(
    E: np.ndarray, I: np.ndarray
) -> dict[str, float]:
    """Extract CV peak metrics from E/I traces."""
    E = np.asarray(E, dtype=float)
    I = np.asarray(I, dtype=float)

    mid = E.shape[0] // 2
    if mid < 2:
        return {}

    idx_c = int(np.argmin(I[:mid]))
    idx_a = int(mid + np.argmax(I[mid:]))

    ep_c = float(E[idx_c])
    ep_a = float(E[idx_a])
    ip_c = float(-I[idx_c])
    ip_a = float(I[idx_a])

    return {
        "ep_c": ep_c,
        "ep_a": ep_a,
        "delta_ep": ep_a - ep_c,
        "ip_c": ip_c,
        "ip_a": ip_a,
    }


def compare_against_analytical(
    spec: TestCaseSpec,
    sim_result,  # SimResult (avoid circular import)
) -> ComparisonMetrics | None:
    """Compare simulation result against known analytical solutions.

    Supports:
    - cottrell_step: Cottrell equation
    - cv_reversible: Randles-Sevcik equation (peak separation check)

    Returns None if no analytical solution is available for the given case.
    """
    if spec.name == "cottrell_step":
        return _compare_cottrell(spec, sim_result)
    elif spec.name == "cv_reversible":
        return _compare_randles_sevcik(spec, sim_result)
    elif spec.name in ("cv_irreversible", "cv_quasireversible"):
        return _compare_cv_peak_separation(spec, sim_result)
    else:
        return None


def _compare_cottrell(spec: TestCaseSpec, sim_result) -> ComparisonMetrics:
    """Compare against the Cottrell equation: J = C * sqrt(D / (pi * t))."""
    time = sim_result.time
    current = sim_result.current

    # Get parameters from spec
    D = spec.species[0].D
    C_bulk = spec.species[0].C_bulk  # uM
    C_bulk_mol = C_bulk * 1e-6  # mol/cm^3

    F = 96485.3321

    # Skip very early times where numerics are less accurate
    valid = time > 0.005
    t_valid = time[valid]
    I_valid = current[valid]

    # Analytical Cottrell current: I = n*F*A * C * sqrt(D/(pi*t))
    # In our simulator, current is in mA per unit area (effectively)
    # The simulator returns flux * F * 1000 (to_mA)
    # Analytical flux = C_bulk_mol * sqrt(D / (pi * t))
    analytical_flux = C_bulk_mol * np.sqrt(D / (np.pi * t_valid))
    analytical_current = F * analytical_flux * 1000.0  # mA

    # The simulator current should be negative (reduction)
    # Take absolute values for comparison
    I_sim_abs = np.abs(I_valid)
    I_ref_abs = np.abs(analytical_current)

    diff = I_sim_abs - I_ref_abs
    denom = np.maximum(I_ref_abs, 1e-12)

    l2 = float(np.sqrt(np.mean(diff ** 2)))
    l_inf = float(np.max(np.abs(diff)))
    max_rel = float(np.max(np.abs(diff) / denom))

    return ComparisonMetrics(
        l2_norm=l2,
        l_inf_norm=l_inf,
        max_relative_error=max_rel,
    )


def _compare_randles_sevcik(spec: TestCaseSpec, sim_result) -> ComparisonMetrics:
    """Compare CV peak separation against Nernstian limit (59.2/n mV).

    Also checks peak current against Randles-Sevcik equation.
    """
    E = sim_result.potential
    I = sim_result.current

    peaks = _extract_cv_peaks(E, I)
    if not peaks:
        return ComparisonMetrics(
            l2_norm=float("inf"),
            l_inf_norm=float("inf"),
            max_relative_error=float("inf"),
        )

    n_electrons = spec.kinetics[0].n_electrons
    expected_delta_ep = 0.0592 / n_electrons  # Nernstian limit in V

    delta_ep_error = abs(peaks["delta_ep"] - expected_delta_ep)

    # Randles-Sevcik peak current (at 25C):
    # ip = 0.4463 * n^(3/2) * F * A * C * sqrt(D * v * F / (R*T))
    # Simplified: ip = 2.69e5 * n^(3/2) * A * D^(1/2) * C * v^(1/2)
    D = spec.species[0].D  # cm^2/s
    C_bulk_mol = spec.species[0].C_bulk * 1e-6  # mol/cm^3
    scan_rate = spec.waveform.params.get("scan_rate", 1.0)

    # Expected peak current density (A/cm^2) -> mA/cm^2
    ip_expected = (
        2.69e5 * (n_electrons ** 1.5) * np.sqrt(D) * C_bulk_mol * np.sqrt(scan_rate)
    )
    ip_expected_mA = ip_expected * 1000.0

    ip_sim = peaks["ip_c"]
    ip_rel_error = abs(ip_sim - ip_expected_mA) / max(abs(ip_expected_mA), 1e-12)

    return ComparisonMetrics(
        l2_norm=delta_ep_error,
        l_inf_norm=delta_ep_error,
        max_relative_error=ip_rel_error,
        peak_position_error=delta_ep_error,
        peak_current_error=ip_rel_error,
    )


def _compare_cv_peak_separation(spec: TestCaseSpec, sim_result) -> ComparisonMetrics:
    """Check that CV peak separation exceeds a minimum threshold."""
    E = sim_result.potential
    I = sim_result.current

    peaks = _extract_cv_peaks(E, I)
    if not peaks:
        return ComparisonMetrics(
            l2_norm=float("inf"),
            l_inf_norm=float("inf"),
            max_relative_error=float("inf"),
        )

    expected_min = spec.expected_metrics.get("delta_ep_min_v", 0.10)
    delta_ep = peaks["delta_ep"]
    error = max(0.0, expected_min - delta_ep)

    return ComparisonMetrics(
        l2_norm=error,
        l_inf_norm=error,
        max_relative_error=error / max(expected_min, 1e-12),
        peak_position_error=delta_ep,
        peak_current_error=None,
    )
