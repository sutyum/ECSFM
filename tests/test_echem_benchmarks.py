import numpy as np

from ecsfm.sim.benchmarks import (
    cottrell_flux_time_series,
    cv_peak_current_vs_concentration,
    cv_peak_separation_vs_k0,
    cv_peak_separation_vs_scan_rate,
    sensor_impedance_sweep,
)


def test_cv_peak_separation_decreases_with_k0():
    k0_values = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], dtype=float)
    bench = cv_peak_separation_vs_k0(k0_values)
    delta_ep = np.asarray(bench["delta_ep"], dtype=float)

    assert np.all(np.diff(delta_ep) < 0.0)
    assert float(delta_ep[0]) > 0.30
    assert float(delta_ep[-1]) < 0.08


def test_cv_peak_separation_increases_with_scan_rate():
    scan_rates = np.array([0.5, 1.0, 2.0, 4.0], dtype=float)
    bench = cv_peak_separation_vs_scan_rate(scan_rates, k0=1e-3)
    delta_ep = np.asarray(bench["delta_ep"], dtype=float)

    assert np.all(np.diff(delta_ep) > 0.0)
    assert float(delta_ep[-1] - delta_ep[0]) > 0.08


def test_cv_peak_current_is_linear_in_concentration():
    concentrations = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    bench = cv_peak_current_vs_concentration(concentrations, k0=1e-2)
    fit = bench["fit_ipc_vs_concentration"]

    assert fit.r2 >= 0.999
    assert abs(fit.intercept) <= 1e-3
    assert fit.slope > 0.0


def test_cottrell_flux_matches_multitime_analytic_decay():
    times = np.array([0.02, 0.05, 0.1], dtype=float)
    bench = cottrell_flux_time_series(times)

    rel_error = np.asarray(bench["rel_error"], dtype=float)
    slope = float(bench["loglog_fit"].slope)

    assert float(np.max(rel_error)) <= 0.02
    assert abs(slope + 0.5) <= 0.04


def test_sensor_impedance_matches_theory_across_frequencies():
    frequencies = np.array([5.0, 20.0, 80.0, 200.0], dtype=float)
    bench = sensor_impedance_sweep(frequencies)

    amp_rel_error = np.asarray(bench["amp_rel_error"], dtype=float)
    phase_error = np.asarray(bench["phase_error_rad"], dtype=float)

    assert float(np.max(amp_rel_error)) <= 0.02
    assert float(np.max(np.abs(phase_error))) <= 0.03
