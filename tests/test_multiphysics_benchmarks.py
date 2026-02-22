import numpy as np

from ecsfm.sim.multiphysics_benchmarks import (
    fouling_cleaning_cycle_benchmark,
    multiphysics_impedance_sweep,
    randles_misfit_benchmark,
    run_multiphysics_benchmarks,
)


def test_fouling_cleaning_cycle_benchmark_shows_recovery():
    summary = fouling_cleaning_cycle_benchmark()
    assert summary["theta_final_clean"] < summary["theta_final_no_clean"]
    assert summary["rfilm_final_clean_ohm"] < summary["rfilm_final_no_clean_ohm"]
    assert summary["theta_peak_clean"] > 0.02


def test_impedance_sweep_shows_fouled_state_has_higher_impedance():
    freq = np.asarray([3.0], dtype=float)
    clean = multiphysics_impedance_sweep(freq, initial_theta=0.0)
    fouled = multiphysics_impedance_sweep(freq, initial_theta=0.8)

    assert fouled["z_mag_ohm"][0] > clean["z_mag_ohm"][0] * 1.5
    assert np.isfinite(clean["z_phase_rad"]).all()
    assert np.isfinite(fouled["z_phase_rad"]).all()


def test_biofouled_eis_is_not_well_fit_by_single_randles_arc():
    result = randles_misfit_benchmark(initial_theta=0.8)
    assert result["fit"]["mean_rel_error"] >= 0.08


def test_multiphysics_benchmark_runner_passes_all_checks():
    summary = run_multiphysics_benchmarks()
    assert summary["overall_pass"] is True
