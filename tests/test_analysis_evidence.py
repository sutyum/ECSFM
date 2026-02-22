import numpy as np

from ecsfm.analysis.evidence import (
    cv_trace_sweep_by_k0,
    cv_trace_sweep_by_scan_rate,
    dataset_recipe_audit,
    simulator_convergence_study,
)


def test_cv_k0_sweep_has_monotonic_delta_ep_drop():
    sweep = cv_trace_sweep_by_k0(np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], dtype=float), nx=24)
    delta_ep = np.asarray([row["delta_ep"] for row in sweep["summary"]], dtype=float)
    assert np.all(np.diff(delta_ep) < 0.0)


def test_cv_scan_sweep_has_monotonic_delta_ep_increase():
    sweep = cv_trace_sweep_by_scan_rate(np.array([0.5, 1.0, 2.0, 4.0], dtype=float), nx=24)
    delta_ep = np.asarray([row["delta_ep"] for row in sweep["summary"]], dtype=float)
    assert np.all(np.diff(delta_ep) > 0.0)


def test_convergence_study_errors_decrease_with_nx():
    study = simulator_convergence_study(np.array([20, 28, 36], dtype=int), reference_nx=44)
    nrmse = np.asarray([row["nrmse_vs_ref"] for row in study["rows"]], dtype=float)
    dep = np.asarray([row["delta_ep_abs_error_vs_ref"] for row in study["rows"]], dtype=float)
    assert np.all(np.diff(nrmse) < 0.0)
    assert np.all(np.diff(dep) < 0.0)


def test_dataset_recipe_audit_returns_consistent_structure():
    report = dataset_recipe_audit(
        recipes=["baseline_random", "curriculum_multitask"],
        n_samples=32,
        max_species=4,
        nx=16,
        seed=7,
        sim_steps=96,
        device_batch_size=32,
        target_sig_len=96,
    )
    assert report["recipes"] == ["baseline_random", "curriculum_multitask"]
    assert len(report["rows"]) == 2
    assert len(report["recipe_ranking"]) == 2

    for row in report["rows"]:
        assert row["total_rows"] >= 32
        assert 0.0 <= row["augmentation_fraction"] <= 1.0
        assert 0.0 <= row["permute_pass_fraction"] <= 1.0
        assert 0.0 <= row["scale_within_20pct_fraction"] <= 1.0
        assert len(row["task_counts"]) == len(report["task_names"])
        assert len(row["stage_counts"]) == len(report["stage_names"])
