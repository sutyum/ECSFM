# ruff: noqa: E402
"""Tests for the ECSFM verification system."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from ecsfm.verification.comparator import ComparisonMetrics, compare_against_analytical, compare_traces
from ecsfm.verification.runner import SimResult, run_case
from ecsfm.verification.test_cases import CANONICAL_CASES, TestCaseSpec


# ---------------------------------------------------------------------------
# 1. Smoke test: all cases produce results
# ---------------------------------------------------------------------------

class TestAllCasesRun:
    """Verify that every canonical case can be run through the simulator."""

    @pytest.mark.parametrize("case_name", list(CANONICAL_CASES.keys()))
    def test_case_produces_result(self, case_name: str):
        spec = CANONICAL_CASES[case_name]
        result = run_case(spec)

        assert isinstance(result, SimResult)
        assert result.time is not None
        assert result.potential is not None
        assert result.current is not None
        assert result.time.shape[0] > 0
        assert result.potential.shape[0] > 0
        assert result.current.shape[0] > 0
        assert np.all(np.isfinite(result.current)), (
            f"Case '{case_name}' produced non-finite current values"
        )


# ---------------------------------------------------------------------------
# 2. Analytical cases pass thresholds
# ---------------------------------------------------------------------------

class TestAnalyticalCasesPass:
    """Verify that cases with analytical solutions pass their tolerances."""

    def test_cottrell_step_passes(self):
        spec = CANONICAL_CASES["cottrell_step"]
        result = run_case(spec)
        metrics = compare_against_analytical(spec, result)

        assert metrics is not None, "Cottrell analytical comparison returned None"
        tol = spec.tolerance.get("cottrell_rel_error_max", 0.05)
        assert metrics.max_relative_error < tol, (
            f"Cottrell max relative error {metrics.max_relative_error:.4f} "
            f"exceeds tolerance {tol}"
        )

    def test_cv_reversible_passes(self):
        spec = CANONICAL_CASES["cv_reversible"]
        result = run_case(spec)
        metrics = compare_against_analytical(spec, result)

        assert metrics is not None, "Randles-Sevcik analytical comparison returned None"
        delta_ep_tol = spec.tolerance.get("delta_ep_abs_tol", 0.025)
        assert metrics.peak_position_error is not None
        assert metrics.peak_position_error < delta_ep_tol, (
            f"CV reversible delta Ep error {metrics.peak_position_error:.4f} V "
            f"exceeds tolerance {delta_ep_tol} V"
        )

    def test_cv_irreversible_large_peak_separation(self):
        spec = CANONICAL_CASES["cv_irreversible"]
        result = run_case(spec)
        metrics = compare_against_analytical(spec, result)

        assert metrics is not None
        # For irreversible case, peak_position_error stores the actual delta_ep
        assert metrics.peak_position_error is not None
        min_delta_ep = spec.expected_metrics.get("delta_ep_min_v", 0.30)
        assert metrics.peak_position_error >= min_delta_ep, (
            f"CV irreversible delta Ep = {metrics.peak_position_error:.4f} V "
            f"below expected minimum {min_delta_ep} V"
        )


# ---------------------------------------------------------------------------
# 3. Comparator with zero error
# ---------------------------------------------------------------------------

class TestComparatorZeroError:
    """Verify that comparing identical traces produces zero error."""

    def test_identical_traces_zero_error(self):
        t = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * t)

        metrics = compare_traces(t, y, t, y, interpolate=False)

        assert metrics.l2_norm == pytest.approx(0.0, abs=1e-15)
        assert metrics.l_inf_norm == pytest.approx(0.0, abs=1e-15)
        assert metrics.max_relative_error == pytest.approx(0.0, abs=1e-10)

    def test_identical_traces_with_interpolation(self):
        t = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * t)

        metrics = compare_traces(t, y, t, y, interpolate=True)

        assert metrics.l2_norm == pytest.approx(0.0, abs=1e-15)
        assert metrics.l_inf_norm == pytest.approx(0.0, abs=1e-15)

    def test_known_offset_error(self):
        t = np.linspace(0, 1, 1000)
        y_ref = np.ones_like(t) * 5.0
        y_sim = np.ones_like(t) * 5.1

        metrics = compare_traces(t, y_sim, t, y_ref, interpolate=False)

        assert metrics.l2_norm == pytest.approx(0.1, abs=1e-10)
        assert metrics.l_inf_norm == pytest.approx(0.1, abs=1e-10)
        assert metrics.max_relative_error == pytest.approx(0.02, abs=1e-10)

    def test_interpolation_between_grids(self):
        t_sim = np.linspace(0, 1, 50)
        t_ref = np.linspace(0, 1, 200)
        y_sim = np.sin(2 * np.pi * t_sim)
        y_ref = np.sin(2 * np.pi * t_ref)

        metrics = compare_traces(t_sim, y_sim, t_ref, y_ref, interpolate=True)

        # Interpolation error should be small for a smooth function
        assert metrics.l2_norm < 0.05
        assert metrics.l_inf_norm < 0.15


# ---------------------------------------------------------------------------
# 4. COMSOL runner builds model (skip if COMSOL not installed)
# ---------------------------------------------------------------------------

class TestComsolRunner:
    """Verify COMSOL runner integration (skipped if COMSOL/MPh unavailable)."""

    def test_comsol_runner_builds_model(self):
        mph = pytest.importorskip("mph")

        from ecsfm.verification.comsol_runner import build_comsol_model

        spec = CANONICAL_CASES["cottrell_step"]
        model = build_comsol_model(spec)
        assert model is not None

    def test_comsol_not_available_returns_none(self):
        """Without COMSOL, functions should return None gracefully."""
        from ecsfm.verification.comsol_runner import _MPH_AVAILABLE

        if _MPH_AVAILABLE:
            pytest.skip("MPh is installed; this test checks the not-installed path")

        from ecsfm.verification.comsol_runner import build_comsol_model, run_comsol_case

        spec = CANONICAL_CASES["cottrell_step"]

        with pytest.warns(UserWarning, match="MPh/COMSOL not available"):
            result = build_comsol_model(spec)
        assert result is None

        with pytest.warns(UserWarning, match="MPh/COMSOL not available"):
            result = run_comsol_case(spec)
        assert result is None

    def test_load_reference_data_missing_file(self, tmp_path):
        from ecsfm.verification.comsol_runner import load_reference_data

        result = load_reference_data("nonexistent_case", data_dir=tmp_path)
        assert result is None

    def test_load_reference_data_roundtrip(self, tmp_path):
        """Write a CSV and verify load_reference_data reads it correctly."""
        from ecsfm.verification.comsol_runner import load_reference_data

        csv_path = tmp_path / "test_case_comsol.csv"
        header = "time_s,potential_V,current_mA"
        data = np.column_stack([
            np.array([0.0, 0.01, 0.02]),
            np.array([0.5, 0.4, 0.3]),
            np.array([-0.1, -0.2, -0.3]),
        ])
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

        result = load_reference_data("test_case", data_dir=tmp_path)
        assert result is not None
        np.testing.assert_allclose(result["time"], [0.0, 0.01, 0.02])
        np.testing.assert_allclose(result["potential"], [0.5, 0.4, 0.3])
        np.testing.assert_allclose(result["current"], [-0.1, -0.2, -0.3])
