import jax.numpy as jnp
import numpy as np
import pytest

from ecsfm.sim.cv import simulate_cv
from ecsfm.sim.experiment import simulate_electrochem
from ecsfm.sim.sensor import apply_sensor_model


def _multispecies_kwargs() -> dict[str, jnp.ndarray]:
    return {
        "D_ox": jnp.array([1.0e-5, 2.0e-5], dtype=jnp.float32),
        "D_red": jnp.array([1.2e-5, 1.8e-5], dtype=jnp.float32),
        "C_bulk_ox": jnp.array([1.0, 0.4], dtype=jnp.float32),
        "C_bulk_red": jnp.array([0.0, 0.2], dtype=jnp.float32),
        "E0": jnp.array([0.0, 0.15], dtype=jnp.float32),
        "k0": jnp.array([0.01, 0.02], dtype=jnp.float32),
        "alpha": jnp.array([0.5, 0.35], dtype=jnp.float32),
    }


def _waveform_library(n_points: int = 96) -> dict[str, np.ndarray]:
    tau = np.linspace(0.0, 1.0, n_points)
    return {
        "cv_like": np.linspace(0.45, -0.35, n_points),
        "ca_step": np.where(tau < 0.4, -0.1, 0.55),
        "eis_sine": 0.08 * np.sin(2.0 * np.pi * 8.0 * tau),
        "swv_like": -0.05 + 0.12 * np.sign(np.sin(2.0 * np.pi * 6.0 * tau)),
    }


@pytest.mark.parametrize("scenario_name", ["cv_like", "ca_step", "eis_sine", "swv_like"])
def test_experiment_scenarios_are_finite_and_keep_bulk_boundary(scenario_name: str):
    waveforms = _waveform_library()
    E_array = jnp.asarray(waveforms[scenario_name], dtype=jnp.float32)

    out = simulate_electrochem(
        E_array=E_array,
        t_max=0.30,
        nx=20,
        save_every=0,
        **_multispecies_kwargs(),
    )
    x, c_ox_hist, c_red_hist, e_hist, i_hist, e_vis, i_vis = out

    bulk_ox = np.array([1.0, 0.4], dtype=np.float32)
    bulk_red = np.array([0.0, 0.2], dtype=np.float32)

    assert x.shape == (20,)
    assert c_ox_hist.ndim == 3 and c_red_hist.ndim == 3
    assert c_ox_hist.shape[1:] == (2, 20)
    assert c_ox_hist.shape == c_red_hist.shape
    assert e_hist.shape == i_hist.shape
    assert e_vis.shape == i_vis.shape
    assert e_vis.shape[0] == c_ox_hist.shape[0]

    assert np.isfinite(c_ox_hist).all()
    assert np.isfinite(c_red_hist).all()
    assert np.isfinite(e_hist).all()
    assert np.isfinite(i_hist).all()
    assert np.isfinite(e_vis).all()
    assert np.isfinite(i_vis).all()

    assert np.min(c_ox_hist) >= -1e-8
    assert np.min(c_red_hist) >= -1e-8

    expected_ox_bulk = np.broadcast_to(bulk_ox[None, :], c_ox_hist[:, :, -1].shape)
    expected_red_bulk = np.broadcast_to(bulk_red[None, :], c_red_hist[:, :, -1].shape)
    np.testing.assert_allclose(c_ox_hist[:, :, -1], expected_ox_bulk, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(c_red_hist[:, :, -1], expected_red_bulk, atol=1e-6, rtol=0.0)

    assert np.min(e_hist) >= float(np.min(waveforms[scenario_name])) - 1e-6
    assert np.max(e_hist) <= float(np.max(waveforms[scenario_name])) + 1e-6


def test_experiment_equilibrium_no_flux_stays_near_constant():
    out = simulate_electrochem(
        E_array=jnp.zeros((96,), dtype=jnp.float32),
        t_max=0.25,
        D_ox=jnp.array([1e-5], dtype=jnp.float32),
        D_red=jnp.array([1e-5], dtype=jnp.float32),
        C_bulk_ox=jnp.array([1.0], dtype=jnp.float32),
        C_bulk_red=jnp.array([1.0], dtype=jnp.float32),
        E0=jnp.array([0.0], dtype=jnp.float32),
        k0=jnp.array([0.02], dtype=jnp.float32),
        alpha=jnp.array([0.5], dtype=jnp.float32),
        nx=20,
        save_every=1,
    )
    _, c_ox_hist, c_red_hist, e_hist, i_hist, e_vis, i_vis = out

    assert e_hist.shape == i_hist.shape
    assert e_vis.shape == i_vis.shape
    np.testing.assert_allclose(e_vis, e_hist, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(i_vis, i_hist, atol=0.0, rtol=0.0)

    assert np.max(np.abs(i_hist)) < 1e-5
    np.testing.assert_allclose(c_ox_hist, 1.0, atol=5e-4, rtol=0.0)
    np.testing.assert_allclose(c_red_hist, 1.0, atol=5e-4, rtol=0.0)


def test_experiment_minimal_grid_and_large_save_interval():
    out = simulate_electrochem(
        E_array=jnp.linspace(-0.2, 0.3, 16, dtype=jnp.float32),
        t_max=6.0,
        D_ox=jnp.array([1e-5], dtype=jnp.float32),
        D_red=jnp.array([1e-5], dtype=jnp.float32),
        C_bulk_ox=jnp.array([1.0], dtype=jnp.float32),
        C_bulk_red=jnp.array([0.0], dtype=jnp.float32),
        E0=jnp.array([0.0], dtype=jnp.float32),
        k0=jnp.array([0.01], dtype=jnp.float32),
        alpha=jnp.array([0.5], dtype=jnp.float32),
        nx=2,
        save_every=10_000,
    )
    _, c_ox_hist, c_red_hist, e_hist, i_hist, e_vis, i_vis = out

    assert c_ox_hist.shape == (1, 1, 2)
    assert c_red_hist.shape == (1, 1, 2)
    assert e_hist.shape == i_hist.shape
    assert e_vis.shape == (1,)
    assert i_vis.shape == (1,)
    np.testing.assert_allclose(c_ox_hist[:, :, -1], np.array([[1.0]], dtype=np.float32), atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(c_red_hist[:, :, -1], np.array([[0.0]], dtype=np.float32), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_experiment_alpha_edge_cases_remain_stable(alpha: float):
    out = simulate_electrochem(
        E_array=jnp.linspace(-0.3, 0.3, 96, dtype=jnp.float32),
        t_max=0.25,
        D_ox=jnp.array([1e-5], dtype=jnp.float32),
        D_red=jnp.array([1e-5], dtype=jnp.float32),
        C_bulk_ox=jnp.array([1.0], dtype=jnp.float32),
        C_bulk_red=jnp.array([0.0], dtype=jnp.float32),
        E0=jnp.array([0.0], dtype=jnp.float32),
        k0=jnp.array([0.01], dtype=jnp.float32),
        alpha=jnp.array([alpha], dtype=jnp.float32),
        nx=16,
        save_every=0,
    )
    _, c_ox_hist, c_red_hist, _, i_hist, _, _ = out

    assert np.isfinite(c_ox_hist).all()
    assert np.isfinite(c_red_hist).all()
    assert np.isfinite(i_hist).all()
    assert np.min(c_ox_hist) >= -1e-8
    assert np.min(c_red_hist) >= -1e-8


def test_cv_reversal_and_bulk_boundary_conditions():
    out = simulate_cv(
        D_ox=1e-5,
        D_red=1e-5,
        C_bulk_ox=1.2,
        C_bulk_red=0.1,
        E0=0.05,
        k0=0.02,
        alpha=0.5,
        scan_rate=4.0,
        E_start=0.6,
        E_vertex=-0.4,
        nx=24,
        save_every=0,
    )
    _, c_ox_hist, c_red_hist, e_hist, i_hist, e_hist_vis = out

    mid = len(e_hist) // 2
    assert e_hist.shape == i_hist.shape
    assert len(e_hist_vis) == c_ox_hist.shape[0]
    assert np.isfinite(c_ox_hist).all()
    assert np.isfinite(c_red_hist).all()
    assert np.isfinite(e_hist).all()
    assert np.isfinite(i_hist).all()

    assert np.isclose(e_hist[0], 0.6, atol=1e-6)
    assert np.min(e_hist) <= -0.39
    assert np.mean(np.diff(e_hist[:mid])) < 0.0
    assert np.mean(np.diff(e_hist[mid:])) > 0.0

    assert np.min(c_ox_hist) >= -1e-9
    assert np.min(c_red_hist) >= -1e-9
    np.testing.assert_allclose(c_ox_hist[:, -1], np.full((c_ox_hist.shape[0],), 1.2), atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(c_red_hist[:, -1], np.full((c_red_hist.shape[0],), 0.1), atol=1e-6, rtol=0.0)


def test_cv_peak_current_increases_with_faster_kinetics():
    common = dict(
        nx=24,
        scan_rate=3.0,
        E_start=0.5,
        E_vertex=-0.5,
        D_ox=1e-5,
        D_red=1e-5,
        C_bulk_ox=1.0,
        C_bulk_red=0.0,
        E0=0.0,
        alpha=0.5,
    )
    i_low = np.asarray(simulate_cv(k0=1e-6, **common)[4])
    i_high = np.asarray(simulate_cv(k0=1.0, **common)[4])

    peak_low = float(np.max(np.abs(i_low)))
    peak_high = float(np.max(np.abs(i_high)))
    assert peak_low > 0.0
    assert peak_high > peak_low * 5.0


def test_sensor_model_matches_faradaic_trace_when_rc_is_negligible():
    t_max = 0.20
    out = simulate_electrochem(
        E_array=jnp.linspace(-0.3, 0.3, 96, dtype=jnp.float32),
        t_max=t_max,
        D_ox=jnp.array([1e-5], dtype=jnp.float32),
        D_red=jnp.array([1e-5], dtype=jnp.float32),
        C_bulk_ox=jnp.array([1.0], dtype=jnp.float32),
        C_bulk_red=jnp.array([0.0], dtype=jnp.float32),
        E0=jnp.array([0.0], dtype=jnp.float32),
        k0=jnp.array([0.01], dtype=jnp.float32),
        alpha=jnp.array([0.5], dtype=jnp.float32),
        nx=20,
        save_every=0,
    )
    _, _, _, e_hist, i_f_hist, _, _ = out
    t = jnp.linspace(0.0, t_max, len(e_hist), dtype=jnp.float32)

    _, i_total_mA, i_cap_mA = apply_sensor_model(
        t=t,
        E_app=jnp.asarray(e_hist),
        I_f_mA=jnp.asarray(i_f_hist),
        Cdl=1e-12,
        Ru=0.0,
    )

    assert np.isfinite(np.asarray(i_total_mA)).all()
    assert np.isfinite(np.asarray(i_cap_mA)).all()
    np.testing.assert_allclose(np.asarray(i_total_mA), np.asarray(i_f_hist), atol=1e-5, rtol=1e-5)
    assert float(np.max(np.abs(np.asarray(i_cap_mA)))) < 1e-4
