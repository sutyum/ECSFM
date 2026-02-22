import numpy as np
import pytest

from ecsfm.sim.cv import simulate_cv

def test_simulate_cv_stable():
    """
    Tests that the default CV simulation runs to completion 
    without generating NaN or Inf values in the output arrays.
    """
    x, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis = simulate_cv(
        nx=50,          # Reduced resolution for faster testing
        scan_rate=1.0   # Fast scan rate for fewer time steps
    )
    
    assert x is not None
    assert C_ox_hist is not None
    assert C_red_hist is not None
    assert I_hist is not None
    
    assert not np.isnan(C_ox_hist).any(), "C_ox contains NaN"
    assert not np.isnan(C_red_hist).any(), "C_red contains NaN"
    assert not np.isnan(I_hist).any(), "Current contains NaN"
    
    assert not np.isinf(C_ox_hist).any(), "C_ox contains Inf"
    assert not np.isinf(C_red_hist).any(), "C_red contains Inf"
    assert not np.isinf(I_hist).any(), "Current contains Inf"

def test_simulate_cv_bounds():
    """
    Tests that concentrations do not fall below zero (within floating point error)
    due to the ReLU constraints, and that current is physically bounded.
    """
    x, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis = simulate_cv(
        nx=50,
        scan_rate=1.0
    )
    
    # Concentrations should be rigidly non-negative
    assert np.min(C_ox_hist) >= -1e-10, f"C_ox dropped below zero: {np.min(C_ox_hist)}"
    assert np.min(C_red_hist) >= -1e-10, f"C_red dropped below zero: {np.min(C_red_hist)}"
    
    # Check that maximum current is not astronomically large (e.g. bounded below 10^5 mA/cm2)
    assert np.max(np.abs(I_hist)) < 100000, f"Current exploded to {np.max(np.abs(I_hist))} mA/cm2"


def test_simulate_cv_rejects_invalid_scan_rate():
    with pytest.raises(ValueError):
        simulate_cv(scan_rate=0.0, nx=50)


def test_simulate_cv_accepts_float32_inputs():
    import jax.numpy as jnp

    x, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis = simulate_cv(
        D_ox=jnp.float32(1e-5),
        D_red=jnp.float32(1e-5),
        C_bulk_ox=jnp.float32(1.0),
        C_bulk_red=jnp.float32(0.0),
        E0=jnp.float32(0.0),
        k0=jnp.float32(0.01),
        alpha=jnp.float32(0.5),
        scan_rate=jnp.float32(1.0),
        nx=50,
    )

    assert x.shape[0] == 50
    assert C_ox_hist.shape == C_red_hist.shape
    assert E_hist.shape == I_hist.shape
    assert E_hist_vis.shape[0] == C_ox_hist.shape[0]
