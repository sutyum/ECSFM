import jax.numpy as jnp
import numpy as np
import pytest

from ecsfm.sim.experiment import simulate_electrochem


def _single_species_params(dtype=jnp.float32):
    return dict(
        E_array=jnp.linspace(-0.2, 0.2, 20, dtype=dtype),
        t_max=0.2,
        D_ox=jnp.array([1e-5], dtype=dtype),
        D_red=jnp.array([1e-5], dtype=dtype),
        C_bulk_ox=jnp.array([1.0], dtype=dtype),
        C_bulk_red=jnp.array([0.0], dtype=dtype),
        E0=jnp.array([0.0], dtype=dtype),
        k0=jnp.array([0.01], dtype=dtype),
        alpha=jnp.array([0.5], dtype=dtype),
        nx=32,
        save_every=0,
    )


def test_simulate_electrochem_float32_stable():
    out = simulate_electrochem(**_single_species_params(dtype=jnp.float32))
    x, c_ox, c_red, e_hist, i_hist, e_vis, i_vis = out

    assert x.shape == (32,)
    assert c_ox.shape == c_red.shape
    assert e_hist.shape == i_hist.shape
    assert e_vis.shape == i_vis.shape
    assert not np.isnan(i_hist).any()


def test_simulate_electrochem_rejects_bad_inputs():
    params = _single_species_params()
    params["t_max"] = 0.0
    with pytest.raises(ValueError):
        simulate_electrochem(**params)
