import jax.numpy as jnp
import numpy as np

from ecsfm.fm.eval_classical import _build_conditioning_vector, _resolve_model_geometry


def test_build_conditioning_vector_appends_task_and_stage_one_hot():
    flat_params = np.arange(6, dtype=np.float32)
    out = _build_conditioning_vector(
        flat_base_params=flat_params,
        n_tasks=3,
        n_stages=2,
        task_idx=1,
        stage_idx=0,
    )

    assert out.shape == (11,)
    assert np.allclose(out[:6], flat_params)
    assert np.allclose(out[6:9], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.allclose(out[9:], np.array([1.0, 0.0], dtype=np.float32))


def test_resolve_model_geometry_supports_multitask_phys_dim():
    state_dim = 100
    target_len = 20
    phys_dim = 35 + 8 + 3

    norm = (
        jnp.zeros((state_dim,)),
        jnp.ones((state_dim,)),
        jnp.zeros((target_len,)),
        jnp.ones((target_len,)),
        jnp.zeros((phys_dim,)),
        jnp.ones((phys_dim,)),
    )
    meta = {
        "max_species": 5,
        "phys_dim_base": 35,
        "n_tasks": 8,
        "n_stages": 3,
        "task_names": [f"task_{i}" for i in range(8)],
        "stage_names": ["foundation", "bridge", "frontier"],
    }

    geometry = _resolve_model_geometry(norm, meta)
    assert geometry["state_dim"] == state_dim
    assert geometry["target_len"] == target_len
    assert geometry["phys_dim"] == phys_dim
    assert geometry["max_species"] == 5
    assert geometry["nx"] == 8
    assert geometry["n_tasks"] == 8
    assert geometry["n_stages"] == 3


def test_resolve_model_geometry_supports_mask_augmented_conditioning():
    state_dim = 120
    target_len = 20
    phys_dim_core = 35 + 8 + 3
    phys_dim = phys_dim_core * 2

    norm = (
        jnp.zeros((state_dim,)),
        jnp.ones((state_dim,)),
        jnp.zeros((target_len,)),
        jnp.ones((target_len,)),
        jnp.zeros((phys_dim_core,)),
        jnp.ones((phys_dim_core,)),
    )
    meta = {
        "max_species": 5,
        "phys_dim_base": 35,
        "phys_dim_core": phys_dim_core,
        "phys_dim": phys_dim,
        "n_tasks": 8,
        "n_stages": 3,
        "param_mask_features": True,
        "signal_channels": 2,
        "task_names": [f"task_{i}" for i in range(8)],
        "stage_names": ["foundation", "bridge", "frontier"],
    }

    geometry = _resolve_model_geometry(norm, meta)
    assert geometry["phys_dim"] == phys_dim
    assert geometry["phys_dim_core"] == phys_dim_core
    assert geometry["param_mask_features"] is True
    assert geometry["signal_channels"] == 2
