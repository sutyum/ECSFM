import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ecsfm.fm.model import VectorFieldNet
from ecsfm.fm.train import _build_param_input, _build_signal_input


def test_build_signal_input_one_channel_applies_mask():
    signal = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    mask = jnp.asarray([[1.0, 0.0, 1.0]], dtype=jnp.float32)
    out = _build_signal_input(signal_norm=signal, signal_mask=mask, signal_channels=1)

    assert out.shape == (1, 3)
    np.testing.assert_allclose(np.asarray(out), np.asarray([[1.0, 0.0, 3.0]], dtype=np.float32))


def test_build_signal_input_two_channels_emits_value_and_mask():
    signal = jnp.asarray([[1.0, -1.0, 2.0]], dtype=jnp.float32)
    mask = jnp.asarray([[0.0, 1.0, 1.0]], dtype=jnp.float32)
    out = _build_signal_input(signal_norm=signal, signal_mask=mask, signal_channels=2)

    assert out.shape == (1, 2, 3)
    np.testing.assert_allclose(np.asarray(out[0, 0]), np.asarray([0.0, -1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(out[0, 1]), np.asarray([0.0, 1.0, 1.0], dtype=np.float32))


def test_build_param_input_appends_binary_mask_features():
    params = jnp.asarray([[0.5, -1.0, 2.0]], dtype=jnp.float32)
    mask = jnp.asarray([[1.0, 0.0, 1.0]], dtype=jnp.float32)
    out = _build_param_input(params_norm=params, param_mask=mask, append_mask_features=True)

    assert out.shape == (1, 6)
    np.testing.assert_allclose(np.asarray(out[0, :3]), np.asarray([0.5, 0.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(out[0, 3:]), np.asarray([1.0, 0.0, 1.0], dtype=np.float32))


def test_vector_field_net_accepts_two_channel_signal_condition():
    key = jax.random.PRNGKey(0)
    model = VectorFieldNet(
        state_dim=16,
        hidden_size=32,
        depth=2,
        cond_dim=8,
        phys_dim=10,
        signal_channels=2,
        key=key,
    )

    t = jnp.asarray([0.2], dtype=jnp.float32)
    x = jnp.zeros((16,), dtype=jnp.float32)
    e = jnp.zeros((2, 24), dtype=jnp.float32)
    p = jnp.zeros((10,), dtype=jnp.float32)

    out = model(t, x, e, p)
    assert out.shape == (16,)
    assert np.isfinite(np.asarray(out)).all()


def test_vector_field_net_rejects_wrong_signal_channels():
    key = jax.random.PRNGKey(1)
    model = VectorFieldNet(
        state_dim=8,
        hidden_size=16,
        depth=1,
        cond_dim=4,
        phys_dim=6,
        signal_channels=2,
        key=key,
    )

    t = jnp.asarray([0.5], dtype=jnp.float32)
    x = jnp.zeros((8,), dtype=jnp.float32)
    e = jnp.zeros((24,), dtype=jnp.float32)  # 1-channel input should fail
    p = jnp.zeros((6,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Signal channels mismatch"):
        _ = model(t, x, e, p)
