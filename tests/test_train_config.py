import json

import jax.numpy as jnp
import numpy as np

from ecsfm.fm.train import (
    load_model_metadata,
    load_saved_normalizers,
    parse_args,
    save_model_metadata,
    save_normalizers,
)


def test_parse_args_uses_config_defaults(tmp_path):
    config_path = tmp_path / "config.json"
    config = {
        "data_path": "/tmp/data_path",
        "artifact_dir": "/tmp/artifacts",
        "n_samples": 123,
        "epochs": 456,
        "batch_size": 7,
        "lr": 0.007,
        "hidden_size": 96,
        "depth": 5,
        "seed": 99,
        "new_run": True,
        "val_split": 0.33,
        "curriculum": False,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    args = parse_args(["--config", str(config_path)])

    assert args.data_path == "/tmp/data_path"
    assert args.artifact_dir == "/tmp/artifacts"
    assert args.n_samples == 123
    assert args.epochs == 456
    assert args.batch_size == 7
    assert np.isclose(args.lr, 0.007)
    assert args.hidden_size == 96
    assert args.depth == 5
    assert args.seed == 99
    assert args.new_run is True
    assert np.isclose(args.val_split, 0.33)
    assert args.curriculum is False


def test_parse_args_cli_overrides_config(tmp_path):
    config_path = tmp_path / "config.json"
    config = {
        "lr": 0.001,
        "depth": 3,
        "new_run": True,
        "curriculum": False,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--lr",
            "0.25",
            "--depth",
            "9",
            "--resume",
            "--curriculum",
        ]
    )

    assert np.isclose(args.lr, 0.25)
    assert args.depth == 9
    assert args.new_run is False
    assert args.curriculum is True


def test_normalizer_and_metadata_roundtrip(tmp_path):
    normalizers_path = tmp_path / "normalizers.npz"
    meta_path = tmp_path / "model_meta.json"

    save_normalizers(
        normalizers_path,
        x_mean=jnp.array([1.0, 2.0]),
        x_std=jnp.array([0.1, 0.2]),
        e_mean=jnp.array([0.3]),
        e_std=jnp.array([0.05]),
        p_mean=jnp.array([4.0, 5.0, 6.0]),
        p_std=jnp.array([0.4, 0.5, 0.6]),
    )

    loaded = load_saved_normalizers(normalizers_path)
    assert len(loaded) == 6
    assert np.allclose(np.asarray(loaded[0]), np.array([1.0, 2.0]))
    assert np.allclose(np.asarray(loaded[4]), np.array([4.0, 5.0, 6.0]))

    meta = {"state_dim": 700, "hidden_size": 128, "depth": 3}
    save_model_metadata(meta_path, meta)
    loaded_meta = load_model_metadata(meta_path)
    assert loaded_meta == meta
