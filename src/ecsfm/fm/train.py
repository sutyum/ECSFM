import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from pydantic import BaseModel, Field
from tqdm import tqdm

from ecsfm.fm.model import VectorFieldNet
from ecsfm.fm.objective import flow_matching_loss

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NORMALIZERS_FILENAME = "normalizers.npz"
MODEL_META_FILENAME = "model_meta.json"
OPTIONAL_DATASET_KEYS = {
    "task_id": 0,
    "stage_id": 0,
    "aug_id": 0,
}


class FlowConfig(BaseModel):
    n_samples: int = Field(0, ge=0, description="Number of trajectories to use (0 means all)")
    epochs: int = Field(500, ge=1, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, description="Batch size")
    lr: float = Field(1e-3, gt=0.0, description="Learning rate")
    weight_decay: float = Field(1e-4, ge=0.0, description="AdamW weight decay")
    hidden_size: int = Field(128, ge=1, description="Hidden size for VectorFieldNet")
    depth: int = Field(3, ge=1, description="Depth for VectorFieldNet")
    seed: int = Field(42, ge=0, description="Random seed")
    new_run: bool = Field(False, description="Start training from scratch, ignoring checkpoints")
    val_split: float = Field(0.2, gt=0.0, lt=1.0, description="Fraction of dataset to use for validation")
    patience: int = Field(0, ge=0, description="Early stopping patience (0 = disabled)")
    curriculum: bool = Field(True, description="Enable curriculum stage-based sampling when stage labels exist")
    partial_obs_training: bool = Field(
        False,
        description="Enable random masking of signal/conditioning during training for partial-observation robustness",
    )
    signal_mask_prob: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description="Per-timepoint masking probability for conditioning signal when partial_obs_training is enabled",
    )
    param_mask_prob: float = Field(
        0.15,
        ge=0.0,
        le=1.0,
        description="Per-parameter masking probability for conditioning vector when partial_obs_training is enabled",
    )


def integrate_flow(model: VectorFieldNet, x0: jax.Array, E: jax.Array, p: jax.Array, n_steps: int = 100) -> jax.Array:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    model_eval = eqx.nn.inference_mode(model)
    dt = 1.0 / n_steps
    x = x0.astype(jnp.float32)
    E = E.astype(jnp.float32)
    p = p.astype(jnp.float32)

    for i in range(n_steps):
        t = i * dt
        t_batch = jnp.full((x.shape[0], 1), t, dtype=x.dtype)
        v = jax.vmap(model_eval)(t_batch, x, E, p)
        x = x + v * dt
    return x


@eqx.filter_value_and_grad
def compute_loss(model, x1, x0, E, p, key):
    return flow_matching_loss(model, x1, x0, E, p, key)


@eqx.filter_jit
def compute_val_loss(model, x1, x0, E, p, key):
    return flow_matching_loss(model, x1, x0, E, p, key)


def _extract_optional_arrays(data: np.lib.npyio.NpzFile, n_rows: int) -> tuple[dict[str, np.ndarray], dict[str, bool], dict[str, list[str]]]:
    optional_arrays: dict[str, np.ndarray] = {}
    optional_present: dict[str, bool] = {}

    for key, default in OPTIONAL_DATASET_KEYS.items():
        if key in data:
            arr = np.asarray(data[key]).reshape(-1)
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"Optional key '{key}' has length {arr.shape[0]} but expected {n_rows}"
                )
            optional_arrays[key] = arr.astype(np.int32)
            optional_present[key] = True
        else:
            optional_arrays[key] = np.full((n_rows,), default, dtype=np.int32)
            optional_present[key] = False

    def _to_name_list(values: np.ndarray) -> list[str]:
        names: list[str] = []
        for value in np.asarray(values).tolist():
            if isinstance(value, (bytes, np.bytes_)):
                names.append(value.decode("utf-8"))
            else:
                names.append(str(value))
        return names

    label_names: dict[str, list[str]] = {}
    for key in ("task_names", "stage_names", "augmentation_names"):
        if key in data:
            label_names[key] = _to_name_list(np.asarray(data[key]))

    return optional_arrays, optional_present, label_names


def load_dataset(
    data_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if os.path.isdir(data_path):
        chunk_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
        if not chunk_files:
            raise FileNotFoundError(f"No .npz chunks found in {data_path}")

        all_ox, all_red, all_i, all_e, all_p = [], [], [], [], []
        optional_lists: dict[str, list[np.ndarray]] = {k: [] for k in OPTIONAL_DATASET_KEYS}
        optional_seen: dict[str, bool] = {k: False for k in OPTIONAL_DATASET_KEYS}
        label_names: dict[str, list[str]] = {}

        for chunk_file in tqdm(chunk_files, desc="Aggregating chunks"):
            data = np.load(chunk_file)
            c_ox = np.asarray(data["ox"])
            c_red = np.asarray(data["red"])
            curr = np.asarray(data["i"])
            sigs = np.asarray(data["e"])
            params = np.asarray(data["p"])

            if c_ox.ndim != 2 or c_red.ndim != 2 or curr.ndim != 2 or sigs.ndim != 2 or params.ndim != 2:
                raise ValueError(f"Chunk {chunk_file} contains non-2D arrays")

            n_rows = c_ox.shape[0]
            if not (c_red.shape[0] == n_rows == curr.shape[0] == sigs.shape[0] == params.shape[0]):
                raise ValueError(f"Chunk {chunk_file} has inconsistent row counts")

            opt_arr, opt_present, names = _extract_optional_arrays(data, n_rows)
            for key in OPTIONAL_DATASET_KEYS:
                optional_lists[key].append(opt_arr[key])
                optional_seen[key] = optional_seen[key] or opt_present[key]

            for key, value in names.items():
                if key not in label_names:
                    label_names[key] = value

            all_ox.append(c_ox)
            all_red.append(c_red)
            all_i.append(curr)
            all_e.append(sigs)
            all_p.append(params)

        c_ox = np.concatenate(all_ox, axis=0)
        c_red = np.concatenate(all_red, axis=0)
        curr = np.concatenate(all_i, axis=0)
        sigs = np.concatenate(all_e, axis=0)
        params = np.concatenate(all_p, axis=0)

        extras: dict[str, Any] = {
            key: (np.concatenate(optional_lists[key], axis=0) if optional_seen[key] else None)
            for key in OPTIONAL_DATASET_KEYS
        }
        extras.update(label_names)
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        data = np.load(data_path)
        c_ox = np.asarray(data["ox"])
        c_red = np.asarray(data["red"])
        curr = np.asarray(data["i"])
        sigs = np.asarray(data["e"])
        params = np.asarray(data["p"])

        if c_ox.ndim != 2 or c_red.ndim != 2 or curr.ndim != 2 or sigs.ndim != 2 or params.ndim != 2:
            raise ValueError("Expected all dataset arrays to be 2D.")

        n_rows = c_ox.shape[0]
        if not (c_red.shape[0] == n_rows == curr.shape[0] == sigs.shape[0] == params.shape[0]):
            raise ValueError("Dataset arrays must have the same number of rows (samples).")

        opt_arr, opt_present, names = _extract_optional_arrays(data, n_rows)
        extras = {
            key: (opt_arr[key] if opt_present[key] else None)
            for key in OPTIONAL_DATASET_KEYS
        }
        extras.update(names)

    return c_ox, c_red, curr, sigs, params, extras


def infer_geometry(
    c_ox: np.ndarray,
    c_red: np.ndarray,
    curr: np.ndarray,
    params: np.ndarray,
) -> tuple[int, int, int, int]:
    if c_ox.shape[1] != c_red.shape[1]:
        raise ValueError("ox and red state vectors must have the same width.")

    phys_dim = params.shape[1]
    if phys_dim % 7 != 0:
        raise ValueError(f"Physical conditioning dim must be divisible by 7, got {phys_dim}")

    max_species = phys_dim // 7
    if max_species <= 0:
        raise ValueError("max_species inferred as <= 0")

    species_state_dim = c_ox.shape[1]
    if species_state_dim % max_species != 0:
        raise ValueError(
            f"State width {species_state_dim} is not divisible by max_species {max_species}"
        )

    nx = species_state_dim // max_species
    target_len = curr.shape[1]
    state_dim = species_state_dim * 2 + target_len
    return max_species, nx, target_len, state_dim


def save_normalizers(
    path: Path,
    x_mean: jax.Array,
    x_std: jax.Array,
    e_mean: jax.Array,
    e_std: jax.Array,
    p_mean: jax.Array,
    p_std: jax.Array,
) -> None:
    np.savez(
        path,
        x_mean=np.asarray(x_mean),
        x_std=np.asarray(x_std),
        e_mean=np.asarray(e_mean),
        e_std=np.asarray(e_std),
        p_mean=np.asarray(p_mean),
        p_std=np.asarray(p_std),
    )


def load_saved_normalizers(normalizers_path: str | Path) -> tuple[jax.Array, ...]:
    normalizers_path = Path(normalizers_path)
    if not normalizers_path.exists():
        raise FileNotFoundError(f"Normalizer file not found at {normalizers_path}")

    data = np.load(normalizers_path)
    required = ("x_mean", "x_std", "e_mean", "e_std", "p_mean", "p_std")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Normalizer file missing keys: {missing}")

    return tuple(jnp.asarray(data[k]) for k in required)


def save_model_metadata(meta_path: Path, metadata: dict[str, Any]) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_model_metadata(meta_path: str | Path) -> dict[str, Any]:
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_loss_curve(history: dict[str, list], output_path: Path) -> None:
    plt.figure()
    if history.get("train"):
        plt.plot(range(len(history["train"])), history["train"], label="Train Loss")
    if history.get("val"):
        val_epochs, val_losses = zip(*history["val"])
        plt.plot(val_epochs, val_losses, label="Val Loss", marker="o")

    plt.title("Flow Matching Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("OT-CFM Loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_comparison(
    model: VectorFieldNet,
    epoch: int,
    key: jax.Array,
    val_x: jax.Array,
    val_e: jax.Array,
    val_p: jax.Array,
    x_mean: jax.Array,
    x_std: jax.Array,
    nx: int,
    max_species: int,
    target_len: int,
    output_path: Path,
) -> None:
    del target_len
    if val_x.shape[0] == 0:
        return

    n_samples = int(min(4, val_x.shape[0]))
    x_cond = val_x[:n_samples]
    e_cond = val_e[:n_samples]
    p_cond = val_p[:n_samples]

    x0 = jax.random.normal(key, (n_samples, val_x.shape[1]))
    x_generated_norm = integrate_flow(model, x0, e_cond, p_cond, n_steps=100)

    gt_x = np.asarray(x_cond * x_std + x_mean)
    gen_x = np.asarray(x_generated_norm * x_std + x_mean)

    species_state_dim = nx * max_species

    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 4 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(f"Surrogate Comparison - Epoch {epoch}", fontsize=14)

    for i in range(n_samples):
        gt_ox = gt_x[i, :species_state_dim].reshape(max_species, nx)
        gt_red = gt_x[i, species_state_dim : 2 * species_state_dim].reshape(max_species, nx)
        gt_curr = gt_x[i, 2 * species_state_dim :]

        pred_ox = gen_x[i, :species_state_dim].reshape(max_species, nx)
        pred_red = gen_x[i, species_state_dim : 2 * species_state_dim].reshape(max_species, nx)
        pred_curr = gen_x[i, 2 * species_state_dim :]

        ax_states = axes[i, 0]
        ax_states.plot(pred_ox[0], label="Pred Ox", color="blue")
        ax_states.plot(pred_red[0], label="Pred Red", color="red")
        ax_states.plot(gt_ox[0], label="GT Ox", color="blue", linestyle="--")
        ax_states.plot(gt_red[0], label="GT Red", color="red", linestyle="--")
        ax_states.set_title("Species 1 Final Profiles")
        ax_states.legend()

        ax_current = axes[i, 1]
        ax_current.plot(pred_curr, label="Pred Current", color="green")
        ax_current.plot(gt_curr, label="GT Current", color="green", linestyle="--")
        ax_current.set_title("Current Trace")
        ax_current.legend()

        ax_error = axes[i, 2]
        ax_error.plot(pred_curr - gt_curr, color="purple")
        ax_error.set_title("Current Error")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _build_task_stage_features(
    task_id: np.ndarray,
    stage_id: np.ndarray,
    n_tasks_hint: int | None = None,
    n_stages_hint: int | None = None,
) -> tuple[np.ndarray, int, int]:
    n_tasks = max(1, int(np.max(task_id)) + 1)
    n_stages = max(1, int(np.max(stage_id)) + 1)
    if n_tasks_hint is not None:
        n_tasks = max(n_tasks, int(n_tasks_hint))
    if n_stages_hint is not None:
        n_stages = max(n_stages, int(n_stages_hint))

    task_oh = np.eye(n_tasks, dtype=np.float32)[task_id]
    stage_oh = np.eye(n_stages, dtype=np.float32)[stage_id]
    return np.concatenate([task_oh, stage_oh], axis=1), n_tasks, n_stages


def _build_signal_input(
    signal_norm: jax.Array,
    signal_mask: jax.Array | None,
    signal_channels: int,
) -> jax.Array:
    if signal_channels not in (1, 2):
        raise ValueError(f"signal_channels must be 1 or 2, got {signal_channels}")
    if signal_norm.ndim != 2:
        raise ValueError(f"signal_norm must be 2D [batch, len], got shape {signal_norm.shape}")

    if signal_mask is None:
        signal_mask = jnp.ones_like(signal_norm)
    elif signal_mask.shape != signal_norm.shape:
        raise ValueError(
            f"signal_mask shape mismatch: expected {signal_norm.shape}, got {signal_mask.shape}"
        )

    signal_mask = signal_mask.astype(signal_norm.dtype)
    masked_signal = signal_norm * signal_mask
    if signal_channels == 1:
        return masked_signal
    return jnp.stack([masked_signal, signal_mask], axis=1)


def _build_param_input(
    params_norm: jax.Array,
    param_mask: jax.Array | None,
    append_mask_features: bool,
) -> jax.Array:
    if params_norm.ndim != 2:
        raise ValueError(f"params_norm must be 2D [batch, dim], got shape {params_norm.shape}")

    if param_mask is None:
        param_mask = jnp.ones_like(params_norm)
    elif param_mask.shape != params_norm.shape:
        raise ValueError(
            f"param_mask shape mismatch: expected {params_norm.shape}, got {param_mask.shape}"
        )

    param_mask = param_mask.astype(params_norm.dtype)
    masked = params_norm * param_mask
    if append_mask_features:
        return jnp.concatenate([masked, param_mask], axis=1)
    return masked


def train_surrogate(config: FlowConfig, data_path: str, artifact_dir: str) -> None:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = artifact_path / "surrogate_model.eqx"
    history_path = artifact_path / "training_history.json"
    loss_curve_path = artifact_path / "loss_curve.png"
    normalizers_path = artifact_path / NORMALIZERS_FILENAME
    model_meta_path = artifact_path / MODEL_META_FILENAME

    key = jax.random.PRNGKey(np.uint32(config.seed))

    c_ox, c_red, curr, sigs, params_base, extras = load_dataset(data_path)
    task_id = extras.get("task_id")
    stage_id = extras.get("stage_id")
    task_names_raw = extras.get("task_names")
    stage_names_raw = extras.get("stage_names")
    has_task_stage_labels = (task_id is not None) or (stage_id is not None)

    if config.n_samples > 0:
        n_keep = min(config.n_samples, c_ox.shape[0])
        c_ox = c_ox[:n_keep]
        c_red = c_red[:n_keep]
        curr = curr[:n_keep]
        sigs = sigs[:n_keep]
        params_base = params_base[:n_keep]
        if task_id is not None:
            task_id = task_id[:n_keep]
        if stage_id is not None:
            stage_id = stage_id[:n_keep]

    if c_ox.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 trajectories after filtering to train+validate, got {c_ox.shape[0]}"
        )

    max_species, nx, target_len, state_dim = infer_geometry(c_ox, c_red, curr, params_base)

    use_task_stage_features = has_task_stage_labels or task_names_raw is not None or stage_names_raw is not None
    if use_task_stage_features:
        if task_id is None:
            task_id = np.zeros((c_ox.shape[0],), dtype=np.int32)
        if stage_id is None:
            stage_id = np.zeros((c_ox.shape[0],), dtype=np.int32)

        n_tasks_hint = len(task_names_raw) if task_names_raw is not None else None
        n_stages_hint = len(stage_names_raw) if stage_names_raw is not None else None
        task_stage_features, n_tasks, n_stages = _build_task_stage_features(
            task_id=task_id,
            stage_id=stage_id,
            n_tasks_hint=n_tasks_hint,
            n_stages_hint=n_stages_hint,
        )
        params_cond_core = np.concatenate([params_base.astype(np.float32), task_stage_features], axis=1)
    else:
        n_tasks = 0
        n_stages = 0
        task_id = np.zeros((c_ox.shape[0],), dtype=np.int32)
        stage_id = np.zeros((c_ox.shape[0],), dtype=np.int32)
        params_cond_core = params_base.astype(np.float32)

    c_ox = jnp.asarray(c_ox)
    c_red = jnp.asarray(c_red)
    curr = jnp.asarray(curr)
    sigs = jnp.asarray(sigs)
    params_cond_core = jnp.asarray(params_cond_core)
    task_id = jnp.asarray(task_id)
    stage_id = jnp.asarray(stage_id)

    dataset_x = jnp.concatenate([c_ox, c_red, curr], axis=1)

    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, len(dataset_x))
    dataset_x = dataset_x[indices]
    sigs = sigs[indices]
    params_cond_core = params_cond_core[indices]
    task_id = task_id[indices]
    stage_id = stage_id[indices]

    total = int(len(dataset_x))
    val_size = int(total * config.val_split)
    val_size = min(max(1, val_size), total - 1)

    train_x = dataset_x[:-val_size]
    val_x = dataset_x[-val_size:]
    train_e_base = sigs[:-val_size]
    val_e_base = sigs[-val_size:]
    train_p_core = params_cond_core[:-val_size]
    val_p_core = params_cond_core[-val_size:]
    train_stage = np.asarray(stage_id[:-val_size])

    x_mean = jnp.mean(train_x, axis=0)
    x_std = jnp.std(train_x, axis=0) + 1e-5
    e_mean = jnp.mean(train_e_base, axis=0)
    e_std = jnp.std(train_e_base, axis=0) + 1e-5
    p_mean = jnp.mean(train_p_core, axis=0)
    p_std = jnp.std(train_p_core, axis=0) + 1e-5

    train_x = (train_x - x_mean) / x_std
    val_x = (val_x - x_mean) / x_std
    train_e_base = (train_e_base - e_mean) / e_std
    val_e_base = (val_e_base - e_mean) / e_std
    train_p_core = (train_p_core - p_mean) / p_std
    val_p_core = (val_p_core - p_mean) / p_std

    signal_channels = 2 if config.partial_obs_training else 1
    append_param_mask_features = bool(config.partial_obs_training)
    phys_dim_core = int(train_p_core.shape[1])
    model_phys_dim = phys_dim_core * (2 if append_param_mask_features else 1)

    # Fixed key for deterministic validation masking (reproducible val loss across epochs).
    val_mask_key = jax.random.PRNGKey(np.uint32(config.seed) ^ np.uint32(0xDEAD))

    def _build_val_inputs(val_e_base_arr, val_p_core_arr):
        """Build validation inputs with the same masking distribution as training."""
        if config.partial_obs_training:
            vk1, vk2 = jax.random.split(val_mask_key)
            val_e_mask = jax.random.bernoulli(
                vk1,
                p=1.0 - config.signal_mask_prob,
                shape=val_e_base_arr.shape,
            ).astype(val_e_base_arr.dtype)
            val_e_mask = val_e_mask.at[:, 0].set(jnp.asarray(1.0, dtype=val_e_base_arr.dtype))
            val_e_mask = val_e_mask.at[:, -1].set(jnp.asarray(1.0, dtype=val_e_base_arr.dtype))
            val_p_mask = jax.random.bernoulli(
                vk2,
                p=1.0 - config.param_mask_prob,
                shape=val_p_core_arr.shape,
            ).astype(val_p_core_arr.dtype)
        else:
            val_e_mask = None
            val_p_mask = None
        val_e = _build_signal_input(
            signal_norm=val_e_base_arr,
            signal_mask=val_e_mask,
            signal_channels=signal_channels,
        )
        val_p = _build_param_input(
            params_norm=val_p_core_arr,
            param_mask=val_p_mask,
            append_mask_features=append_param_mask_features,
        )
        return val_e, val_p

    val_e, val_p = _build_val_inputs(val_e_base, val_p_core)

    print(f"Train size: {len(train_x)}, Val size: {len(val_x)}")
    print(
        f"Model dimensions: state_dim={state_dim}, max_species={max_species}, "
        f"nx={nx}, target_len={target_len}, phys_dim={model_phys_dim}, "
        f"phys_dim_core={phys_dim_core}, signal_channels={signal_channels}, "
        f"tasks={n_tasks}, stages={n_stages}, curriculum={config.curriculum}, "
        f"partial_obs_training={config.partial_obs_training}"
    )

    save_normalizers(
        normalizers_path,
        x_mean=x_mean,
        x_std=x_std,
        e_mean=e_mean,
        e_std=e_std,
        p_mean=p_mean,
        p_std=p_std,
    )

    if n_tasks == 0:
        task_names = []
    elif task_names_raw is None:
        task_names = [f"task_{i}" for i in range(n_tasks)]
    else:
        task_names = list(task_names_raw)[:n_tasks]
        if len(task_names) < n_tasks:
            task_names.extend(f"task_{i}" for i in range(len(task_names), n_tasks))

    if n_stages == 0:
        stage_names = []
    elif stage_names_raw is None:
        stage_names = [f"stage_{i}" for i in range(n_stages)]
    else:
        stage_names = list(stage_names_raw)[:n_stages]
        if len(stage_names) < n_stages:
            stage_names.extend(f"stage_{i}" for i in range(len(stage_names), n_stages))

    model_meta = {
        "state_dim": int(state_dim),
        "max_species": int(max_species),
        "nx": int(nx),
        "target_len": int(target_len),
        "phys_dim_base": int(params_base.shape[1]),
        "phys_dim_core": int(phys_dim_core),
        "phys_dim": int(model_phys_dim),
        "param_mask_features": bool(append_param_mask_features),
        "signal_channels": int(signal_channels),
        "n_tasks": int(n_tasks),
        "n_stages": int(n_stages),
        "cond_dim": 32,
        "hidden_size": int(config.hidden_size),
        "depth": int(config.depth),
        "seed": int(config.seed),
        "val_split": float(config.val_split),
        "curriculum_enabled": bool(config.curriculum),
        "task_names": task_names,
        "stage_names": stage_names,
        "normalizers_file": NORMALIZERS_FILENAME,
    }
    save_model_metadata(model_meta_path, model_meta)

    key, subkey = jax.random.split(key)
    model = VectorFieldNet(
        state_dim=state_dim,
        hidden_size=config.hidden_size,
        depth=config.depth,
        cond_dim=32,
        phys_dim=model_phys_dim,
        signal_channels=signal_channels,
        key=subkey,
    )

    start_epoch = 0
    history: dict[str, list] = {"train": [], "val": []}

    if checkpoint_path.exists() and not config.new_run:
        try:
            model = eqx.tree_deserialise_leaves(checkpoint_path, model)
            if history_path.exists():
                with open(history_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    history = saved.get("history", history)
                    start_epoch = int(saved.get("epoch", 0))
            print(f"Resumed checkpoint from {checkpoint_path} at epoch {start_epoch}")
        except Exception as exc:
            print(f"Checkpoint load failed ({exc}); starting new run.")

    warmup_epochs = max(1, config.epochs // 10)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.lr * 0.1,
        peak_value=config.lr,
        warmup_steps=warmup_epochs,
        decay_steps=config.epochs,
        end_value=config.lr * 0.01,
    )
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, x1, x0, E, p, step_key):
        loss, grads = compute_loss(model, x1, x0, E, p, step_key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    best_val_loss = float("inf")
    patience_counter = 0
    best_checkpoint_path = artifact_path / "surrogate_model_best.eqx"
    checkpoint_interval = max(1, min(500, config.epochs // 10))

    pbar = tqdm(range(start_epoch, config.epochs), desc="Training")
    for epoch in pbar:
        if config.curriculum and n_stages > 1:
            progress = epoch / max(1, config.epochs - 1)
            allowed_stage = min(n_stages - 1, int(np.floor(progress * n_stages)))
            eligible = np.where(train_stage <= allowed_stage)[0]
            if eligible.size < config.batch_size:
                eligible = np.arange(len(train_x))
            key, subkey = jax.random.split(key)
            perm_local = np.asarray(jax.random.permutation(subkey, eligible.size))
            epoch_indices = eligible[perm_local]
        else:
            allowed_stage = n_stages - 1
            key, subkey = jax.random.split(key)
            epoch_indices = np.asarray(jax.random.permutation(subkey, len(train_x)))

        shuffled_x1 = train_x[epoch_indices]
        shuffled_e_base = train_e_base[epoch_indices]
        shuffled_p_core = train_p_core[epoch_indices]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(shuffled_x1), config.batch_size):
            batch_x1 = shuffled_x1[i : i + config.batch_size]
            batch_e_base = shuffled_e_base[i : i + config.batch_size]
            batch_p_core = shuffled_p_core[i : i + config.batch_size]

            if config.partial_obs_training:
                key, e_mask_key, p_mask_key = jax.random.split(key, 3)
                e_keep_prob = 1.0 - config.signal_mask_prob
                p_keep_prob = 1.0 - config.param_mask_prob

                e_mask = jax.random.bernoulli(
                    e_mask_key,
                    p=e_keep_prob,
                    shape=batch_e_base.shape,
                ).astype(batch_e_base.dtype)
                # Keep endpoints observed to avoid fully unanchored signals.
                e_mask = e_mask.at[:, 0].set(jnp.asarray(1.0, dtype=batch_e_base.dtype))
                e_mask = e_mask.at[:, -1].set(jnp.asarray(1.0, dtype=batch_e_base.dtype))

                p_mask = jax.random.bernoulli(
                    p_mask_key,
                    p=p_keep_prob,
                    shape=batch_p_core.shape,
                ).astype(batch_p_core.dtype)
            else:
                e_mask = None
                p_mask = None

            batch_e = _build_signal_input(
                signal_norm=batch_e_base,
                signal_mask=e_mask,
                signal_channels=signal_channels,
            )
            batch_p = _build_param_input(
                params_norm=batch_p_core,
                param_mask=p_mask,
                append_mask_features=append_param_mask_features,
            )

            key, sample_key, step_key = jax.random.split(key, 3)
            batch_x0 = jax.random.normal(sample_key, batch_x1.shape)

            model, opt_state, loss = make_step(model, opt_state, batch_x1, batch_x0, batch_e, batch_p, step_key)
            epoch_loss += float(loss)
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches were formed. Check dataset size and batch_size.")

        avg_loss = epoch_loss / n_batches
        history["train"].append(avg_loss)

        # Validate every epoch with inference_mode (dropout disabled)
        key, sample_key, step_key = jax.random.split(key, 3)
        val_x0 = jax.random.normal(sample_key, val_x.shape)
        model_eval = eqx.nn.inference_mode(model)
        val_loss = float(compute_val_loss(model_eval, val_x, val_x0, val_e, val_p, step_key))
        history["val"].append((epoch, val_loss))
        pbar.set_postfix(
            {
                "train": f"{avg_loss:.5f}",
                "val": f"{val_loss:.5f}",
                "stage<=": str(allowed_stage),
            }
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            eqx.tree_serialise_leaves(best_checkpoint_path, model)
        else:
            patience_counter += 1

        if config.patience > 0 and patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch} (patience={config.patience})")
            break

        if epoch > start_epoch and epoch % checkpoint_interval == 0:
            eqx.tree_serialise_leaves(checkpoint_path, model)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump({"history": history, "epoch": epoch}, f)
            save_loss_curve(history, loss_curve_path)

            key, cmp_key = jax.random.split(key)
            save_comparison(
                model=model,
                epoch=epoch,
                key=cmp_key,
                val_x=val_x,
                val_e=val_e,
                val_p=val_p,
                x_mean=x_mean,
                x_std=x_std,
                nx=nx,
                max_species=max_species,
                target_len=target_len,
                output_path=artifact_path / f"surrogate_comparison_ep{epoch}.png",
            )

    # Restore best model if early stopping was used and a best checkpoint exists
    if best_checkpoint_path.exists():
        try:
            model = eqx.tree_deserialise_leaves(best_checkpoint_path, model)
            print(f"Restored best model (val_loss={best_val_loss:.5f})")
        except Exception:
            pass

    final_loss = history["train"][-1] if history["train"] else 0.0
    print(f"Final epoch loss: {final_loss:.5f}")

    eqx.tree_serialise_leaves(checkpoint_path, model)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "epoch": config.epochs}, f)

    save_loss_curve(history, loss_curve_path)
    key, cmp_key = jax.random.split(key)
    save_comparison(
        model=model,
        epoch=config.epochs,
        key=cmp_key,
        val_x=val_x,
        val_e=val_e,
        val_p=val_p,
        x_mean=x_mean,
        x_std=x_std,
        nx=nx,
        max_species=max_species,
        target_len=target_len,
        output_path=artifact_path / f"surrogate_comparison_ep{config.epochs}.png",
    )

    print(f"Saved model to {checkpoint_path}")
    print(f"Saved history to {history_path}")
    print(f"Saved normalizers to {normalizers_path}")
    print(f"Saved model metadata to {model_meta_path}")


def _load_config_defaults(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON object, got {type(raw)}")
    return raw


def _cfg(defaults: dict[str, Any], key: str, fallback: Any, aliases: tuple[str, ...] = ()) -> Any:
    for candidate in (key, *aliases):
        if candidate in defaults:
            return defaults[candidate]
    return fallback


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    base_args, _ = base_parser.parse_known_args(argv)

    file_defaults = _load_config_defaults(base_args.config)

    parser = argparse.ArgumentParser(
        description="Train Flow Matching surrogate",
        parents=[base_parser],
    )

    parser.add_argument(
        "--dataset",
        "--data-path",
        dest="data_path",
        type=str,
        default=_cfg(file_defaults, "data_path", "/tmp/ecsfm/dataset_massive", aliases=("dataset",)),
        help="Path to NPZ file or directory of NPZ chunks",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=_cfg(file_defaults, "artifact_dir", "/tmp/ecsfm"),
        help="Output directory for model/artifacts",
    )
    parser.add_argument("--n-samples", type=int, default=int(_cfg(file_defaults, "n_samples", 0)), help="Samples to use (0 = all)")
    parser.add_argument("--epochs", type=int, default=int(_cfg(file_defaults, "epochs", 500)), help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=int(_cfg(file_defaults, "batch_size", 32)), help="Batch size")
    parser.add_argument("--lr", type=float, default=float(_cfg(file_defaults, "lr", 1e-3)), help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=float(_cfg(file_defaults, "weight_decay", 1e-4)), help="AdamW weight decay")
    parser.add_argument("--hidden-size", type=int, default=int(_cfg(file_defaults, "hidden_size", 128)), help="MLP hidden size")
    parser.add_argument("--depth", type=int, default=int(_cfg(file_defaults, "depth", 3)), help="MLP depth")
    parser.add_argument("--seed", type=int, default=int(_cfg(file_defaults, "seed", 42)), help="Random seed")
    parser.add_argument(
        "--new-run",
        action="store_true",
        default=bool(_cfg(file_defaults, "new_run", False)),
        help="Ignore checkpoint and start from scratch",
    )
    parser.add_argument(
        "--resume",
        dest="new_run",
        action="store_false",
        help="Resume from checkpoint if available (overrides config new_run=true)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=bool(_cfg(file_defaults, "curriculum", True)),
        help="Enable curriculum stage-based sampling",
    )
    parser.add_argument(
        "--no-curriculum",
        dest="curriculum",
        action="store_false",
        help="Disable curriculum stage-based sampling",
    )
    parser.add_argument(
        "--partial-obs-training",
        action="store_true",
        default=bool(_cfg(file_defaults, "partial_obs_training", False, aliases=("partial_obs",))),
        help="Enable random masking of signal/conditioning during training",
    )
    parser.add_argument(
        "--no-partial-obs-training",
        dest="partial_obs_training",
        action="store_false",
        help="Disable random masking of signal/conditioning during training",
    )
    parser.add_argument(
        "--signal-mask-prob",
        type=float,
        default=float(_cfg(file_defaults, "signal_mask_prob", 0.15)),
        help="Per-timepoint masking probability for conditioning signals",
    )
    parser.add_argument(
        "--param-mask-prob",
        type=float,
        default=float(_cfg(file_defaults, "param_mask_prob", 0.15)),
        help="Per-parameter masking probability for conditioning vector",
    )
    parser.add_argument("--val-split", type=float, default=float(_cfg(file_defaults, "val_split", 0.2)), help="Validation fraction")
    parser.add_argument("--patience", type=int, default=int(_cfg(file_defaults, "patience", 0)), help="Early stopping patience (0 = disabled)")

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    config = FlowConfig(
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        depth=args.depth,
        seed=args.seed,
        new_run=args.new_run,
        val_split=args.val_split,
        patience=args.patience,
        curriculum=args.curriculum,
        partial_obs_training=args.partial_obs_training,
        signal_mask_prob=args.signal_mask_prob,
        param_mask_prob=args.param_mask_prob,
    )

    train_surrogate(config=config, data_path=args.data_path, artifact_dir=args.artifact_dir)


if __name__ == "__main__":
    main()
