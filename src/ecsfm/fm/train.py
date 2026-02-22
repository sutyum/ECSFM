import argparse
import glob
import json
import os
from pathlib import Path

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


class FlowConfig(BaseModel):
    n_samples: int = Field(0, description="Number of trajectories to use (0 means all)")
    epochs: int = Field(500, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size")
    lr: float = Field(1e-3, description="Learning rate")
    hidden_size: int = Field(128, description="Hidden size for VectorFieldNet")
    depth: int = Field(3, description="Depth for VectorFieldNet")
    seed: int = Field(42, description="Random seed")
    new_run: bool = Field(False, description="Start training from scratch, ignoring checkpoints")
    val_split: float = Field(0.2, description="Fraction of dataset to use for validation")


def integrate_flow(model: VectorFieldNet, x0: jax.Array, E: jax.Array, p: jax.Array, n_steps: int = 100) -> jax.Array:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    dt = 1.0 / n_steps
    x = x0.astype(jnp.float32)
    E = E.astype(jnp.float32)
    p = p.astype(jnp.float32)

    for i in range(n_steps):
        t = i * dt
        t_batch = jnp.full((x.shape[0], 1), t, dtype=x.dtype)
        v = jax.vmap(model)(t_batch, x, E, p)
        x = x + v * dt
    return x


@eqx.filter_value_and_grad
def compute_loss(model, x1, x0, E, p, key):
    return flow_matching_loss(model, x1, x0, E, p, key)


@eqx.filter_jit
def compute_val_loss(model, x1, x0, E, p, key):
    return flow_matching_loss(model, x1, x0, E, p, key)


def load_dataset(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if os.path.isdir(data_path):
        chunk_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
        if not chunk_files:
            raise FileNotFoundError(f"No .npz chunks found in {data_path}")

        all_ox, all_red, all_i, all_e, all_p = [], [], [], [], []
        for chunk_file in tqdm(chunk_files, desc="Aggregating chunks"):
            data = np.load(chunk_file)
            all_ox.append(data["ox"])
            all_red.append(data["red"])
            all_i.append(data["i"])
            all_e.append(data["e"])
            all_p.append(data["p"])

        c_ox = np.concatenate(all_ox, axis=0)
        c_red = np.concatenate(all_red, axis=0)
        curr = np.concatenate(all_i, axis=0)
        sigs = np.concatenate(all_e, axis=0)
        params = np.concatenate(all_p, axis=0)
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        data = np.load(data_path)
        c_ox = data["ox"]
        c_red = data["red"]
        curr = data["i"]
        sigs = data["e"]
        params = data["p"]

    if c_ox.ndim != 2 or c_red.ndim != 2 or curr.ndim != 2 or sigs.ndim != 2 or params.ndim != 2:
        raise ValueError("Expected all dataset arrays to be 2D.")

    n = c_ox.shape[0]
    if not (c_red.shape[0] == n == curr.shape[0] == sigs.shape[0] == params.shape[0]):
        raise ValueError("Dataset arrays must have the same number of rows (samples).")

    return c_ox, c_red, curr, sigs, params


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


def train_surrogate(config: FlowConfig, data_path: str, artifact_dir: str) -> None:
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.epochs}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if config.lr <= 0:
        raise ValueError(f"lr must be positive, got {config.lr}")
    if not (0.0 < config.val_split < 1.0):
        raise ValueError(f"val_split must be in (0, 1), got {config.val_split}")

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_path / "surrogate_model.eqx"
    history_path = artifact_path / "training_history.json"
    loss_curve_path = artifact_path / "loss_curve.png"

    key = jax.random.PRNGKey(np.uint32(config.seed))

    c_ox, c_red, curr, sigs, params = load_dataset(data_path)

    if config.n_samples > 0:
        n_keep = min(config.n_samples, c_ox.shape[0])
        c_ox = c_ox[:n_keep]
        c_red = c_red[:n_keep]
        curr = curr[:n_keep]
        sigs = sigs[:n_keep]
        params = params[:n_keep]

    if c_ox.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 trajectories after filtering to train+validate, got {c_ox.shape[0]}"
        )

    max_species, nx, target_len, state_dim = infer_geometry(c_ox, c_red, curr, params)

    c_ox = jnp.asarray(c_ox)
    c_red = jnp.asarray(c_red)
    curr = jnp.asarray(curr)
    sigs = jnp.asarray(sigs)
    params = jnp.asarray(params)

    dataset_x = jnp.concatenate([c_ox, c_red, curr], axis=1)

    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, len(dataset_x))
    dataset_x = dataset_x[indices]
    sigs = sigs[indices]
    params = params[indices]

    total = int(len(dataset_x))
    val_size = int(total * config.val_split)
    val_size = min(max(1, val_size), total - 1)

    train_x = dataset_x[:-val_size]
    val_x = dataset_x[-val_size:]
    train_e = sigs[:-val_size]
    val_e = sigs[-val_size:]
    train_p = params[:-val_size]
    val_p = params[-val_size:]

    x_mean = jnp.mean(train_x, axis=0)
    x_std = jnp.std(train_x, axis=0) + 1e-5
    e_mean = jnp.mean(train_e, axis=0)
    e_std = jnp.std(train_e, axis=0) + 1e-5
    p_mean = jnp.mean(train_p, axis=0)
    p_std = jnp.std(train_p, axis=0) + 1e-5

    train_x = (train_x - x_mean) / x_std
    val_x = (val_x - x_mean) / x_std
    train_e = (train_e - e_mean) / e_std
    val_e = (val_e - e_mean) / e_std
    train_p = (train_p - p_mean) / p_std
    val_p = (val_p - p_mean) / p_std

    print(f"Train size: {len(train_x)}, Val size: {len(val_x)}")
    print(
        f"Model dimensions: state_dim={state_dim}, max_species={max_species}, "
        f"nx={nx}, target_len={target_len}, phys_dim={params.shape[1]}"
    )

    key, subkey = jax.random.split(key)
    model = VectorFieldNet(
        state_dim=state_dim,
        hidden_size=config.hidden_size,
        depth=config.depth,
        cond_dim=32,
        phys_dim=params.shape[1],
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

    optimizer = optax.adamw(learning_rate=config.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, x1, x0, E, p, step_key):
        loss, grads = compute_loss(model, x1, x0, E, p, step_key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    pbar = tqdm(range(start_epoch, config.epochs), desc="Training")
    for epoch in pbar:
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, len(train_x))

        shuffled_x1 = train_x[perms]
        shuffled_e = train_e[perms]
        shuffled_p = train_p[perms]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_x), config.batch_size):
            batch_x1 = shuffled_x1[i : i + config.batch_size]
            batch_e = shuffled_e[i : i + config.batch_size]
            batch_p = shuffled_p[i : i + config.batch_size]

            key, sample_key, step_key = jax.random.split(key, 3)
            batch_x0 = jax.random.normal(sample_key, batch_x1.shape)

            model, opt_state, loss = make_step(model, opt_state, batch_x1, batch_x0, batch_e, batch_p, step_key)
            epoch_loss += float(loss)
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("No training batches were formed. Check dataset size and batch_size.")

        avg_loss = epoch_loss / n_batches
        history["train"].append(avg_loss)

        if epoch % 1000 == 0 or epoch == config.epochs - 1:
            key, sample_key, step_key = jax.random.split(key, 3)
            val_x0 = jax.random.normal(sample_key, val_x.shape)
            val_loss = float(compute_val_loss(model, val_x, val_x0, val_e, val_p, step_key))
            history["val"].append((epoch, val_loss))
            pbar.set_postfix({"train": f"{avg_loss:.5f}", "val": f"{val_loss:.5f}"})

        if epoch > start_epoch and epoch % 10000 == 0:
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


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    base_args, _ = base_parser.parse_known_args()

    file_defaults = {}
    if base_args.config is not None:
        with open(base_args.config, "r", encoding="utf-8") as f:
            file_defaults = json.load(f)

    parser = argparse.ArgumentParser(
        description="Train Flow Matching surrogate",
        parents=[base_parser],
    )

    parser.set_defaults(**file_defaults)

    parser.add_argument(
        "--dataset",
        "--data-path",
        dest="data_path",
        type=str,
        default="/tmp/ecsfm/dataset_massive",
        help="Path to NPZ file or directory of NPZ chunks",
    )
    parser.add_argument("--artifact-dir", type=str, default="/tmp/ecsfm", help="Output directory for model/artifacts")
    parser.add_argument("--n-samples", type=int, default=0, help="Samples to use (0 = all)")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=128, help="MLP hidden size")
    parser.add_argument("--depth", type=int, default=3, help="MLP depth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--new-run", action="store_true", help="Ignore checkpoint and start from scratch")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = FlowConfig(
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        depth=args.depth,
        seed=args.seed,
        new_run=args.new_run,
        val_split=args.val_split,
    )

    train_surrogate(config=config, data_path=args.data_path, artifact_dir=args.artifact_dir)


if __name__ == "__main__":
    main()
