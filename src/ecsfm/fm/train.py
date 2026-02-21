import os
import multiprocessing
import json

cores = str(multiprocessing.cpu_count())
os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={cores}"
os.environ["OMP_NUM_THREADS"] = cores

import jax
from jax import config
config.update("jax_enable_x64", False)
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from pydantic import BaseModel, Field
import concurrent.futures
import multiprocessing

from ecsfm.sim.cv import simulate_cv
from ecsfm.fm.model import VectorFieldNet
from ecsfm.fm.objective import flow_matching_loss

class FlowConfig(BaseModel):
    n_samples: int = Field(100, description="Number of simulation trajectories")
    epochs: int = Field(500, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size")
    lr: float = Field(1e-3, description="Learning rate")
    hidden_size: int = Field(128, description="Hidden size for VectorFieldNet")
    depth: int = Field(3, description="Depth for VectorFieldNet")
    seed: int = Field(42, description="Random seed")
    new_run: bool = Field(False, description="Start training from scratch, ignoring checkpoints")
    val_split: float = Field(0.2, description="Fraction of dataset to use for validation")


def integrate_flow(model, x0, n_steps=100):
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = i * dt
        t_batch = jnp.full((x.shape[0], 1), t)
        v = jax.vmap(model)(t_batch, x)
        x = x + v * dt
    return x

def _run_single_sim(args):
    D_ox, k0, scan_rate, nx = args
    _, C_ox_hist, _, _, _, _ = simulate_cv(
        D_ox=D_ox,
        D_red=D_ox, 
        k0=k0,
        scan_rate=scan_rate,
        nx=nx,
        save_every=0
    )
    return C_ox_hist[-1]

def generate_training_data(n_samples: int, key: jax.random.PRNGKey) -> jax.Array:
    print(f"Generating dataset of {n_samples} physics simulations...")
    nx = 50
    final_states = []
    keys = jax.random.split(key, n_samples)
    
    # Pre-generate random args to feed into the parallel executor
    sim_args = []
    for i in range(n_samples):
        k1, k2, k3 = jax.random.split(keys[i], 3)
        D_ox = float(jnp.exp(jax.random.uniform(k1, minval=jnp.log(1e-6), maxval=jnp.log(1e-4))))
        k0 = float(jnp.exp(jax.random.uniform(k2, minval=jnp.log(1e-3), maxval=jnp.log(1e-1))))
        scan_rate = float(jax.random.uniform(k3, minval=0.01, maxval=1.0))
        sim_args.append((D_ox, k0, scan_rate, nx))
        
    # Parallel dispatch across all available CPU cores
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(_run_single_sim, sim_args), total=n_samples, desc="Simulating Physical Trajectories"):
            final_states.append(result)
            
    return jnp.stack(final_states)

@eqx.filter_value_and_grad
def compute_loss(model, x1, x0, key):
    return flow_matching_loss(model, x1, x0, key)

@eqx.filter_jit
def compute_val_loss(model, x1, x0, key):
    return flow_matching_loss(model, x1, x0, key)

def save_comparison(model, epoch, state_dim, key, data_mean=0.0, data_std=1.0):
    print(f"Generating surrogate comparison for epoch {epoch}...")
    n_samples = 50
    sample_key, _ = jax.random.split(key)
    x0 = jax.random.normal(sample_key, (n_samples, state_dim))
    x_generated = integrate_flow(model, x0, n_steps=100)
    x_generated = x_generated * data_std + data_mean
    
    gt_samples = []
    keys = jax.random.split(key, n_samples)
    for i in range(n_samples):
        k1, k2, k3 = jax.random.split(keys[i], 3)
        D_ox = jnp.exp(jax.random.uniform(k1, minval=jnp.log(1e-6), maxval=jnp.log(1e-4)))
        k0 = jnp.exp(jax.random.uniform(k2, minval=jnp.log(1e-3), maxval=jnp.log(1e-1)))
        scan_rate = jax.random.uniform(k3, minval=0.01, maxval=1.0)
        _, C_ox_hist, _, _, _, _ = simulate_cv(
            D_ox=float(D_ox),
            D_red=float(D_ox), 
            k0=float(k0),
            scan_rate=float(scan_rate),
            nx=state_dim,
            save_every=0
        )
        gt_samples.append(C_ox_hist[-1])
        
    gt_samples = np.array(gt_samples)
    x_generated = np.array(x_generated)
    
    gen_mean, gen_std = x_generated.mean(axis=0), x_generated.std(axis=0)
    gt_mean, gt_std = gt_samples.mean(axis=0), gt_samples.std(axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Surrogate Distribution Comparison - Epoch {epoch}", fontsize=16)
    
    axes[0].set_title(f"Generated Samples ({n_samples})\n(From base Gaussian noise)")
    axes[0].set_xlabel("Distance [cm] (Grid Index)")
    axes[0].set_ylabel("Concentration [mM]")
    for i in range(n_samples):
        axes[0].plot(x_generated[i], color='royalblue', alpha=0.15)
        
    axes[1].set_title(f"Ground Truth Samples ({n_samples})\n(From Physical Simulator)")
    axes[1].set_xlabel("Distance [cm] (Grid Index)")
    for i in range(n_samples):
        axes[1].plot(gt_samples[i], color='crimson', alpha=0.15)
        
    axes[2].set_title(f"Distribution Statistics Overlap\n(Mean \u00b1 1 Std Dev)")
    axes[2].set_xlabel("Distance [cm] (Grid Index)")
    x_axis = np.arange(state_dim)
    axes[2].plot(gen_mean, color='royalblue', label='Generated Mean')
    axes[2].fill_between(x_axis, gen_mean - gen_std, gen_mean + gen_std, color='royalblue', alpha=0.2)
    
    axes[2].plot(gt_mean, color='crimson', label='Ground Truth Mean', linestyle='--')
    axes[2].fill_between(x_axis, gt_mean - gt_std, gt_mean + gt_std, color='crimson', alpha=0.2)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"surrogate_comparison_ep{epoch}.png")
    plt.close()

def train_surrogate(config: FlowConfig):
    import numpy as np
    key = jax.random.PRNGKey(np.uint32(config.seed))
    key, subkey = jax.random.split(key)
    
    dataset_file = "dataset_cache.npy"
    if os.path.exists(dataset_file) and not config.new_run:
        print("Loading cached dataset...")
        dataset = jnp.load(dataset_file)
        if dataset.shape[0] < config.n_samples:
            print("Cached dataset too small, generating a new one...")
            dataset = generate_training_data(config.n_samples, subkey)
            jnp.save(dataset_file, dataset)
        else:
            dataset = dataset[:config.n_samples]
    else:
        dataset = generate_training_data(config.n_samples, subkey)
        jnp.save(dataset_file, dataset)
    
    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, len(dataset))
    dataset = dataset[indices]
    
    state_dim = dataset.shape[1]
    val_size = max(1, int(len(dataset) * config.val_split))
    train_dataset = dataset[:-val_size]
    val_dataset = dataset[-val_size:]
    
    data_mean = jnp.mean(train_dataset, axis=0)
    data_std = jnp.std(train_dataset, axis=0) + 1e-5
    
    train_dataset = (train_dataset - data_mean) / data_std
    val_dataset = (val_dataset - data_mean) / data_std
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    key, subkey = jax.random.split(key)
    model = VectorFieldNet(
        state_dim=state_dim,
        hidden_size=config.hidden_size,
        depth=config.depth,
        key=subkey
    )
    
    checkpoint_path = "surrogate_model.eqx"
    start_epoch = 0
    history = {'train': [], 'val': []}
    
    if os.path.exists(checkpoint_path) and not config.new_run:
        try:
            model = eqx.tree_deserialise_leaves(checkpoint_path, model)
            print("Loaded existing checkpoint: resuming from", checkpoint_path)
            if os.path.exists("training_history.json"):
                with open("training_history.json", "r") as f:
                    saved_history = json.load(f)
                    history = saved_history.get('history', {'train': [], 'val': []})
                    start_epoch = saved_history.get('epoch', 0)
                    print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            
    optimizer = optax.adamw(learning_rate=config.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state, x1, x0, step_key):
        loss, grads = compute_loss(model, x1, x0, step_key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    print("Training Flow Matching Surrogate...")
    
    pbar = tqdm(range(start_epoch, config.epochs), desc="Training Phase")
    for epoch in pbar:
        key, subkey = jax.random.split(key)
        
        perms = jax.random.permutation(subkey, len(train_dataset))
        shuffled_x1 = train_dataset[perms]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(train_dataset), config.batch_size):
            batch_x1 = shuffled_x1[i:i + config.batch_size]
            key, sample_key, step_key = jax.random.split(key, 3)
            batch_x0 = jax.random.normal(sample_key, batch_x1.shape)
            
            model, opt_state, loss = make_step(model, opt_state, batch_x1, batch_x0, step_key)
            epoch_loss += loss
            n_batches += 1
            
        avg_loss = epoch_loss / n_batches
        history['train'].append(float(avg_loss))
        
        if epoch % 1000 == 0 or epoch == config.epochs - 1:
            key, sample_key, step_key = jax.random.split(key, 3)
            val_x0 = jax.random.normal(sample_key, val_dataset.shape)
            val_loss = compute_val_loss(model, val_dataset, val_x0, step_key)
            history['val'].append((epoch, float(val_loss)))
            pbar.set_postfix({"Train Loss": f"{avg_loss:.5f}", "Val Loss": f"{val_loss:.5f}"})
            
        if epoch > start_epoch and epoch % 10000 == 0:
            eqx.tree_serialise_leaves(checkpoint_path, model)
            with open("training_history.json", "w") as f:
                json.dump({'history': history, 'epoch': epoch}, f)
            print(f"Checkpoint saved at epoch {epoch}")
            
            plt.figure()
            plt.plot(range(start_epoch, epoch + 1), history['train'][-(epoch - start_epoch + 1):], label='Train Loss')
            val_epochs, val_losses = zip(*history['val'])
            # Filter validation epochs to only plot ones from the current session
            valid_idx = [i for i, ve in enumerate(val_epochs) if ve >= start_epoch]
            plt.plot([val_epochs[i] for i in valid_idx], [val_losses[i] for i in valid_idx], label='Val Loss', marker='o')
            plt.title("Flow Matching Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("OT-CFM Loss")
            plt.yscale('log')
            plt.legend()
            plt.savefig("loss_curve.png")
            plt.close()
            
            save_comparison(model, epoch, state_dim, subkey, data_mean, data_std)
            
    avg_loss = history['train'][-1] if history.get('train') else 0.0
    print(f"Final Epoch | Loss: {avg_loss:.5f}")
    
    eqx.tree_serialise_leaves(checkpoint_path, model)
    with open("training_history.json", "w") as f:
        json.dump({'history': history, 'epoch': config.epochs}, f)
    print("Saved Flow Matching final model weights to surrogate_model.eqx")
    
    # Always plot and verify the surrogate at the very end of any run
    if history.get('train'):
        plt.figure()
        plt.plot(range(len(history['train'])), history['train'], label='Train Loss')
        if history.get('val'):
            val_epochs, val_losses = zip(*history['val'])
            plt.plot(val_epochs, val_losses, label='Val Loss', marker='o')
        plt.title("Flow Matching Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("OT-CFM Loss")
        plt.yscale('log')
        plt.legend()
        plt.savefig("loss_curve.png")
        plt.close()
        
    save_comparison(model, config.epochs, state_dim, subkey, data_mean, data_std)

def load_config_with_cli_overrides() -> FlowConfig:
    parser = argparse.ArgumentParser(description="Train Flow Matching Surrogate")
    parser.add_argument("--config", type=str, default="config.json", help="Path to JSON config file. Overridden by CLI args.")
    
    early_args, remaining_args = parser.parse_known_args()
    
    if os.path.exists(early_args.config):
        import json
        with open(early_args.config, "r") as f:
            file_data = json.load(f)
            config = FlowConfig(**file_data)
            print(f"Loaded base config from {early_args.config}")
    else:
        config = FlowConfig() 
        
    for key, field_info in FlowConfig.model_fields.items():
        flag_name = f"--{key.replace('_', '-')}"
        if field_info.annotation is bool:
            parser.add_argument(flag_name, action="store_true", default=argparse.SUPPRESS, help=field_info.description)
        else:
            parser.add_argument(flag_name, type=field_info.annotation, default=argparse.SUPPRESS, help=field_info.description)

    final_args = parser.parse_args(remaining_args)
    
    cli_overrides = vars(final_args)
    cli_overrides.pop('config', None) 
    
    config = config.model_copy(update=cli_overrides)
    return config

if __name__ == "__main__":
    final_config = load_config_with_cli_overrides()
    train_surrogate(final_config)
