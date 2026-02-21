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

from ecsfm.sim.experiment import simulate_electrochem
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


def integrate_flow(model, x0, E, p, n_steps=100):
    dt = 1.0 / n_steps
    x = x0.astype(jnp.float32)
    E = E.astype(jnp.float32)
    p = p.astype(jnp.float32)
    for i in range(n_steps):
        t = i * dt
        t_batch = jnp.full((x.shape[0], 1), t)
        v = jax.vmap(model)(t_batch, x, E, p)
        x = x + v * dt
    return x

@eqx.filter_value_and_grad
def compute_loss(model, x1, x0, E, p, key):
    return flow_matching_loss(model, x1, x0, E, p, key)

@eqx.filter_jit
def compute_val_loss(model, x1, x0, E, p, key):
    return flow_matching_loss(model, x1, x0, E, p, key)

def save_comparison(model, epoch, state_dim, nx, key, e_mean, e_std, p_mean, p_std):
    print(f"Generating multi-task surrogate comparison for epoch {epoch}...")
    n_samples = 4
    sample_key, _ = jax.random.split(key)
    
    from ecsfm.data.generate import get_cv_waveform
    
    gt_ox, gt_red, gt_i, e_raw, p_raw = [], [], [], [], []
    keys = jax.random.split(key, n_samples)
    target_len = 200
    for i in range(n_samples):
        k1, k2 = jax.random.split(keys[i])
        E_start, E_vertex, scan_rate = 0.5, -0.5, 0.1
        E_t, t_max = get_cv_waveform(E_start, E_vertex, scan_rate)
        
        # Build test arrays for max_species = 5. Assume 2 active species for the visualization plot.
        D_ox = np.ones(5) * 1e-5
        D_red = np.ones(5) * 1e-5
        C_ox = np.zeros(5)
        C_red = np.zeros(5)
        E0 = np.zeros(5)
        k0 = np.ones(5) * 0.01
        alpha = np.ones(5) * 0.5
        
        # 2 active species with distinct peaks
        C_ox[0] = 1.0; E0[0] = 0.1
        C_ox[1] = 0.5; E0[1] = -0.2
        
        params = (D_ox, D_red, C_ox, C_red, E0, k0, alpha)
        
        flat_params = np.concatenate([
            np.log(D_ox), np.log(D_red), C_ox, C_red, E0, np.log(k0), alpha
        ])
        p_raw.append(flat_params)
        
        _, C_ox_hist, C_red_hist, _, _, E_hist_vis, I_hist_vis = simulate_electrochem(
            E_array=E_t, t_max=t_max, D_ox=D_ox, D_red=D_red, C_bulk_ox=C_ox, 
            C_bulk_red=C_red, E0=E0, k0=k0, alpha=alpha, nx=nx, save_every=0
        )
        gt_ox.append(C_ox_hist[-1].flatten())
        gt_red.append(C_red_hist[-1].flatten())
        
        orig_indices = np.linspace(0, 1, len(E_hist_vis))
        target_indices = np.linspace(0, 1, target_len)
        e_raw.append(np.interp(target_indices, orig_indices, E_hist_vis))
        gt_i.append(np.interp(target_indices, orig_indices, I_hist_vis))
            
    e_raw_arr = jnp.array(e_raw)
    p_raw_arr = jnp.array(p_raw)
    e_normalized = (e_raw_arr - e_mean) / e_std
    p_normalized = (p_raw_arr - p_mean) / p_std
    
    x0 = jax.random.normal(sample_key, (n_samples, state_dim))
    x_generated = integrate_flow(model, x0, e_normalized, p_normalized, n_steps=100)
    
    # We do not unnormalize purely for visual inspection comparison right now, 
    # but we will just compare normalized GT vs Normalized Gen shapes.
    # To do full unnormalization we'd need to pass ox_mean, ox_std, etc.
    # For now, let's just plot the generated distributions vs ground truth.
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
    fig.suptitle(f"Multi-Task Generation - Epoch {epoch}", fontsize=16)
    
    for i in range(n_samples):
        # Extract components from generated joint vector
        # x_generated is shape (batch, nx + nx + target_len)
        gen_ox = x_generated[i, :nx]
        gen_red = x_generated[i, nx:2*nx]
        gen_current = x_generated[i, 2*nx:]
        
        ax1 = axes[i, 0] if n_samples > 1 else axes[0]
        ax2 = axes[i, 1] if n_samples > 1 else axes[1]
        ax3 = axes[i, 2] if n_samples > 1 else axes[2]
        
        ax1.plot(gen_ox, label='Gen Ox', color='blue')
        ax1.plot(gen_red, label='Gen Red', color='red')
        ax1.set_title("Generated States")
        ax1.legend()
        
        ax2.plot(gt_ox[i], label='GT Ox', color='blue', linestyle='--')
        ax2.plot(gt_red[i], label='GT Red', color='red', linestyle='--')
        ax2.set_title("Ground Truth States")
        ax2.legend()
        
        ax3.plot(gen_current, label='Gen Current', color='green')
        ax3.plot(gt_i[i], label='GT Current', color='green', linestyle='--')
        ax3.set_title("Current Observable (I(t))")
        ax3.legend()
        
    plt.tight_layout()
    plt.savefig(f"/tmp/ecsfm/surrogate_comparison_ep{epoch}.png")
    plt.close()

def train_surrogate(config: FlowConfig, data_path: str):
    import numpy as np
    key = jax.random.PRNGKey(np.uint32(config.seed))
    key, subkey = jax.random.split(key)
    
    os.system("open /tmp/ecsfm")
    os.makedirs("/tmp/ecsfm", exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset block not found at {data_path}. Please run python -m ecsfm.data.generate first to amass a database block.")
        
    print(f"Loading massive pre-computed multi-species dataset from {data_path}...")
    data = np.load(data_path)
    c_ox, c_red, curr, sigs, params = data['ox'], data['red'], data['i'], data['e'], data['p']
    
    # Optional subset if n_samples explicitly overrides
    if config.n_samples > 0 and config.n_samples < c_ox.shape[0]:
        c_ox = c_ox[:config.n_samples]
        c_red = c_red[:config.n_samples]
        curr = curr[:config.n_samples]
        sigs = sigs[:config.n_samples]
        params = params[:config.n_samples]
        
    c_ox, c_red, curr, sigs, params = jnp.array(c_ox), jnp.array(c_red), jnp.array(curr), jnp.array(sigs), jnp.array(params)
    
    # Combine tasks into a master ground-truth vector: [Ox(x), Red(x), I(t)]
    # This teaches the model the joint distribution of physics and observables
    dataset_x = jnp.concatenate([c_ox, c_red, curr], axis=1)
    
    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, len(dataset_x))
    dataset_x = dataset_x[indices]
    sigs = sigs[indices]
    params = params[indices]
    
    state_dim = dataset_x.shape[1]
    val_size = max(1, int(len(dataset_x) * config.val_split))
    
    train_x = dataset_x[:-val_size]
    val_x = dataset_x[-val_size:]
    train_e = sigs[:-val_size]
    val_e = sigs[-val_size:]
    train_p = params[:-val_size]
    val_p = params[-val_size:]
    
    # Normalizing States + Observables (we'll just use global normalizers for the joint vector for this demo)
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
    
    nx = 50 # Our grid dimension
    cond_dim = 32 # Encoded size of the signal
    phys_dim = params.shape[1] # 7
    
    key, subkey = jax.random.split(key)
    model = VectorFieldNet(
        state_dim=state_dim,
        hidden_size=config.hidden_size,
        depth=config.depth,
        cond_dim=cond_dim,
        phys_dim=phys_dim,
        key=subkey
    )
    
    checkpoint_path = "/tmp/ecsfm/surrogate_model.eqx"
    start_epoch = 0
    history = {'train': [], 'val': []}
    
    if os.path.exists(checkpoint_path) and not config.new_run:
        try:
            model = eqx.tree_deserialise_leaves(checkpoint_path, model)
            print("Loaded existing checkpoint: resuming from", checkpoint_path)
            if os.path.exists("/tmp/ecsfm/training_history.json"):
                with open("/tmp/ecsfm/training_history.json", "r") as f:
                    saved_history = json.load(f)
                    history = saved_history.get('history', {'train': [], 'val': []})
                    start_epoch = saved_history.get('epoch', 0)
                    print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            
    optimizer = optax.adamw(learning_rate=config.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state, x1, x0, E, p, step_key):
        loss, grads = compute_loss(model, x1, x0, E, p, step_key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    print("Training Conditional Flow Matching Surrogate...")
    
    pbar = tqdm(range(start_epoch, config.epochs), desc="Training Phase")
    for epoch in pbar:
        key, subkey = jax.random.split(key)
        
        perms = jax.random.permutation(subkey, len(train_x))
        shuffled_x1 = train_x[perms]
        shuffled_e = train_e[perms]
        shuffled_p = train_p[perms]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(train_x), config.batch_size):
            batch_x1 = shuffled_x1[i:i + config.batch_size]
            batch_e = shuffled_e[i:i + config.batch_size]
            batch_p = shuffled_p[i:i + config.batch_size]
            key, sample_key, step_key = jax.random.split(key, 3)
            batch_x0 = jax.random.normal(sample_key, batch_x1.shape)
            
            model, opt_state, loss = make_step(model, opt_state, batch_x1, batch_x0, batch_e, batch_p, step_key)
            epoch_loss += loss
            n_batches += 1
            
        avg_loss = epoch_loss / n_batches
        history['train'].append(float(avg_loss))
        
        if epoch % 1000 == 0 or epoch == config.epochs - 1:
            key, sample_key, step_key = jax.random.split(key, 3)
            val_x0 = jax.random.normal(sample_key, val_x.shape)
            val_loss = compute_val_loss(model, val_x, val_x0, val_e, val_p, step_key)
            history['val'].append((epoch, float(val_loss)))
            pbar.set_postfix({"Train Loss": f"{avg_loss:.5f}", "Val Loss": f"{val_loss:.5f}"})
            
        if epoch > start_epoch and epoch % 10000 == 0:
            eqx.tree_serialise_leaves(checkpoint_path, model)
            with open("/tmp/ecsfm/training_history.json", "w") as f:
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
            plt.savefig("/tmp/ecsfm/loss_curve.png")
            plt.close()
            
            save_comparison(model, epoch, state_dim, nx, subkey, e_mean, e_std, p_mean, p_std)
            
    avg_loss = history['train'][-1] if history.get('train') else 0.0
    print(f"Final Epoch | Loss: {avg_loss:.5f}")
    
    eqx.tree_serialise_leaves(checkpoint_path, model)
    with open("/tmp/ecsfm/training_history.json", "w") as f:
        json.dump({'history': history, 'epoch': config.epochs}, f)
    print("Saved Flow Matching final model weights to /tmp/ecsfm/surrogate_model.eqx")
    
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
        plt.savefig("/tmp/ecsfm/loss_curve.png")
        plt.close()
        
    save_comparison(model, config.epochs, state_dim, nx, subkey, e_mean, e_std, p_mean, p_std)

def main():
    parser = argparse.ArgumentParser(description="Train Flow Matching Surrogate")
    parser.add_argument("--data-path", type=str, default="/tmp/ecsfm/dataset_multi_species.npz", help="Path to precomputed dataset NPZ chunk")
    parser.add_argument("--n-samples", type=int, default=0, help="Number of trajectories to use. 0 = use all in chunk")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for VectorFieldNet")
    parser.add_argument("--depth", type=int, default=3, help="Depth for VectorFieldNet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--new-run", action="store_true", help="Start training from scratch, ignoring checkpoints")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of dataset to use for validation")
    
    args = parser.parse_args()
    
    # Pack into simple object for compatibility
    class Config:
        pass
    
    config = Config()
    for k, v in vars(args).items():
        setattr(config, k, v)
        
    train_surrogate(config, args.data_path)

if __name__ == "__main__":
    main()
