import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from ecsfm.fm.model import VectorFieldNet
from ecsfm.sim.cv import simulate_cv

def integrate_flow(model, x0, n_steps=100):
    """
    Integrates the ODE dx/dt = v_theta(t, x) from t=0 to t=1 using Euler method.
    """
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = i * dt
        # model expects unbatched t (shape: (1,)) and x (shape: (dim,))
        # we have batched x, so we vmap the model. We pass a batched t.
        t_batch = jnp.full((x.shape[0], 1), t)
        v = jax.vmap(model)(t_batch, x)
        x = x + v * dt
    return x

def verify_surrogate():
    # 1. Initialize model architecture identical to training
    nx = 50
    state_dim = nx
    key = jax.random.PRNGKey(0)
    
    model = VectorFieldNet(
        state_dim=state_dim,
        hidden_size=128,
        depth=3,
        key=key
    )
    
    # 2. Load trained weights
    try:
        model = eqx.tree_deserialise_leaves("surrogate_model.eqx", model)
        print("Loaded surrogate_model.eqx successfully.")
    except Exception as e:
        print("Could not load surrogate_model.eqx. Have you run the training script yet?")
        return

    # 3. Generate samples from the surrogate
    print("Sampling from the Flow Matching surrogate...")
    n_samples = 5
    sample_key, _ = jax.random.split(key)
    # The OT-CFM starts from a standard normal distribution!
    x0 = jax.random.normal(sample_key, (n_samples, state_dim))
    
    x_generated = integrate_flow(model, x0, n_steps=100)
    
    # 4. Generate some ground truth samples for comparison
    print("Generating ground truth simulations for comparison...")
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
            nx=nx,
            save_every=0
        )
        gt_samples.append(C_ox_hist[-1])
        
    gt_samples = np.array(gt_samples)
    x_generated = np.array(x_generated)
    
    # 5. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].set_title("Surrogate Generated Profiles")
    axes[0].set_xlabel("Distance [cm] (Grid Index)")
    axes[0].set_ylabel("Concentration [mM]")
    for i in range(n_samples):
        axes[0].plot(x_generated[i], alpha=0.8)
        
    axes[1].set_title("Ground Truth Physics Simulator")
    axes[1].set_xlabel("Distance [cm] (Grid Index)")
    for i in range(n_samples):
        axes[1].plot(gt_samples[i], alpha=0.8)
        
    plt.tight_layout()
    plt.savefig("surrogate_comparison.png")
    print("Saved comparison plot to surrogate_comparison.png!")

if __name__ == "__main__":
    verify_surrogate()
