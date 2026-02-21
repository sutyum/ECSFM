import os
import multiprocessing

cores = str(multiprocessing.cpu_count())
os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={cores}"
os.environ["OMP_NUM_THREADS"] = cores

import jax
from jax import config
config.update("jax_enable_x64", False)
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ecsfm.sim.experiment import simulate_electrochem
from ecsfm.fm.model import VectorFieldNet
from ecsfm.data.generate import get_cv_waveform, get_swv_waveform, get_ca_waveform, get_eis_waveform

def get_normalizers(data_path: str, seed: int = 42, val_split: float = 0.2):
    """Replicates the normalizer extraction from train.py to ensure identical normalization."""
    import glob
    if os.path.isdir(data_path):
        chunk_files = glob.glob(os.path.join(data_path, "*.npz"))
        all_ox, all_red, all_i, all_e, all_p = [], [], [], [], []
        for cf in chunk_files:
            data = np.load(cf)
            all_ox.append(data['ox'])
            all_red.append(data['red'])
            all_i.append(data['i'])
            all_e.append(data['e'])
            all_p.append(data['p'])
        c_ox = np.concatenate(all_ox, axis=0)
        c_red = np.concatenate(all_red, axis=0)
        curr = np.concatenate(all_i, axis=0)
        sigs = np.concatenate(all_e, axis=0)
        params = np.concatenate(all_p, axis=0)
    else:
        data = np.load(data_path)
        c_ox, c_red, curr, sigs, params = data['ox'], data['red'], data['i'], data['e'], data['p']
    
    c_ox, c_red, curr, sigs, params = jnp.array(c_ox), jnp.array(c_red), jnp.array(curr), jnp.array(sigs), jnp.array(params)
    dataset_x = jnp.concatenate([c_ox, c_red, curr], axis=1)
    
    key = jax.random.PRNGKey(np.uint32(seed))
    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, len(dataset_x))
    
    dataset_x = dataset_x[indices]
    sigs = sigs[indices]
    params = params[indices]
    
    val_size = max(1, int(len(dataset_x) * val_split))
    train_x = dataset_x[:-val_size]
    train_e = sigs[:-val_size]
    train_p = params[:-val_size]
    
    x_mean = jnp.mean(train_x, axis=0)
    x_std = jnp.std(train_x, axis=0) + 1e-5
    e_mean = jnp.mean(train_e, axis=0)
    e_std = jnp.std(train_e, axis=0) + 1e-5
    p_mean = jnp.mean(train_p, axis=0)
    p_std = jnp.std(train_p, axis=0) + 1e-5
    
    return x_mean, x_std, e_mean, e_std, p_mean, p_std

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

def calculate_metrics(gt_trace, gen_trace, gtox, genox, gtred, genred):
    """Computes quantitative metrics between GT and prediction."""
    gt_trace_comp = np.array(gt_trace)
    gen_trace_comp = np.array(gen_trace)
    
    metrics = {
        "current_mse": float(mean_squared_error(gt_trace_comp, gen_trace_comp)),
        "current_mae": float(mean_absolute_error(gt_trace_comp, gen_trace_comp)),
        "current_r2": float(r2_score(gt_trace_comp, gen_trace_comp)),
        "ox_mse": float(mean_squared_error(np.array(gtox).flatten(), np.array(genox).flatten())),
        "red_mse": float(mean_squared_error(np.array(gtred).flatten(), np.array(genred).flatten()))
    }
    
    # Simple peak error heuristic: Difference in min/max values 
    metrics["peak_ox_err"] = float(abs(np.max(gt_trace_comp) - np.max(gen_trace_comp)))
    metrics["peak_red_err"] = float(abs(np.min(gt_trace_comp) - np.min(gen_trace_comp)))
    
    return metrics

def run_classical_eval(name, E_t, t_max, params, model, norm, key, nx=50, target_len=200):
    """Helper to run GT simulator + CFM Surrogate and plot."""
    D_ox, D_red, C_ox, C_red, E0, k0, alpha = params
    x_mean, x_std, e_mean, e_std, p_mean, p_std = norm
    
    # Ground Truth simulation
    _, C_ox_hist, C_red_hist, _, _, E_hist_vis, I_hist_vis = simulate_electrochem(
        E_array=E_t, t_max=t_max, D_ox=D_ox, D_red=D_red, C_bulk_ox=C_ox, 
        C_bulk_red=C_red, E0=E0, k0=k0, alpha=alpha, nx=nx, save_every=0
    )
    gt_ox = C_ox_hist[-1].flatten()
    gt_red = C_red_hist[-1].flatten()
    
    # Resample GT signals for Surrogate compatibility
    orig_indices = np.linspace(0, 1, len(E_hist_vis))
    target_indices = np.linspace(0, 1, target_len)
    e_signal = np.interp(target_indices, orig_indices, E_hist_vis)
    gt_i = np.interp(target_indices, orig_indices, I_hist_vis)
    
    # Flat parameters for Surrogate condition
    max_species = 5
    D_ox_pad = np.ones(max_species) * 1e-5; D_ox_pad[:len(np.atleast_1d(D_ox))] = np.atleast_1d(D_ox)
    D_red_pad = np.ones(max_species) * 1e-5; D_red_pad[:len(np.atleast_1d(D_red))] = np.atleast_1d(D_red)
    C_ox_pad = np.zeros(max_species); C_ox_pad[:len(np.atleast_1d(C_ox))] = np.atleast_1d(C_ox)
    C_red_pad = np.zeros(max_species); C_red_pad[:len(np.atleast_1d(C_red))] = np.atleast_1d(C_red)
    E0_pad = np.zeros(max_species); E0_pad[:len(np.atleast_1d(E0))] = np.atleast_1d(E0)
    k0_pad = np.ones(max_species) * 0.01; k0_pad[:len(np.atleast_1d(k0))] = np.atleast_1d(k0)
    alpha_pad = np.ones(max_species) * 0.5; alpha_pad[:len(np.atleast_1d(alpha))] = np.atleast_1d(alpha)

    flat_params = np.concatenate([
        np.log(D_ox_pad), np.log(D_red_pad), C_ox_pad, C_red_pad, E0_pad, np.log(k0_pad), alpha_pad
    ])
    
    e_normalized = (jnp.array([e_signal]) - e_mean) / e_std
    p_normalized = (jnp.array([flat_params]) - p_mean) / p_std
    
    state_dim = max_species * nx * 2 + target_len
    x0 = jax.random.normal(key, (1, state_dim))
    
    x_generated = integrate_flow(model, x0, e_normalized, p_normalized, n_steps=100)
    
    # Unnormalize
    gen_x = (x_generated[0] * x_std) + x_mean
    
    max_species = 5
    gen_ox = gen_x[:nx*max_species].reshape(max_species, nx)
    gen_red = gen_x[nx*max_species:2*nx*max_species].reshape(max_species, nx)
    gen_current = gen_x[2*nx*max_species:]
    
    gt_ox_rs = gt_ox.reshape(max_species, nx)
    gt_red_rs = gt_red.reshape(max_species, nx)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Surrogate Evaluation: {name}", fontsize=16)
    
    ax1 = axes[0]
    ax1.plot(gen_ox[0], label='Pred Ox', color='blue')
    ax1.plot(gen_red[0], label='Pred Red', color='red')
    ax1.plot(gt_ox_rs[0], label='GT Ox', color='darkblue', linestyle='--')
    ax1.plot(gt_red_rs[0], label='GT Red', color='darkred', linestyle='--')
    
    if C_ox[1] > 0.0:
        ax1.plot(gen_ox[1], label='Pred Ox (sp2)', color='cyan')
        ax1.plot(gen_red[1], label='Pred Red (sp2)', color='magenta')
        ax1.plot(gt_ox_rs[1], label='GT Ox (sp2)', color='darkcyan', linestyle='--')
        ax1.plot(gt_red_rs[1], label='GT Red (sp2)', color='darkmagenta', linestyle='--')
    
    ax1.set_title("Concentration Profiles")
    ax1.legend()
    
    ax2 = axes[1]
    ax2.plot(e_signal, gen_current, label='Pred CV', color='green')
    ax2.plot(e_signal, gt_i, label='GT CV', color='green', linestyle='--')
    ax2.set_xlabel("Potential (V)")
    ax2.set_ylabel("Current")
    ax2.set_title("I-V Curve")
    ax2.legend()
    
    ax3 = axes[2]
    times = np.linspace(0, t_max, target_len)
    ax3.plot(times, gen_current, label='Pred I(t)', color='green')
    ax3.plot(times, gt_i, label='GT I(t)', color='green', linestyle='--')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Current")
    ax3.set_title("Chronoamperogram I-t Curve")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"/tmp/ecsfm/eval_{name.replace(' ', '_').lower()}.png")
    plt.close()
    
    metrics = calculate_metrics(gt_i, gen_current, gt_ox_rs, gen_ox, gt_red_rs, gen_red)
    print(f"[{name}] Completed. R2: {metrics['current_r2']:.4f} | MSE: {metrics['current_mse']:.2e}")
    
    return metrics

def main():
    checkpoint_path = "/tmp/ecsfm/surrogate_model.eqx"
    data_path = "/tmp/ecsfm/dataset_massive"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
    print("Loading explicit formal Normalizers...")
    norm = get_normalizers(data_path)
    state_dim = 700 # Explicitly set to 700 to match the 799-footprint checkout
    cond_dim = 32
    phys_dim = 35 # Explicitly override this layout explicitly using 1-species configuration param unpacking (5 species * 7 params)
    
    key = jax.random.PRNGKey(0)
    _, subkey = jax.random.split(key)
    
    model = VectorFieldNet(
        state_dim=state_dim,
        hidden_size=128,
        depth=3,
        cond_dim=cond_dim,
        phys_dim=phys_dim,
        key=subkey
    )
    
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
    print("Model Loaded. Running Classical Scenarios for Scorecard...")
    
    results_scorecard = {}
    
    # Define baseline physical arrays for up to 5 species
    def get_base_params():
        D_ox = np.ones(5) * 1e-5
        D_red = np.ones(5) * 1e-5
        C_ox = np.zeros(5)
        C_red = np.zeros(5)
        E0 = np.zeros(5)
        k0 = np.ones(5) * 0.01
        alpha = np.ones(5) * 0.5
        return D_ox, D_red, C_ox, C_red, E0, k0, alpha
    
    # Scenario 1: Ferrocene (Reversible 1-e transfer)
    D_ox, D_red, C_ox, C_red, E0, k0, alpha = get_base_params()
    C_ox[0] = 1.0 # 1 mM Bulk Ox
    E0[0] = 0.0
    k0[0] = 0.1 # Fast kinetics
    params_ferro = (D_ox, D_red, C_ox, C_red, E0, k0, alpha)
    E_cv, t_cv = get_cv_waveform(0.5, -0.5, 0.1)
    
    key, subkey = jax.random.split(key)
    results_scorecard["Ferrocene_Reversible_CV"] = run_classical_eval("Ferrocene Reversible CV", E_cv, t_cv, params_ferro, model, norm, subkey)
    
    # Scenario 2: Multi-specie Peak Separation
    D_ox, D_red, C_ox, C_red, E0, k0, alpha = get_base_params()
    C_ox[0] = 1.0; E0[0] = 0.3; k0[0] = 0.01
    C_ox[1] = 0.8; E0[1] = -0.2; k0[1] = 0.01
    params_multi = (D_ox, D_red, C_ox, C_red, E0, k0, alpha)
    E_cv2, t_cv2 = get_cv_waveform(0.6, -0.6, 0.1)
    
    key, subkey = jax.random.split(key)
    results_scorecard["Multi_Species_CV"] = run_classical_eval("Multi_Species CV", E_cv2, t_cv2, params_multi, model, norm, subkey)
    
    # Scenario 3: SWV vs CV (Using Ferrocene params again to compare shapes)
    E_swv, t_swv = get_swv_waveform(0.5, -0.5, 0.01, 0.05, 15.0)
    key, subkey = jax.random.split(key)
    # We label it SWV, the physics are base Ferrocene
    results_scorecard["SWV_Ferrocene"] = run_classical_eval("SWV_Ferrocene", E_swv, t_swv, params_ferro, model, norm, subkey)
    
    # Scenario 4: CA vs EIS
    D_ox, D_red, C_ox, C_red, E0, k0, alpha = get_base_params()
    C_ox[0] = 1.0; E0[0] = 0.1; k0[0] = 0.005 # Slower kinetics
    params_slow = (D_ox, D_red, C_ox, C_red, E0, k0, alpha)
    
    E_ca, t_ca = get_ca_waveform(0.5, -0.2, 2.0, 0.5)
    key, subkey = jax.random.split(key)
    results_scorecard["CA_Step"] = run_classical_eval("CA_Step", E_ca, t_ca, params_slow, model, norm, subkey)
    
    E_eis, t_eis = get_eis_waveform(0.0, 0.05, 10.0, 2.0)
    key, subkey = jax.random.split(key)
    results_scorecard["EIS_Sine"] = run_classical_eval("EIS_Sine", E_eis, t_eis, params_slow, model, norm, subkey)
    
    # Generate a rigorous physics score out of 100 based on overall R-squared
    # R2 = 1.0 means perfect prediction (100)
    # R2 <= 0.0 means worse-than-mean baseline prediction (0)
    avg_r2 = np.mean([res["current_r2"] for res in results_scorecard.values()])
    final_score = max(0.0, min(100.0, avg_r2 * 100.0))
    results_scorecard["Final_Score_Out_Of_100"] = round(float(final_score), 2)
    
    print("\n" + "="*50)
    print(f"FINAL EVALUATION SCORECARD (Global Score: {results_scorecard['Final_Score_Out_Of_100']}/100)")
    print("="*50)
    print(json.dumps(results_scorecard, indent=4))
    
    with open('/tmp/ecsfm/evaluation_scorecard.json', 'w') as f:
        json.dump(results_scorecard, f, indent=4)
        
    print("\nScorecard saved to /tmp/ecsfm/evaluation_scorecard.json")
    print("All classical evaluations computed and plotted to /tmp/ecsfm/")

if __name__ == "__main__":
    main()
