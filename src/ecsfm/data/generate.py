import os
import argparse
import multiprocessing
import json

cores = str(multiprocessing.cpu_count())
os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={cores}"
os.environ["OMP_NUM_THREADS"] = cores

import concurrent.futures
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from ecsfm.sim.experiment import simulate_electrochem

def get_cv_waveform(E_start, E_vertex, scan_rate):
    t_max = 2 * abs(E_start - E_vertex) / scan_rate
    dt = 0.001
    t = np.arange(0, t_max, dt)
    half_time = t_max / 2
    E = np.where(t < half_time,
                    E_start - scan_rate * t,
                    E_vertex + scan_rate * (t - half_time))
    return E, t_max

def get_ca_waveform(E_rest, E_step, t_total, t_step):
    dt = 0.001
    t = np.arange(0, t_total, dt)
    E = np.where(t < t_step, E_rest, E_step)
    return E, t_total

def get_eis_waveform(E_dc, amplitude, frequency, t_total):
    dt = 0.001
    t = np.arange(0, t_total, dt)
    E = E_dc + amplitude * np.sin(2 * np.pi * frequency * t)
    return E, t_total

def get_swv_waveform(E_start, E_end, step_E, amplitude, freq):
    t_step = 1.0 / freq
    n_steps = int(abs(E_end - E_start) / step_E)
    t_total = n_steps * t_step
    
    dt = 0.001
    t = np.arange(0, t_total, dt)
    
    step_indices = (t / t_step).astype(int)
    E_base = E_start + np.sign(E_end - E_start) * step_indices * step_E
    
    half_step = t_step / 2
    is_forward = (t % t_step) < half_step
    direction = np.sign(E_end - E_start)
    if direction == 0: direction = 1
    
    E_pulse = np.where(is_forward, amplitude * direction, -amplitude * direction)
    E = E_base + E_pulse
    return E, t_total

def _run_single_sim(args):
    E_arr, t_max, params, nx = args
    D_ox, D_red, C_ox, C_red, E0, k0, alpha = params
    
    _, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis, I_hist_vis = simulate_electrochem(
        E_array=E_arr,
        t_max=t_max,
        D_ox=D_ox,
        D_red=D_red, 
        C_bulk_ox=C_ox,
        C_bulk_red=C_red,
        E0=E0,
        k0=k0,
        alpha=alpha,
        nx=nx,
        save_every=0
    )
    
    from ecsfm.sim.sensor import apply_sensor_model
    
    # 1. Reconstruct exact high-res time array native to simulator
    t_array = jnp.linspace(0, t_max, len(E_hist))
    
    # 2. Add realistic potentiostat RC low-pass filter (EIS response)
    # Using typical moderate uncompensated resistance and double layer capacitance
    Cdl = 1e-5  # 10 uF
    Ru = 100.0  # 100 Ohms
    
    # Apply sensor model to capture real voltage at interface and capacitive current
    E_real, I_total_mA, _ = apply_sensor_model(
        t=t_array,
        E_app=jnp.array(E_hist),
        I_f_mA=jnp.array(I_hist),
        Cdl=Cdl,
        Ru=Ru,
        noise_std_mA=0.0
    )
    
    # We thin down the measured total current back to the visualization rate for the ML dataset
    save_every = max(1, len(E_hist) // 200)
    I_total_vis = I_total_mA[::save_every]
    
    # The experimentalist only knows the commanded applied potential (E_hist), not E_real!
    # So the dataset condition remains E_hist.
    return C_ox_hist[-1], C_red_hist[-1], I_total_vis, E_hist_vis

def generate_multi_species_dataset(n_samples: int, key: jax.random.PRNGKey, max_species: int = 5):
    print(f"Generating Multi-Species dataset of {n_samples} diverse physics simulations...")
    nx = 50
    final_ox = []
    final_red = []
    i_hists = []
    signals = []
    phys_params = []
    keys = jax.random.split(key, n_samples)
    
    sim_args = []
    for i in range(n_samples):
        k1, k_wav = jax.random.split(keys[i])
        rng = np.random.default_rng(np.array(k1))
        
        # Determine how many active redox species are in this specific simulation
        num_active = rng.integers(1, max_species + 1)
        
        # Generate parameter arrays shaped (max_species,) padding with inactive species
        D_ox_arr = np.ones(max_species) * 1e-5
        D_red_arr = np.ones(max_species) * 1e-5
        C_ox_arr = np.zeros(max_species)
        C_red_arr = np.zeros(max_species)
        E0_arr = np.zeros(max_species)
        k0_arr = np.ones(max_species) * 0.01
        alpha_arr = np.ones(max_species) * 0.5
        
        for s in range(num_active):
            D_ox_arr[s] = float(np.exp(rng.uniform(np.log(1e-7), np.log(1e-4))))
            D_red_arr[s] = float(np.exp(rng.uniform(np.log(1e-7), np.log(1e-4))))
            C_ox_arr[s] = float(rng.uniform(0.1, 10.0))
            C_red_arr[s] = float(rng.uniform(0.0, 5.0))
            E0_arr[s] = float(rng.uniform(-0.8, 0.8))
            k0_arr[s] = float(np.exp(rng.uniform(np.log(1e-6), np.log(1e0))))
            alpha_arr[s] = float(rng.uniform(0.1, 0.9))
            
        params = (D_ox_arr, D_red_arr, C_ox_arr, C_red_arr, E0_arr, k0_arr, alpha_arr)
        
        # Flatten the parameters for network conditioning: shape (max_species * 7,)
        flat_params = np.concatenate([
            np.log(D_ox_arr), np.log(D_red_arr), C_ox_arr, C_red_arr, E0_arr, np.log(k0_arr), alpha_arr
        ])
        phys_params.append(flat_params)
        
        # Randomize Waveforms (Heavily diversified)
        wave_type = rng.choice(["cv", "ca", "eis", "swv"])
        if wave_type == "cv":
            scan_rate = rng.uniform(0.005, 5.0)
            E, t_max = get_cv_waveform(rng.uniform(0.1, 0.8), rng.uniform(-0.8, -0.1), scan_rate)
        elif wave_type == "ca":
            E, t_max = get_ca_waveform(0.0, rng.uniform(-1.0, 1.0), 3.0, 0.5)
        elif wave_type == "swv":
            E, t_max = get_swv_waveform(rng.uniform(0.1, 0.5), rng.uniform(-0.5, -0.1), 0.01, rng.uniform(0.02, 0.1), rng.uniform(5.0, 50.0))
        else:
            freq = rng.uniform(0.1, 1000.0)
            E, t_max = get_eis_waveform(rng.uniform(-0.5, 0.5), rng.uniform(0.01, 0.2), freq, 3.0)
            
        sim_args.append((E, t_max, params, nx))
        
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_sim, args): args for args in sim_args}
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_samples, desc="Simulating Multi-Species Trajectories"):
            c_ox, c_red, i_hist, signal = future.result()
            final_ox.append(c_ox.flatten())
            final_red.append(c_red.flatten())
            i_hists.append(i_hist)
            signals.append(signal)
            
    target_sig_len = 200
    resampled_signals = []
    resampled_currents = []
    for sig, cur in zip(signals, i_hists):
        orig_indices = np.linspace(0, 1, len(sig))
        target_indices = np.linspace(0, 1, target_sig_len)
        resampled_sig = np.interp(target_indices, orig_indices, sig)
        resampled_cur = np.interp(target_indices, orig_indices, cur)
        resampled_signals.append(resampled_sig)
        resampled_currents.append(resampled_cur)
        
    return (jnp.stack(final_ox), jnp.stack(final_red), 
            jnp.stack(resampled_currents), jnp.stack(resampled_signals), 
            jnp.array(phys_params))


def main():
    os.system("open /tmp/ecsfm")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5000, help="Samples per chunk")
    parser.add_argument("--n-chunks", type=int, default=1, help="Number of chunks to generate")
    parser.add_argument("--max-species", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="/tmp/ecsfm/dataset_massive")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for chunk in range(args.n_chunks):
        key = jax.random.PRNGKey(42 + chunk)
        print(f"\n--- Generating Chunk {chunk+1}/{args.n_chunks} ---")
        c_ox, c_red, curr, sigs, params = generate_multi_species_dataset(args.n_samples, key, args.max_species)
        
        chunk_path = os.path.join(args.output_dir, f"chunk_{chunk}.npz")
        np.savez(chunk_path, ox=c_ox, red=c_red, i=curr, e=sigs, p=params)
        print(f"Saved chunk {chunk} to {chunk_path}")
        
    print(f"\nSuccessfully generated {args.n_chunks * args.n_samples} total massive edge-case trajectories!")


if __name__ == "__main__":
    main()
