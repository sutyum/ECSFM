import argparse
import concurrent.futures
import multiprocessing
import os

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ecsfm.sim.experiment import simulate_electrochem
from ecsfm.sim.sensor import apply_sensor_model


def get_cv_waveform(E_start: float, E_vertex: float, scan_rate: float) -> tuple[np.ndarray, float]:
    if scan_rate <= 0:
        raise ValueError(f"scan_rate must be positive, got {scan_rate}")
    if E_start == E_vertex:
        raise ValueError("E_start and E_vertex must differ for a CV waveform")

    t_max = 2.0 * abs(E_start - E_vertex) / scan_rate
    dt = 1e-3
    t = np.arange(0.0, t_max, dt)
    half_time = t_max / 2.0
    E = np.where(
        t < half_time,
        E_start - scan_rate * t,
        E_vertex + scan_rate * (t - half_time),
    )
    return E, t_max


def get_ca_waveform(E_rest: float, E_step: float, t_total: float, t_step: float) -> tuple[np.ndarray, float]:
    if t_total <= 0:
        raise ValueError(f"t_total must be positive, got {t_total}")
    if t_step < 0 or t_step > t_total:
        raise ValueError(f"t_step must be in [0, t_total], got {t_step}")

    dt = 1e-3
    t = np.arange(0.0, t_total, dt)
    E = np.where(t < t_step, E_rest, E_step)
    return E, t_total


def get_eis_waveform(
    E_dc: float,
    amplitude: float,
    frequency: float,
    t_total: float,
    min_samples_per_cycle: int = 20,
) -> tuple[np.ndarray, float]:
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    if t_total <= 0:
        raise ValueError(f"t_total must be positive, got {t_total}")
    if min_samples_per_cycle < 5:
        raise ValueError("min_samples_per_cycle must be >= 5")

    dt_nyquist_safe = 1.0 / (min_samples_per_cycle * frequency)
    dt = min(1e-3, dt_nyquist_safe)
    dt = max(dt, 2e-5)

    t = np.arange(0.0, t_total, dt)
    E = E_dc + amplitude * np.sin(2.0 * np.pi * frequency * t)
    return E, t_total


def get_swv_waveform(
    E_start: float,
    E_end: float,
    step_E: float,
    amplitude: float,
    freq: float,
) -> tuple[np.ndarray, float]:
    if step_E <= 0:
        raise ValueError(f"step_E must be positive, got {step_E}")
    if freq <= 0:
        raise ValueError(f"freq must be positive, got {freq}")
    if E_start == E_end:
        raise ValueError("E_start and E_end must differ for SWV waveform")

    t_step = 1.0 / freq
    n_steps = int(abs(E_end - E_start) / step_E)
    if n_steps < 1:
        raise ValueError("SWV configuration results in zero staircase steps")

    t_total = n_steps * t_step
    dt = 1e-3
    t = np.arange(0.0, t_total, dt)

    step_indices = (t / t_step).astype(int)
    direction = np.sign(E_end - E_start)
    if direction == 0:
        direction = 1

    E_base = E_start + direction * step_indices * step_E

    half_step = t_step / 2.0
    is_forward = (t % t_step) < half_step
    E_pulse = np.where(is_forward, amplitude * direction, -amplitude * direction)
    E = E_base + E_pulse
    return E, t_total


def _run_single_sim(args: tuple[np.ndarray, float, tuple[np.ndarray, ...], int]) -> tuple[np.ndarray, ...]:
    E_arr, t_max, params, nx = args
    D_ox, D_red, C_ox, C_red, E0, k0, alpha = params

    _, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis, _ = simulate_electrochem(
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
        save_every=0,
    )

    t_array = jnp.linspace(0.0, t_max, len(E_hist))

    _, I_total_mA, _ = apply_sensor_model(
        t=t_array,
        E_app=jnp.asarray(E_hist),
        I_f_mA=jnp.asarray(I_hist),
        Cdl=1e-5,
        Ru=100.0,
        noise_std_mA=0.0,
    )

    save_every = max(1, len(E_hist) // 200)
    I_total_vis = np.asarray(I_total_mA[::save_every])

    return (
        np.asarray(C_ox_hist[-1]),
        np.asarray(C_red_hist[-1]),
        I_total_vis,
        np.asarray(E_hist_vis),
    )


def generate_multi_species_dataset(
    n_samples: int,
    key: jax.Array,
    max_species: int = 5,
    nx: int = 50,
    max_workers: int | None = None,
) -> tuple[jnp.ndarray, ...]:
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if max_species <= 0:
        raise ValueError(f"max_species must be positive, got {max_species}")
    if nx < 2:
        raise ValueError(f"nx must be >= 2, got {nx}")

    print(f"Generating multi-species dataset of {n_samples} simulations...")

    final_ox: list[np.ndarray] = []
    final_red: list[np.ndarray] = []
    i_hists: list[np.ndarray] = []
    signals: list[np.ndarray] = []
    phys_params: list[np.ndarray] = []

    keys = jax.random.split(key, n_samples)
    sim_args: list[tuple[np.ndarray, float, tuple[np.ndarray, ...], int]] = []

    for i in range(n_samples):
        seed = int(jax.random.randint(keys[i], shape=(), minval=0, maxval=2**31 - 1))
        rng = np.random.default_rng(seed)

        num_active = int(rng.integers(1, max_species + 1))

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

        flat_params = np.concatenate(
            [
                np.log(D_ox_arr),
                np.log(D_red_arr),
                C_ox_arr,
                C_red_arr,
                E0_arr,
                np.log(k0_arr),
                alpha_arr,
            ]
        )
        phys_params.append(flat_params)

        wave_type = rng.choice(["cv", "ca", "eis", "swv"])
        if wave_type == "cv":
            scan_rate = float(rng.uniform(0.005, 5.0))
            E, t_max = get_cv_waveform(float(rng.uniform(0.1, 0.8)), float(rng.uniform(-0.8, -0.1)), scan_rate)
        elif wave_type == "ca":
            E, t_max = get_ca_waveform(0.0, float(rng.uniform(-1.0, 1.0)), 3.0, 0.5)
        elif wave_type == "swv":
            E, t_max = get_swv_waveform(
                float(rng.uniform(0.1, 0.5)),
                float(rng.uniform(-0.5, -0.1)),
                0.01,
                float(rng.uniform(0.02, 0.1)),
                float(rng.uniform(5.0, 50.0)),
            )
        else:
            freq = float(rng.uniform(0.1, 1000.0))
            E, t_max = get_eis_waveform(
                float(rng.uniform(-0.5, 0.5)),
                float(rng.uniform(0.01, 0.2)),
                freq,
                3.0,
            )

        sim_args.append((E, t_max, params, nx))

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_sim, args): args for args in sim_args}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=n_samples,
            desc="Simulating trajectories",
        ):
            c_ox, c_red, i_hist, signal = future.result()
            final_ox.append(c_ox.flatten())
            final_red.append(c_red.flatten())
            i_hists.append(i_hist)
            signals.append(signal)

    target_sig_len = 200
    resampled_signals = []
    resampled_currents = []
    for sig, cur in zip(signals, i_hists):
        orig_indices = np.linspace(0.0, 1.0, len(sig))
        target_indices = np.linspace(0.0, 1.0, target_sig_len)
        resampled_signals.append(np.interp(target_indices, orig_indices, sig))
        resampled_currents.append(np.interp(target_indices, orig_indices, cur))

    return (
        jnp.asarray(np.stack(final_ox)),
        jnp.asarray(np.stack(final_red)),
        jnp.asarray(np.stack(resampled_currents)),
        jnp.asarray(np.stack(resampled_signals)),
        jnp.asarray(np.stack(phys_params)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-species electrochemical dataset chunks")
    parser.add_argument("--n-samples", type=int, default=5000, help="Samples per chunk")
    parser.add_argument("--n-chunks", type=int, default=1, help="Number of chunks to generate")
    parser.add_argument("--max-species", type=int, default=5, help="Maximum species represented in each sample")
    parser.add_argument("--nx", type=int, default=50, help="Spatial grid size")
    parser.add_argument("--workers", type=int, default=None, help="Process workers (default: cpu_count-1)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output-dir", type=str, default="/tmp/ecsfm/dataset_massive", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {args.n_chunks}")

    os.makedirs(args.output_dir, exist_ok=True)

    for chunk in range(args.n_chunks):
        key = jax.random.PRNGKey(args.seed + chunk)
        print(f"\n--- Generating chunk {chunk + 1}/{args.n_chunks} ---")
        c_ox, c_red, curr, sigs, params = generate_multi_species_dataset(
            n_samples=args.n_samples,
            key=key,
            max_species=args.max_species,
            nx=args.nx,
            max_workers=args.workers,
        )

        chunk_path = os.path.join(args.output_dir, f"chunk_{chunk}.npz")
        np.savez(chunk_path, ox=np.asarray(c_ox), red=np.asarray(c_red), i=np.asarray(curr), e=np.asarray(sigs), p=np.asarray(params))
        print(f"Saved chunk {chunk} to {chunk_path}")

    print(f"\nSuccessfully generated {args.n_chunks * args.n_samples} total trajectories")


if __name__ == "__main__":
    main()
