import argparse
import concurrent.futures
import multiprocessing
import os
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ecsfm.sim.experiment import simulate_electrochem
from ecsfm.sim.sensor import apply_sensor_model

TASK_NAMES = [
    "cv_reversible",
    "ca_step",
    "cv_multispecies",
    "swv_pulse",
    "eis_low_freq",
    "eis_high_freq",
    "kinetics_limited",
    "diffusion_limited",
]
TASK_TO_ID = {name: idx for idx, name in enumerate(TASK_NAMES)}

STAGE_NAMES = ["foundation", "bridge", "frontier"]
AUGMENTATION_NAMES = ["none", "permute_species", "scale_concentration"]

BASE_STAGE_TASKS = {
    0: ["cv_reversible", "ca_step"],
    1: ["cv_multispecies", "swv_pulse", "kinetics_limited"],
    2: ["eis_low_freq", "eis_high_freq", "diffusion_limited"],
}


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


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _pack_flat_params(
    D_ox_arr: np.ndarray,
    D_red_arr: np.ndarray,
    C_ox_arr: np.ndarray,
    C_red_arr: np.ndarray,
    E0_arr: np.ndarray,
    k0_arr: np.ndarray,
    alpha_arr: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
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


def _permute_flat_params(flat_params: np.ndarray, perm: np.ndarray, max_species: int) -> np.ndarray:
    out = flat_params.copy()
    for segment in range(7):
        start = segment * max_species
        end = start + max_species
        out[start:end] = flat_params[start:end][perm]
    return out


def _sample_task_for_stage(rng: np.random.Generator, stage_id: int, recipe: str) -> str:
    if recipe == "baseline_random":
        return str(rng.choice(TASK_NAMES))
    if recipe == "stress_mixture":
        hard_pool = ["eis_high_freq", "diffusion_limited", "kinetics_limited", "cv_multispecies", "swv_pulse"]
        return str(rng.choice(hard_pool))

    pool: list[str] = []
    for sid in range(stage_id + 1):
        pool.extend(BASE_STAGE_TASKS[sid])
    return str(rng.choice(pool))


def _sample_stage_ids(
    n_samples: int,
    recipe: str,
    stage_proportions: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    if recipe in ("baseline_random", "stress_mixture"):
        return np.full((n_samples,), 2, dtype=np.int32)

    proportions = np.asarray(stage_proportions, dtype=float)
    proportions = np.clip(proportions, 0.0, None)
    if proportions.sum() == 0.0:
        raise ValueError("stage_proportions must contain at least one positive value")
    proportions = proportions / proportions.sum()

    counts = np.floor(proportions * n_samples).astype(int)
    while counts.sum() < n_samples:
        counts[np.argmin(counts)] += 1

    stage_ids = np.concatenate([np.full((counts[s],), s, dtype=np.int32) for s in range(3)])
    rng.shuffle(stage_ids)
    return stage_ids


def _stage_ranges(stage_id: int) -> dict[str, tuple[float, float]]:
    if stage_id == 0:
        return {
            "D": (3e-6, 5e-5),
            "k0": (3e-3, 3e-1),
            "C_ox": (0.5, 3.0),
            "C_red": (0.0, 0.8),
            "E0": (-0.4, 0.4),
            "scan_rate": (0.02, 0.3),
        }
    if stage_id == 1:
        return {
            "D": (1e-6, 1e-4),
            "k0": (3e-4, 5e-1),
            "C_ox": (0.2, 6.0),
            "C_red": (0.0, 2.0),
            "E0": (-0.6, 0.6),
            "scan_rate": (0.01, 1.0),
        }
    return {
        "D": (3e-7, 1e-4),
        "k0": (1e-6, 1.0),
        "C_ox": (0.05, 10.0),
        "C_red": (0.0, 5.0),
        "E0": (-0.8, 0.8),
        "scan_rate": (0.005, 5.0),
    }


def _sample_params_for_task(
    rng: np.random.Generator,
    max_species: int,
    task_name: str,
    stage_id: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    ranges = _stage_ranges(stage_id)

    if task_name in ("cv_reversible", "ca_step", "kinetics_limited", "diffusion_limited"):
        num_active = 1
    elif task_name == "cv_multispecies":
        num_active = int(rng.integers(2, max_species + 1))
    else:
        num_active = int(rng.integers(1, min(3, max_species) + 1))

    D_ox_arr = np.ones(max_species) * 1e-5
    D_red_arr = np.ones(max_species) * 1e-5
    C_ox_arr = np.zeros(max_species)
    C_red_arr = np.zeros(max_species)
    E0_arr = np.zeros(max_species)
    k0_arr = np.ones(max_species) * 0.01
    alpha_arr = np.ones(max_species) * 0.5

    for s in range(num_active):
        D_ox_arr[s] = _log_uniform(rng, *ranges["D"])
        D_red_arr[s] = _log_uniform(rng, *ranges["D"])
        C_ox_arr[s] = float(rng.uniform(*ranges["C_ox"]))
        C_red_arr[s] = float(min(C_ox_arr[s], rng.uniform(*ranges["C_red"])))
        E0_arr[s] = float(rng.uniform(*ranges["E0"]))
        k0_arr[s] = _log_uniform(rng, *ranges["k0"])
        alpha_arr[s] = float(rng.uniform(0.2, 0.8))

    if task_name == "cv_reversible":
        k0_arr[:num_active] = [
            _log_uniform(rng, max(ranges["k0"][0], 1e-2), max(ranges["k0"][1], 2e-1))
            for _ in range(num_active)
        ]
        alpha_arr[:num_active] = rng.uniform(0.45, 0.55, size=num_active)

    if task_name == "kinetics_limited":
        k0_arr[:num_active] = [_log_uniform(rng, 1e-6, 5e-4) for _ in range(num_active)]

    if task_name == "diffusion_limited":
        D_ox_arr[:num_active] = [_log_uniform(rng, 1e-7, 2e-6) for _ in range(num_active)]
        D_red_arr[:num_active] = [_log_uniform(rng, 1e-7, 2e-6) for _ in range(num_active)]
        k0_arr[:num_active] = [_log_uniform(rng, 5e-2, 1.0) for _ in range(num_active)]

    if task_name == "cv_multispecies" and num_active > 1:
        e0_sorted = np.sort(rng.uniform(ranges["E0"][0], ranges["E0"][1], size=num_active))
        min_sep = 0.08
        for i in range(1, num_active):
            if e0_sorted[i] - e0_sorted[i - 1] < min_sep:
                e0_sorted[i] = e0_sorted[i - 1] + min_sep
        E0_arr[:num_active] = np.clip(e0_sorted, -0.8, 0.8)

    params = (D_ox_arr, D_red_arr, C_ox_arr, C_red_arr, E0_arr, k0_arr, alpha_arr)
    flat_params = _pack_flat_params(*params)
    return params, flat_params


def _sample_waveform_for_task(
    rng: np.random.Generator,
    task_name: str,
    stage_id: int,
) -> tuple[np.ndarray, float]:
    ranges = _stage_ranges(stage_id)

    if task_name in ("cv_reversible", "cv_multispecies", "kinetics_limited", "diffusion_limited"):
        scan_rate = float(rng.uniform(*ranges["scan_rate"]))
        E_start = float(rng.uniform(0.2, 0.8))
        E_vertex = float(rng.uniform(-0.8, -0.1))
        return get_cv_waveform(E_start, E_vertex, scan_rate)

    if task_name == "ca_step":
        E_rest = float(rng.uniform(-0.2, 0.2))
        E_step = float(rng.uniform(-1.0, 1.0))
        t_total = float(rng.uniform(2.0, 4.0))
        t_step = float(rng.uniform(0.2, 1.0))
        return get_ca_waveform(E_rest, E_step, t_total, t_step)

    if task_name == "swv_pulse":
        return get_swv_waveform(
            E_start=float(rng.uniform(0.2, 0.6)),
            E_end=float(rng.uniform(-0.6, -0.1)),
            step_E=float(rng.uniform(0.005, 0.02)),
            amplitude=float(rng.uniform(0.02, 0.12)),
            freq=float(rng.uniform(5.0, 60.0)),
        )

    if task_name == "eis_low_freq":
        return get_eis_waveform(
            E_dc=float(rng.uniform(-0.4, 0.4)),
            amplitude=float(rng.uniform(0.01, 0.08)),
            frequency=float(rng.uniform(0.1, 20.0)),
            t_total=float(rng.uniform(2.0, 4.0)),
            min_samples_per_cycle=30,
        )

    # eis_high_freq
    return get_eis_waveform(
        E_dc=float(rng.uniform(-0.3, 0.3)),
        amplitude=float(rng.uniform(0.005, 0.05)),
        frequency=float(rng.uniform(20.0, 1200.0)),
        t_total=float(rng.uniform(1.0, 2.5)),
        min_samples_per_cycle=25,
    )


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


def _resample_trace(trace: np.ndarray, target_len: int) -> np.ndarray:
    if trace.shape[0] == target_len:
        return trace
    source = np.linspace(0.0, 1.0, trace.shape[0])
    target = np.linspace(0.0, 1.0, target_len)
    return np.interp(target, source, trace)


def _augment_permute_species(
    final_ox_flat: np.ndarray,
    final_red_flat: np.ndarray,
    flat_params: np.ndarray,
    max_species: int,
    nx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = rng.permutation(max_species)
    aug_ox = final_ox_flat.reshape(max_species, nx)[perm].reshape(-1)
    aug_red = final_red_flat.reshape(max_species, nx)[perm].reshape(-1)
    aug_params = _permute_flat_params(flat_params, perm, max_species)
    return aug_ox, aug_red, aug_params


def _augment_scale_concentration(
    final_ox_flat: np.ndarray,
    final_red_flat: np.ndarray,
    current: np.ndarray,
    flat_params: np.ndarray,
    max_species: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))

    aug_ox = final_ox_flat * scale
    aug_red = final_red_flat * scale
    aug_current = current * scale

    aug_params = flat_params.copy()
    c_ox_start = 2 * max_species
    c_red_start = 3 * max_species
    aug_params[c_ox_start : c_ox_start + max_species] *= scale
    aug_params[c_red_start : c_red_start + max_species] *= scale

    return aug_ox, aug_red, aug_current, aug_params


def _iter_sim_results(
    sim_args: list[tuple[np.ndarray, float, tuple[np.ndarray, ...], int]],
    max_workers: int,
) -> Iterable[tuple[np.ndarray, ...]]:
    if max_workers == 1:
        for item in map(_run_single_sim, sim_args):
            yield item
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map preserves input ordering; this keeps simulation outputs aligned
        # with the corresponding parameter metadata list.
        for item in executor.map(_run_single_sim, sim_args):
            yield item


def generate_multi_species_dataset(
    n_samples: int,
    key: jax.Array,
    max_species: int = 5,
    nx: int = 50,
    max_workers: int | None = None,
    recipe: str = "curriculum_multitask",
    stage_proportions: tuple[float, float, float] = (0.35, 0.35, 0.30),
    include_invariant_pairs: bool = True,
    invariant_fraction: float = 0.35,
    target_sig_len: int = 200,
) -> tuple[jnp.ndarray, ...]:
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if max_species <= 0:
        raise ValueError(f"max_species must be positive, got {max_species}")
    if nx < 2:
        raise ValueError(f"nx must be >= 2, got {nx}")
    if target_sig_len < 20:
        raise ValueError(f"target_sig_len must be >= 20, got {target_sig_len}")
    if not (0.0 <= invariant_fraction <= 1.0):
        raise ValueError(f"invariant_fraction must be in [0, 1], got {invariant_fraction}")

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    max_workers = max(1, min(max_workers, n_samples))

    base_seed = int(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1))
    rng = np.random.default_rng(base_seed)

    print(f"Generating {n_samples} base simulations with recipe='{recipe}'...")

    stage_ids = _sample_stage_ids(n_samples, recipe, stage_proportions, rng)

    sim_args: list[tuple[np.ndarray, float, tuple[np.ndarray, ...], int]] = []
    sample_specs: list[dict[str, np.ndarray | int | str]] = []

    for i in range(n_samples):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        sample_rng = np.random.default_rng(sample_seed)

        stage_id = int(stage_ids[i])
        task_name = _sample_task_for_stage(sample_rng, stage_id=stage_id, recipe=recipe)
        task_id = int(TASK_TO_ID[task_name])

        params, flat_params = _sample_params_for_task(sample_rng, max_species=max_species, task_name=task_name, stage_id=stage_id)
        E, t_max = _sample_waveform_for_task(sample_rng, task_name=task_name, stage_id=stage_id)

        sim_args.append((E, t_max, params, nx))
        sample_specs.append(
            {
                "flat_params": flat_params,
                "task_id": task_id,
                "stage_id": stage_id,
            }
        )

    final_ox: list[np.ndarray] = []
    final_red: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    signals: list[np.ndarray] = []
    phys_params: list[np.ndarray] = []
    task_ids: list[int] = []
    stage_out: list[int] = []
    aug_ids: list[int] = []

    iterator = _iter_sim_results(sim_args, max_workers=max_workers)
    for i, (c_ox, c_red, i_hist, signal) in enumerate(tqdm(iterator, total=n_samples, desc="Simulating trajectories")):
        spec = sample_specs[i]

        c_ox_flat = c_ox.flatten()
        c_red_flat = c_red.flatten()
        i_resampled = _resample_trace(i_hist, target_sig_len)
        e_resampled = _resample_trace(signal, target_sig_len)
        p_flat = np.asarray(spec["flat_params"], dtype=float)

        final_ox.append(c_ox_flat)
        final_red.append(c_red_flat)
        currents.append(i_resampled)
        signals.append(e_resampled)
        phys_params.append(p_flat)
        task_ids.append(int(spec["task_id"]))
        stage_out.append(int(spec["stage_id"]))
        aug_ids.append(0)

        if include_invariant_pairs and rng.random() < invariant_fraction:
            aug_type = str(rng.choice(["permute_species", "scale_concentration"]))
            if aug_type == "permute_species":
                aug_ox, aug_red, aug_params = _augment_permute_species(
                    final_ox_flat=c_ox_flat,
                    final_red_flat=c_red_flat,
                    flat_params=p_flat,
                    max_species=max_species,
                    nx=nx,
                    rng=rng,
                )
                aug_curr = i_resampled
            else:
                aug_ox, aug_red, aug_curr, aug_params = _augment_scale_concentration(
                    final_ox_flat=c_ox_flat,
                    final_red_flat=c_red_flat,
                    current=i_resampled,
                    flat_params=p_flat,
                    max_species=max_species,
                    rng=rng,
                )

            final_ox.append(aug_ox)
            final_red.append(aug_red)
            currents.append(aug_curr)
            signals.append(e_resampled)
            phys_params.append(aug_params)
            task_ids.append(int(spec["task_id"]))
            stage_out.append(int(spec["stage_id"]))
            aug_ids.append(AUGMENTATION_NAMES.index(aug_type))

    return (
        jnp.asarray(np.stack(final_ox)),
        jnp.asarray(np.stack(final_red)),
        jnp.asarray(np.stack(currents)),
        jnp.asarray(np.stack(signals)),
        jnp.asarray(np.stack(phys_params)),
        jnp.asarray(np.asarray(task_ids, dtype=np.int32)),
        jnp.asarray(np.asarray(stage_out, dtype=np.int32)),
        jnp.asarray(np.asarray(aug_ids, dtype=np.int32)),
    )


def parse_stage_proportions(raw: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("stage_proportions must contain exactly 3 comma-separated values")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-species electrochemical dataset chunks")
    parser.add_argument("--n-samples", type=int, default=5000, help="Base simulations per chunk")
    parser.add_argument("--n-chunks", type=int, default=1, help="Number of chunks to generate")
    parser.add_argument("--max-species", type=int, default=5, help="Maximum species represented in each sample")
    parser.add_argument("--nx", type=int, default=50, help="Spatial grid size")
    parser.add_argument("--workers", type=int, default=None, help="Process workers (default: cpu_count-1)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--recipe", choices=["baseline_random", "curriculum_multitask", "stress_mixture"], default="curriculum_multitask", help="Dataset generation recipe")
    parser.add_argument("--stage-proportions", type=str, default="0.35,0.35,0.30", help="Foundation,bridge,frontier proportions for curriculum recipe")
    parser.add_argument("--target-sig-len", type=int, default=200, help="Resampled length for current and waveform signals")
    parser.add_argument("--invariant-fraction", type=float, default=0.35, help="Probability of adding invariant-augmentation pair per base sample")
    parser.add_argument("--no-invariants", action="store_true", help="Disable invariant pair augmentation")
    parser.add_argument("--output-dir", type=str, default="/tmp/ecsfm/dataset_massive", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {args.n_chunks}")

    stage_proportions = parse_stage_proportions(args.stage_proportions)

    os.makedirs(args.output_dir, exist_ok=True)

    for chunk in range(args.n_chunks):
        key = jax.random.PRNGKey(args.seed + chunk)
        print(f"\n--- Generating chunk {chunk + 1}/{args.n_chunks} ---")
        c_ox, c_red, curr, sigs, params, task_id, stage_id, aug_id = generate_multi_species_dataset(
            n_samples=args.n_samples,
            key=key,
            max_species=args.max_species,
            nx=args.nx,
            max_workers=args.workers,
            recipe=args.recipe,
            stage_proportions=stage_proportions,
            include_invariant_pairs=not args.no_invariants,
            invariant_fraction=args.invariant_fraction,
            target_sig_len=args.target_sig_len,
        )

        chunk_path = os.path.join(args.output_dir, f"chunk_{chunk}.npz")
        np.savez(
            chunk_path,
            ox=np.asarray(c_ox),
            red=np.asarray(c_red),
            i=np.asarray(curr),
            e=np.asarray(sigs),
            p=np.asarray(params),
            task_id=np.asarray(task_id),
            stage_id=np.asarray(stage_id),
            aug_id=np.asarray(aug_id),
            task_names=np.asarray(TASK_NAMES),
            stage_names=np.asarray(STAGE_NAMES),
            augmentation_names=np.asarray(AUGMENTATION_NAMES),
        )
        print(f"Saved chunk {chunk} to {chunk_path} ({len(c_ox)} total rows including augmentations)")

    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()
