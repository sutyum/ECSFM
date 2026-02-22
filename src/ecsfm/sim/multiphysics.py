from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ecsfm.sim.mesh import Mesh1D


@dataclass(frozen=True)
class MultiPhysicsConfig:
    """Configuration for coupled transport-kinetics-fouling-instrument simulation."""

    Ru_ohm: float = 100.0
    Cdl_F: float = 1e-5
    cdl_floor_fraction: float = 0.25
    cdl_theta_fraction: float = 0.45
    electrode_area_cm2: float = 0.01

    Rfilm_base_ohm: float = 0.0
    Rfilm_theta_max_ohm: float = 600.0
    rfilm_theta_power: float = 1.2

    area_floor_fraction: float = 0.15
    area_theta_power: float = 1.0
    k0_theta_coeff: float = 2.0

    k_ads: float = 2.0e3
    k_des: float = 1.0e-4
    k_clean: float = 6.0e-2
    k_reaction: float = 5.0e-4

    gamma_ads_mol_cm2: float = 2.0e-9
    n_ads: int = 1

    initial_theta: float = 0.0
    fouling_species_idx: int = 0
    cleaning_potential_threshold: float = 0.75
    cleaning_gain: float = 1.0
    kinetics_exp_clip: float = 35.0
    max_current_A: float = 0.05


def _validate_vector(name: str, arr: jax.Array) -> None:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")


def _clip01(x: jax.Array) -> jax.Array:
    return jnp.clip(x, jnp.asarray(0.0, dtype=x.dtype), jnp.asarray(1.0, dtype=x.dtype))


def build_biofouling_protocol(
    *,
    dt: float = 1e-3,
    n_cycles: int = 3,
    baseline_duration_s: float = 2.0,
    foul_duration_s: float = 25.0,
    probe_duration_s: float = 8.0,
    recovery_duration_s: float = 4.0,
    hold_potential_v: float = -0.05,
    fouling_potential_v: float = 0.25,
    probe_freq_hz: float = 20.0,
    probe_amplitude_v: float = 0.02,
    cleaning_steps: tuple[tuple[float, float], ...] = ((0.85, 0.6), (1.05, 0.3), (0.65, 0.5)),
) -> dict[str, Any]:
    """Builds a multi-cycle protocol with fouling phases, EIS probes, and cleaning pulses."""
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be positive, got {n_cycles}")
    if probe_freq_hz <= 0:
        raise ValueError(f"probe_freq_hz must be positive, got {probe_freq_hz}")
    if probe_duration_s <= 0:
        raise ValueError(f"probe_duration_s must be positive, got {probe_duration_s}")

    def n_steps(duration_s: float) -> int:
        if duration_s < 0:
            raise ValueError(f"Duration must be non-negative, got {duration_s}")
        if duration_s == 0:
            return 0
        return max(1, int(np.round(duration_s / dt)))

    e_segments: list[np.ndarray] = []
    clean_segments: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    cursor = 0

    def add_segment(kind: str, e_values: np.ndarray, clean_values: np.ndarray, cycle: int | None = None) -> None:
        nonlocal cursor
        length = int(e_values.shape[0])
        start = cursor
        end = cursor + length
        e_segments.append(e_values.astype(np.float32))
        clean_segments.append(clean_values.astype(np.float32))
        segments.append({"kind": kind, "cycle": cycle, "start_idx": start, "end_idx": end})
        cursor = end

    baseline_steps = n_steps(baseline_duration_s)
    if baseline_steps > 0:
        add_segment(
            "baseline",
            np.full((baseline_steps,), hold_potential_v, dtype=np.float32),
            np.zeros((baseline_steps,), dtype=np.float32),
            cycle=None,
        )

    for cycle_idx in range(n_cycles):
        foul_steps = n_steps(foul_duration_s)
        add_segment(
            "fouling",
            np.full((foul_steps,), fouling_potential_v, dtype=np.float32),
            np.zeros((foul_steps,), dtype=np.float32),
            cycle=cycle_idx,
        )

        probe_steps = n_steps(probe_duration_s)
        t_probe = np.arange(probe_steps, dtype=np.float64) * dt
        probe_wave = hold_potential_v + probe_amplitude_v * np.sin(2.0 * np.pi * probe_freq_hz * t_probe)
        add_segment(
            "probe",
            probe_wave.astype(np.float32),
            np.zeros((probe_steps,), dtype=np.float32),
            cycle=cycle_idx,
        )

        for step_idx, (clean_v, clean_duration_s) in enumerate(cleaning_steps):
            clean_steps = n_steps(clean_duration_s)
            add_segment(
                f"clean_{step_idx}",
                np.full((clean_steps,), clean_v, dtype=np.float32),
                np.ones((clean_steps,), dtype=np.float32),
                cycle=cycle_idx,
            )

        recovery_steps = n_steps(recovery_duration_s)
        add_segment(
            "recovery",
            np.full((recovery_steps,), hold_potential_v, dtype=np.float32),
            np.zeros((recovery_steps,), dtype=np.float32),
            cycle=cycle_idx,
        )

    E_array = np.concatenate(e_segments, axis=0)
    cleaning_mask = np.concatenate(clean_segments, axis=0)
    t_max = float(E_array.shape[0] * dt)
    return {
        "E_array": E_array,
        "cleaning_mask": cleaning_mask,
        "dt": dt,
        "t_max": t_max,
        "segments": segments,
    }


def simulate_multiphysics_electrochem(
    E_array: jax.Array,
    t_max: float,
    D_ox: jax.Array,
    D_red: jax.Array,
    C_bulk_ox: jax.Array,
    C_bulk_red: jax.Array,
    E0: jax.Array,
    k0: jax.Array,
    alpha: jax.Array,
    *,
    config: MultiPhysicsConfig | None = None,
    cleaning_mask: jax.Array | None = None,
    L: float = 0.05,
    nx: int = 200,
    save_every: int | None = 0,
) -> tuple[np.ndarray, ...]:
    """Coupled diffusion + Butler-Volmer + fouling/cleaning + in-loop Ru/Cdl response."""
    cfg = config or MultiPhysicsConfig()

    if nx < 2:
        raise ValueError(f"nx must be >= 2, got {nx}")
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if cfg.Cdl_F <= 0:
        raise ValueError(f"Cdl_F must be positive, got {cfg.Cdl_F}")
    if cfg.Ru_ohm < 0:
        raise ValueError(f"Ru_ohm must be non-negative, got {cfg.Ru_ohm}")
    if cfg.electrode_area_cm2 <= 0:
        raise ValueError(f"electrode_area_cm2 must be positive, got {cfg.electrode_area_cm2}")
    if not (0.0 <= cfg.initial_theta <= 1.0):
        raise ValueError(f"initial_theta must be in [0,1], got {cfg.initial_theta}")
    if not (0.0 <= cfg.area_floor_fraction <= 1.0):
        raise ValueError(f"area_floor_fraction must be in [0,1], got {cfg.area_floor_fraction}")
    if not (0.0 <= cfg.cdl_floor_fraction <= 1.0):
        raise ValueError(f"cdl_floor_fraction must be in [0,1], got {cfg.cdl_floor_fraction}")
    if cfg.cdl_theta_fraction < 0.0:
        raise ValueError(f"cdl_theta_fraction must be non-negative, got {cfg.cdl_theta_fraction}")
    if cfg.area_theta_power <= 0:
        raise ValueError(f"area_theta_power must be positive, got {cfg.area_theta_power}")
    if cfg.rfilm_theta_power <= 0:
        raise ValueError(f"rfilm_theta_power must be positive, got {cfg.rfilm_theta_power}")
    if cfg.kinetics_exp_clip <= 0:
        raise ValueError(f"kinetics_exp_clip must be positive, got {cfg.kinetics_exp_clip}")
    if cfg.max_current_A <= 0:
        raise ValueError(f"max_current_A must be positive, got {cfg.max_current_A}")
    if cfg.k_ads < 0 or cfg.k_des < 0 or cfg.k_clean < 0 or cfg.k_reaction < 0:
        raise ValueError("k_ads, k_des, k_clean, k_reaction must be non-negative")
    if cfg.gamma_ads_mol_cm2 < 0:
        raise ValueError(f"gamma_ads_mol_cm2 must be non-negative, got {cfg.gamma_ads_mol_cm2}")
    if cfg.n_ads <= 0:
        raise ValueError(f"n_ads must be positive, got {cfg.n_ads}")

    E_array = jnp.asarray(E_array)
    D_ox = jnp.asarray(D_ox)
    D_red = jnp.asarray(D_red)
    C_bulk_ox = jnp.asarray(C_bulk_ox)
    C_bulk_red = jnp.asarray(C_bulk_red)
    E0 = jnp.asarray(E0)
    k0 = jnp.asarray(k0)
    alpha = jnp.asarray(alpha)

    _validate_vector("E_array", E_array)
    _validate_vector("D_ox", D_ox)
    _validate_vector("D_red", D_red)
    _validate_vector("C_bulk_ox", C_bulk_ox)
    _validate_vector("C_bulk_red", C_bulk_red)
    _validate_vector("E0", E0)
    _validate_vector("k0", k0)
    _validate_vector("alpha", alpha)

    if E_array.shape[0] < 2:
        raise ValueError(f"E_array must have at least 2 points, got {E_array.shape[0]}")

    n_species = int(D_ox.shape[0])
    if n_species < 1:
        raise ValueError("At least one species is required")
    for name, arr in (
        ("D_red", D_red),
        ("C_bulk_ox", C_bulk_ox),
        ("C_bulk_red", C_bulk_red),
        ("E0", E0),
        ("k0", k0),
        ("alpha", alpha),
    ):
        if arr.shape[0] != n_species:
            raise ValueError(f"{name} length ({arr.shape[0]}) must match D_ox length ({n_species})")

    if bool(jnp.any(D_ox <= 0)) or bool(jnp.any(D_red <= 0)):
        raise ValueError("All diffusion coefficients must be positive")
    if bool(jnp.any(k0 <= 0)):
        raise ValueError("All k0 values must be positive")
    if bool(jnp.any((alpha < 0) | (alpha > 1))):
        raise ValueError("All alpha values must lie in [0,1]")

    fidx = int(cfg.fouling_species_idx)
    if fidx < 0 or fidx >= n_species:
        raise ValueError(f"fouling_species_idx must be in [0,{n_species - 1}], got {fidx}")

    if cleaning_mask is None:
        cleaning_mask_arr = None
    else:
        cleaning_mask_arr = jnp.asarray(cleaning_mask)
        _validate_vector("cleaning_mask", cleaning_mask_arr)
        if cleaning_mask_arr.shape[0] != E_array.shape[0]:
            raise ValueError(
                f"cleaning_mask length ({cleaning_mask_arr.shape[0]}) must match E_array length ({E_array.shape[0]})"
            )

    dtype = jnp.result_type(
        E_array,
        D_ox,
        D_red,
        C_bulk_ox,
        C_bulk_red,
        E0,
        k0,
        alpha,
        jnp.asarray(t_max),
        jnp.asarray(L),
    )
    dtype = jnp.promote_types(dtype, jnp.float32)

    E_array = E_array.astype(dtype)
    D_ox = D_ox.astype(dtype)
    D_red = D_red.astype(dtype)
    C_bulk_ox = C_bulk_ox.astype(dtype)
    C_bulk_red = C_bulk_red.astype(dtype)
    E0 = E0.astype(dtype)
    k0 = k0.astype(dtype)
    alpha = alpha.astype(dtype)
    if cleaning_mask_arr is not None:
        cleaning_mask_arr = cleaning_mask_arr.astype(dtype)

    to_mol_cm3 = jnp.asarray(1e-6, dtype=dtype)
    to_mA = jnp.asarray(1000.0, dtype=dtype)
    F = jnp.asarray(96485.3321, dtype=dtype)
    f_const = jnp.asarray(96485.3321 / (8.314462618 * 298.15), dtype=dtype)
    eps = jnp.asarray(1e-12, dtype=dtype)

    C_bulk_ox_mol = C_bulk_ox * to_mol_cm3
    C_bulk_red_mol = C_bulk_red * to_mol_cm3

    mesh = Mesh1D(x_min=0.0, x_max=L, n_points=nx, dtype=dtype)
    dx = jnp.asarray(mesh.dx, dtype=dtype)

    max_D = jnp.max(jnp.maximum(D_ox, D_red))
    dt = (jnp.asarray(0.1, dtype=dtype) * (dx**2) / max_D) / jnp.asarray(10.0, dtype=dtype)
    dt_f = float(dt)
    if not np.isfinite(dt_f) or dt_f <= 0:
        raise ValueError(f"Computed invalid timestep dt={dt_f}")

    n_steps = int(np.ceil(float(t_max) / dt_f))
    if n_steps < 1:
        raise ValueError(
            f"Simulation would run zero steps (n_steps={n_steps}). Increase t_max or reduce dt."
        )

    if save_every is None or save_every <= 0:
        save_every = max(1, n_steps // 200)
    n_saved = (n_steps + save_every - 1) // save_every

    E_times = jnp.linspace(0.0, float(t_max), E_array.shape[0], dtype=dtype)
    dt_arr = jnp.asarray(dt_f, dtype=dtype)

    def build_dense_matrix(D: jax.Array) -> jax.Array:
        r = D * dt_arr / (dx**2)
        main_diag = jnp.full((nx,), jnp.asarray(1.0, dtype=dtype) + jnp.asarray(2.0, dtype=dtype) * r, dtype=dtype)
        upper_diag = jnp.full((nx - 1,), -r, dtype=dtype)
        lower_diag = jnp.full((nx - 1,), -r, dtype=dtype)
        main_diag = main_diag.at[0].set(jnp.asarray(1.0, dtype=dtype))
        main_diag = main_diag.at[-1].set(jnp.asarray(1.0, dtype=dtype))
        upper_diag = upper_diag.at[0].set(jnp.asarray(0.0, dtype=dtype))
        lower_diag = lower_diag.at[-1].set(jnp.asarray(0.0, dtype=dtype))
        return jnp.diag(main_diag) + jnp.diag(upper_diag, k=1) + jnp.diag(lower_diag, k=-1)

    M_ox = jax.vmap(build_dense_matrix)(D_ox)
    M_red = jax.vmap(build_dense_matrix)(D_red)
    v_solve = jax.vmap(jnp.linalg.solve, in_axes=(0, 0))

    ru = jnp.asarray(cfg.Ru_ohm, dtype=dtype)
    cdl0 = jnp.asarray(cfg.Cdl_F, dtype=dtype)
    cdl_floor_fraction = jnp.asarray(cfg.cdl_floor_fraction, dtype=dtype)
    cdl_theta_fraction = jnp.asarray(cfg.cdl_theta_fraction, dtype=dtype)
    electrode_area = jnp.asarray(cfg.electrode_area_cm2, dtype=dtype)

    rfilm_base = jnp.asarray(cfg.Rfilm_base_ohm, dtype=dtype)
    rfilm_theta_max = jnp.asarray(cfg.Rfilm_theta_max_ohm, dtype=dtype)
    rfilm_theta_power = jnp.asarray(cfg.rfilm_theta_power, dtype=dtype)

    area_floor = jnp.asarray(cfg.area_floor_fraction, dtype=dtype)
    area_power = jnp.asarray(cfg.area_theta_power, dtype=dtype)
    k0_theta_coeff = jnp.asarray(cfg.k0_theta_coeff, dtype=dtype)

    k_ads = jnp.asarray(cfg.k_ads, dtype=dtype)
    k_des = jnp.asarray(cfg.k_des, dtype=dtype)
    k_clean = jnp.asarray(cfg.k_clean, dtype=dtype)
    k_reaction = jnp.asarray(cfg.k_reaction, dtype=dtype)
    gamma_ads = jnp.asarray(cfg.gamma_ads_mol_cm2, dtype=dtype)
    n_ads = jnp.asarray(float(cfg.n_ads), dtype=dtype)

    clean_thresh = jnp.asarray(cfg.cleaning_potential_threshold, dtype=dtype)
    clean_gain = jnp.asarray(cfg.cleaning_gain, dtype=dtype)
    kinetics_exp_clip = jnp.asarray(cfg.kinetics_exp_clip, dtype=dtype)
    max_current_a = jnp.asarray(cfg.max_current_A, dtype=dtype)

    @jax.jit
    def step_fn(carry, i):
        (
            C_ox_state,
            C_red_state,
            E_real_prev,
            theta_prev,
            C_ox_samples,
            C_red_samples,
        ) = carry

        t = i.astype(dtype) * dt_arr
        E_t = jnp.interp(t, E_times, E_array)

        if cleaning_mask_arr is None:
            clean_u = jnp.where(E_t >= clean_thresh, clean_gain, jnp.asarray(0.0, dtype=dtype))
        else:
            clean_u = jnp.interp(t, E_times, cleaning_mask_arr)
            clean_u = jnp.maximum(clean_u, jnp.asarray(0.0, dtype=dtype))

        theta_prev = _clip01(theta_prev)

        area_fraction = area_floor + (jnp.asarray(1.0, dtype=dtype) - area_floor) * (
            jnp.asarray(1.0, dtype=dtype) - theta_prev
        ) ** area_power
        k0_eff = k0 * jnp.exp(-k0_theta_coeff * theta_prev)

        rfilm = rfilm_base + rfilm_theta_max * (theta_prev**rfilm_theta_power)
        cdl_eff = cdl0 * (jnp.asarray(1.0, dtype=dtype) - cdl_theta_fraction * theta_prev) * area_fraction
        cdl_eff = jnp.maximum(cdl_eff, cdl0 * cdl_floor_fraction)
        r_total = ru + rfilm

        eta = E_real_prev - E0
        arg_red = jnp.clip(-alpha * f_const * eta, -kinetics_exp_clip, kinetics_exp_clip)
        arg_ox = jnp.clip(
            (jnp.asarray(1.0, dtype=dtype) - alpha) * f_const * eta,
            -kinetics_exp_clip,
            kinetics_exp_clip,
        )
        k_red = k0_eff * jnp.exp(arg_red)
        k_ox = k0_eff * jnp.exp(arg_ox)

        raw_flux = k_ox * C_red_state[:, 0] - k_red * C_ox_state[:, 0]
        flux = area_fraction * raw_flux

        max_ox_flux = (C_ox_state[:, 0] * dx) / dt_arr
        max_red_flux = (C_red_state[:, 0] * dx) / dt_arr
        flux = jnp.clip(flux, -max_ox_flux, max_red_flux)

        rate_ox_0 = D_ox * (C_ox_state[:, 1] - C_ox_state[:, 0]) / (dx**2) + flux / dx
        rate_red_0 = D_red * (C_red_state[:, 1] - C_red_state[:, 0]) / (dx**2) - flux / dx

        C_ox_surf = C_ox_state[:, 0] + dt_arr * rate_ox_0
        C_red_surf = C_red_state[:, 0] + dt_arr * rate_red_0

        d_ox = C_ox_state.at[:, 0].set(C_ox_surf)
        d_ox = d_ox.at[:, -1].set(C_bulk_ox_mol)
        d_red = C_red_state.at[:, 0].set(C_red_surf)
        d_red = d_red.at[:, -1].set(C_bulk_red_mol)

        C_ox_next = v_solve(M_ox, d_ox)
        C_red_next = v_solve(M_red, d_red)
        C_ox_next = jax.nn.relu(C_ox_next)
        C_red_next = jax.nn.relu(C_red_next)

        I_f_A = electrode_area * F * jnp.sum(flux)
        I_f_A = jnp.clip(I_f_A, -max_current_a, max_current_a)
        V_eff = E_t - I_f_A * r_total
        tau = jnp.maximum(r_total * cdl_eff, eps)
        decay = jnp.exp(-dt_arr / tau)
        E_real_curr = V_eff + (E_real_prev - V_eff) * decay
        I_cap_A = cdl_eff * (E_real_curr - E_real_prev) / jnp.maximum(dt_arr, eps)

        C_surface_total = C_ox_state[fidx, 0] + C_red_state[fidx, 0]
        fouling_drive = k_ads * C_surface_total * (jnp.asarray(1.0, dtype=dtype) - theta_prev)
        reaction_drive = (
            k_reaction * (jnp.abs(I_f_A) / jnp.maximum(electrode_area, eps)) * (jnp.asarray(1.0, dtype=dtype) - theta_prev)
        )
        desorption = k_des * theta_prev
        cleaning = k_clean * clean_u * theta_prev
        dtheta_dt = fouling_drive + reaction_drive - desorption - cleaning
        theta_next = _clip01(theta_prev + dt_arr * dtheta_dt)

        I_ads_A = electrode_area * n_ads * F * gamma_ads * dtheta_dt
        I_total_A = I_f_A + I_cap_A + I_ads_A

        save_idx = i // save_every
        should_save = (i % save_every) == 0

        def _store(samples):
            ox_samples, red_samples = samples
            ox_samples = ox_samples.at[save_idx].set(C_ox_next)
            red_samples = red_samples.at[save_idx].set(C_red_next)
            return ox_samples, red_samples

        C_ox_samples, C_red_samples = jax.lax.cond(
            should_save,
            _store,
            lambda samples: samples,
            (C_ox_samples, C_red_samples),
        )

        outputs = (
            E_t,
            I_total_A * to_mA,
            I_f_A * to_mA,
            I_cap_A * to_mA,
            I_ads_A * to_mA,
            E_real_curr,
            theta_next,
            area_fraction,
            rfilm,
            cdl_eff,
            clean_u,
        )
        new_carry = (C_ox_next, C_red_next, E_real_curr, theta_next, C_ox_samples, C_red_samples)
        return new_carry, outputs

    C_ox_init = jnp.broadcast_to(C_bulk_ox_mol[:, None], (n_species, nx))
    C_red_init = jnp.broadcast_to(C_bulk_red_mol[:, None], (n_species, nx))
    C_ox_samples_init = jnp.zeros((n_saved, n_species, nx), dtype=dtype)
    C_red_samples_init = jnp.zeros((n_saved, n_species, nx), dtype=dtype)
    E_real_init = E_array[0]
    theta_init = jnp.asarray(cfg.initial_theta, dtype=dtype)

    final_carry, outputs = jax.lax.scan(
        step_fn,
        (C_ox_init, C_red_init, E_real_init, theta_init, C_ox_samples_init, C_red_samples_init),
        jnp.arange(n_steps, dtype=jnp.int32),
    )
    _, _, _, _, C_ox_hist, C_red_hist = final_carry

    (
        E_hist,
        I_total_hist,
        I_f_hist,
        I_cap_hist,
        I_ads_hist,
        E_real_hist,
        theta_hist,
        area_hist,
        rfilm_hist,
        cdl_hist,
        clean_hist,
    ) = outputs

    C_ox_hist = C_ox_hist * jnp.asarray(1e6, dtype=dtype)
    C_red_hist = C_red_hist * jnp.asarray(1e6, dtype=dtype)
    E_hist_vis = E_hist[::save_every]
    I_hist_vis = I_total_hist[::save_every]

    return (
        np.asarray(mesh.x),
        np.asarray(C_ox_hist),
        np.asarray(C_red_hist),
        np.asarray(E_hist),
        np.asarray(I_total_hist),
        np.asarray(E_hist_vis),
        np.asarray(I_hist_vis),
        np.asarray(theta_hist),
        np.asarray(area_hist),
        np.asarray(rfilm_hist),
        np.asarray(cdl_hist),
        np.asarray(E_real_hist),
        np.asarray(I_f_hist),
        np.asarray(I_cap_hist),
        np.asarray(I_ads_hist),
        np.asarray(clean_hist),
    )


def _fit_sine_component(t_s: np.ndarray, y: np.ndarray, frequency_hz: float) -> tuple[float, float]:
    omega = 2.0 * np.pi * float(frequency_hz)
    xmat = np.column_stack([np.sin(omega * t_s), np.cos(omega * t_s), np.ones_like(t_s)])
    coeffs, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    a, b, _ = coeffs
    amplitude = float(np.hypot(a, b))
    phase = float(np.arctan2(b, a))
    return amplitude, phase


def estimate_impedance_from_trace(
    t_s: np.ndarray,
    potential_v: np.ndarray,
    current_mA: np.ndarray,
    frequency_hz: float,
    *,
    discard_fraction: float = 0.5,
) -> dict[str, float]:
    """Estimates single-tone impedance from time-domain E/I traces."""
    if frequency_hz <= 0:
        raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")
    if not (0.0 <= discard_fraction < 1.0):
        raise ValueError(f"discard_fraction must be in [0,1), got {discard_fraction}")

    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    potential_v = np.asarray(potential_v, dtype=float).reshape(-1)
    current_mA = np.asarray(current_mA, dtype=float).reshape(-1)
    if not (t_s.shape[0] == potential_v.shape[0] == current_mA.shape[0]):
        raise ValueError("t_s, potential_v, and current_mA must have matching lengths")
    if t_s.shape[0] < 8:
        raise ValueError("Need at least 8 points to estimate impedance")

    start = int(discard_fraction * t_s.shape[0])
    start = min(start, t_s.shape[0] - 4)
    tt = t_s[start:]
    ee = potential_v[start:]
    ii = current_mA[start:]

    amp_e, phase_e = _fit_sine_component(tt, ee, frequency_hz=frequency_hz)
    amp_i_mA, phase_i = _fit_sine_component(tt, ii, frequency_hz=frequency_hz)
    amp_i_a = amp_i_mA / 1000.0
    if amp_i_a <= 1e-12:
        raise ValueError("Estimated current amplitude is too small for impedance calculation")

    z_mag = amp_e / amp_i_a
    z_phase = phase_e - phase_i
    z_real = z_mag * np.cos(z_phase)
    z_imag = z_mag * np.sin(z_phase)
    return {
        "amplitude_v": amp_e,
        "amplitude_mA": amp_i_mA,
        "phase_v_rad": phase_e,
        "phase_i_rad": phase_i,
        "z_mag_ohm": float(z_mag),
        "z_phase_rad": float(z_phase),
        "z_real_ohm": float(z_real),
        "z_imag_ohm": float(z_imag),
    }
