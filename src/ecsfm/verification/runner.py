"""Run verification test cases through the ECSFM simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ecsfm.verification.test_cases import TestCaseSpec


@dataclass
class SimResult:
    """Structured result from a simulator run."""

    time: np.ndarray
    potential: np.ndarray
    current: np.ndarray
    concentration_profiles: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def _build_step_waveform(spec: TestCaseSpec) -> tuple[jnp.ndarray, float]:
    """Build a potential step waveform from spec.

    Returns (E_array, t_max).
    """
    params = spec.waveform.params
    E_initial = params.get("E_initial", 0.5)
    E_step = params.get("E_step", -0.5)
    t_max = params.get("t_max", 0.1)
    # Create a waveform that steps from E_initial to E_step at t=0+
    # Use 500 points: first point at E_initial, rest at E_step
    n_wave = 500
    E_array = jnp.full(n_wave, E_step)
    E_array = E_array.at[0].set(E_initial)
    return E_array, t_max


def _build_cv_waveform(spec: TestCaseSpec) -> tuple[jnp.ndarray, float]:
    """Build a cyclic voltammetry waveform from spec.

    Returns (E_array, t_max).
    """
    params = spec.waveform.params
    E_start = params.get("E_start", 0.5)
    E_vertex = params.get("E_vertex", -0.5)
    scan_rate = params.get("scan_rate", 1.0)

    E_range = abs(E_start - E_vertex)
    t_max = 2.0 * E_range / scan_rate
    n_wave = 1000

    half = n_wave // 2
    E_forward = jnp.linspace(E_start, E_vertex, half)
    E_reverse = jnp.linspace(E_vertex, E_start, n_wave - half)
    E_array = jnp.concatenate([E_forward, E_reverse])
    return E_array, t_max


def _extract_species_arrays(spec: TestCaseSpec) -> dict:
    """Extract D_ox, D_red, C_bulk_ox, C_bulk_red arrays from spec species list.

    Species are paired: (Ox_0, Red_0), (Ox_1, Red_1), ...
    For N kinetics entries, we need N oxidized and N reduced species.
    """
    n_kinetics = len(spec.kinetics)

    # If species has 2*n_kinetics entries, pair them as (ox, red)
    if len(spec.species) == 2 * n_kinetics:
        D_ox = []
        D_red = []
        C_bulk_ox = []
        C_bulk_red = []
        for i in range(n_kinetics):
            ox_spec = spec.species[2 * i]
            red_spec = spec.species[2 * i + 1]
            D_ox.append(ox_spec.D)
            D_red.append(red_spec.D)
            C_bulk_ox.append(ox_spec.C_bulk)
            C_bulk_red.append(red_spec.C_bulk)
    elif len(spec.species) == n_kinetics:
        # Single species per kinetics entry, assume reduced form absent
        D_ox = [s.D for s in spec.species]
        D_red = [s.D for s in spec.species]
        C_bulk_ox = [s.C_bulk for s in spec.species]
        C_bulk_red = [0.0] * n_kinetics
    else:
        raise ValueError(
            f"Cannot map {len(spec.species)} species to {n_kinetics} kinetics entries. "
            f"Expected {n_kinetics} or {2 * n_kinetics} species."
        )

    return {
        "D_ox": jnp.array(D_ox),
        "D_red": jnp.array(D_red),
        "C_bulk_ox": jnp.array(C_bulk_ox),
        "C_bulk_red": jnp.array(C_bulk_red),
    }


def _extract_kinetics_arrays(spec: TestCaseSpec) -> dict:
    """Extract E0, k0, alpha arrays from spec kinetics list."""
    return {
        "E0": jnp.array([k.E0 for k in spec.kinetics]),
        "k0": jnp.array([k.k0 for k in spec.kinetics]),
        "alpha": jnp.array([k.alpha for k in spec.kinetics]),
    }


def run_case(spec: TestCaseSpec) -> SimResult:
    """Run a test case through the ECSFM simulator and return structured results.

    Handles step, cv, and eis waveform types by mapping spec fields to
    the ``simulate_electrochem()`` API.
    """
    from ecsfm.sim.experiment import simulate_electrochem

    species_kw = _extract_species_arrays(spec)
    kinetics_kw = _extract_kinetics_arrays(spec)

    waveform_type = spec.waveform.type

    if waveform_type == "eis":
        return _run_eis_case(spec, species_kw, kinetics_kw)

    if waveform_type == "step":
        E_array, t_max = _build_step_waveform(spec)
    elif waveform_type == "cv":
        E_array, t_max = _build_cv_waveform(spec)
    else:
        raise ValueError(f"Unknown waveform type: {waveform_type!r}")

    x, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis, I_hist_vis = (
        simulate_electrochem(
            E_array=E_array,
            t_max=t_max,
            L=spec.domain_length_cm,
            nx=spec.n_points,
            grading_factor=spec.grading_factor,
            save_every=0,
            **species_kw,
            **kinetics_kw,
        )
    )

    n_steps = E_hist.shape[0]
    time = np.linspace(0.0, t_max, n_steps)

    return SimResult(
        time=time,
        potential=np.asarray(E_hist),
        current=np.asarray(I_hist),
        concentration_profiles=np.asarray(C_ox_hist),
        metadata={
            "case_name": spec.name,
            "category": spec.category,
            "waveform_type": waveform_type,
            "n_steps": n_steps,
            "t_max": t_max,
            "x": x,
            "C_ox_hist": np.asarray(C_ox_hist),
            "C_red_hist": np.asarray(C_red_hist),
            "E_hist_vis": np.asarray(E_hist_vis),
            "I_hist_vis": np.asarray(I_hist_vis),
        },
    )


def _run_eis_case(
    spec: TestCaseSpec,
    species_kw: dict,
    kinetics_kw: dict,
) -> SimResult:
    """Run an EIS test case using multiphysics impedance sweep.

    Falls back to single-frequency probes using ``simulate_electrochem``
    if the multiphysics module is unavailable or the case is simple enough.
    """
    from ecsfm.sim.multiphysics import (
        MultiPhysicsConfig,
        estimate_impedance_from_trace,
        simulate_multiphysics_electrochem,
    )

    params = spec.waveform.params
    frequencies = np.asarray(params.get("frequencies_hz", [1.0, 5.0, 10.0]), dtype=float)
    amplitude_v = params.get("amplitude_v", 0.01)
    dc_potential = params.get("dc_potential_v", -0.02)
    t_window = params.get("t_window_s", 8.0)

    cfg = MultiPhysicsConfig(
        initial_theta=0.0,
        k_ads=0.0,
        k_des=0.0,
        k_reaction=0.0,
        k_clean=0.0,
    )

    z_real_list = []
    z_imag_list = []
    z_mag_list = []
    z_phase_list = []

    for freq in frequencies:
        dt_wave = 1e-3
        t_wave = np.arange(0.0, t_window, dt_wave, dtype=np.float32)
        e_wave = dc_potential + amplitude_v * np.sin(2.0 * np.pi * float(freq) * t_wave)

        out = simulate_multiphysics_electrochem(
            E_array=jnp.asarray(e_wave),
            t_max=float(t_window),
            nx=spec.n_points,
            L=spec.domain_length_cm,
            config=cfg,
            **species_kw,
            **kinetics_kw,
        )
        e_hist = np.asarray(out[3], dtype=float)
        i_hist = np.asarray(out[4], dtype=float)
        t_hist = np.linspace(0.0, t_window, e_hist.shape[0], endpoint=False, dtype=float)

        metrics = estimate_impedance_from_trace(
            t_s=t_hist,
            potential_v=e_hist,
            current_mA=i_hist,
            frequency_hz=float(freq),
            discard_fraction=0.5,
        )
        z_real_list.append(metrics["z_real_ohm"])
        z_imag_list.append(metrics["z_imag_ohm"])
        z_mag_list.append(metrics["z_mag_ohm"])
        z_phase_list.append(metrics["z_phase_rad"])

    # Build a combined "current" trace from impedance data (magnitude at each freq)
    z_real = np.array(z_real_list)
    z_imag = np.array(z_imag_list)
    z_mag = np.array(z_mag_list)
    z_phase = np.array(z_phase_list)

    return SimResult(
        time=frequencies,
        potential=z_real,
        current=z_imag,
        concentration_profiles=None,
        metadata={
            "case_name": spec.name,
            "category": spec.category,
            "waveform_type": "eis",
            "frequencies_hz": frequencies,
            "z_real_ohm": z_real,
            "z_imag_ohm": z_imag,
            "z_mag_ohm": z_mag,
            "z_phase_rad": z_phase,
        },
    )
