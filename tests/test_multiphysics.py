import jax.numpy as jnp
import numpy as np

from ecsfm.sim.multiphysics import (
    MultiPhysicsConfig,
    build_biofouling_protocol,
    estimate_impedance_from_trace,
    simulate_multiphysics_electrochem,
)


def _species_kwargs() -> dict[str, jnp.ndarray]:
    return {
        "D_ox": jnp.asarray([1.0e-5], dtype=jnp.float32),
        "D_red": jnp.asarray([1.0e-5], dtype=jnp.float32),
        "C_bulk_ox": jnp.asarray([1.0], dtype=jnp.float32),
        "C_bulk_red": jnp.asarray([0.0], dtype=jnp.float32),
        "E0": jnp.asarray([0.0], dtype=jnp.float32),
        "k0": jnp.asarray([0.02], dtype=jnp.float32),
        "alpha": jnp.asarray([0.5], dtype=jnp.float32),
    }


def test_build_biofouling_protocol_contains_probe_and_cleaning_segments():
    protocol = build_biofouling_protocol(
        dt=2e-3,
        n_cycles=2,
        baseline_duration_s=0.2,
        foul_duration_s=1.0,
        probe_duration_s=0.4,
        recovery_duration_s=0.2,
        cleaning_steps=((0.9, 0.2), (1.0, 0.1)),
    )
    e_array = np.asarray(protocol["E_array"])
    cleaning_mask = np.asarray(protocol["cleaning_mask"])
    segments = protocol["segments"]

    assert e_array.ndim == 1
    assert cleaning_mask.ndim == 1
    assert e_array.shape[0] == cleaning_mask.shape[0]
    assert e_array.shape[0] > 10
    assert float(np.max(cleaning_mask)) > 0.5
    kinds = {item["kind"] for item in segments}
    assert "probe" in kinds
    assert "clean_0" in kinds


def test_multiphysics_solver_outputs_are_finite_and_bounded():
    t_window = 2.0
    dt_wave = 1e-3
    t_wave = np.arange(0.0, t_window, dt_wave, dtype=np.float32)
    e_wave = -0.02 + 0.01 * np.sin(2.0 * np.pi * 3.0 * t_wave)

    out = simulate_multiphysics_electrochem(
        E_array=jnp.asarray(e_wave),
        t_max=t_window,
        nx=22,
        config=MultiPhysicsConfig(),
        **_species_kwargs(),
    )
    _, c_ox_hist, c_red_hist, e_hist, i_hist, _, _, theta, area, rfilm, cdl, e_real, *_ = out

    assert np.isfinite(c_ox_hist).all()
    assert np.isfinite(c_red_hist).all()
    assert np.isfinite(e_hist).all()
    assert np.isfinite(i_hist).all()
    assert np.isfinite(theta).all()
    assert np.isfinite(area).all()
    assert np.isfinite(rfilm).all()
    assert np.isfinite(cdl).all()
    assert np.isfinite(e_real).all()
    assert float(np.min(c_ox_hist)) >= -1e-8
    assert float(np.min(c_red_hist)) >= -1e-8
    assert float(np.min(theta)) >= -1e-8
    assert float(np.max(theta)) <= 1.0 + 1e-8
    assert float(np.min(area)) >= MultiPhysicsConfig().area_floor_fraction - 1e-6
    assert float(np.max(area)) <= 1.0 + 1e-6
    assert float(np.min(cdl)) > 0.0


def test_cleaning_protocol_reduces_final_fouling_and_film_resistance():
    protocol = build_biofouling_protocol(
        dt=1e-3,
        n_cycles=3,
        baseline_duration_s=0.4,
        foul_duration_s=6.0,
        probe_duration_s=1.0,
        recovery_duration_s=0.4,
        cleaning_steps=((0.95, 0.8), (1.05, 0.6)),
    )
    e_array = jnp.asarray(protocol["E_array"], dtype=jnp.float32)
    cleaning_mask = jnp.asarray(protocol["cleaning_mask"], dtype=jnp.float32)
    t_max = float(protocol["t_max"])

    cfg = MultiPhysicsConfig(
        k_ads=4500.0,
        k_reaction=0.004,
        k_clean=0.45,
        Rfilm_theta_max_ohm=1200.0,
        cdl_theta_fraction=0.75,
        area_floor_fraction=0.12,
        k0_theta_coeff=2.2,
    )

    out_clean = simulate_multiphysics_electrochem(
        E_array=e_array,
        cleaning_mask=cleaning_mask,
        t_max=t_max,
        nx=24,
        config=cfg,
        **_species_kwargs(),
    )
    out_no_clean = simulate_multiphysics_electrochem(
        E_array=e_array,
        cleaning_mask=jnp.zeros_like(cleaning_mask),
        t_max=t_max,
        nx=24,
        config=cfg,
        **_species_kwargs(),
    )

    theta_clean = np.asarray(out_clean[7], dtype=float)
    theta_no_clean = np.asarray(out_no_clean[7], dtype=float)
    rfilm_clean = np.asarray(out_clean[9], dtype=float)
    rfilm_no_clean = np.asarray(out_no_clean[9], dtype=float)

    assert float(theta_clean[-1]) < float(theta_no_clean[-1])
    assert float(rfilm_clean[-1]) < float(rfilm_no_clean[-1])
    assert float(np.max(theta_no_clean)) > 0.02


def test_impedance_magnitude_increases_for_biofouled_state():
    freq_hz = 3.0
    t_window = 6.0
    dt_wave = 1e-3
    t_wave = np.arange(0.0, t_window, dt_wave, dtype=np.float32)
    e_wave = -0.02 + 0.012 * np.sin(2.0 * np.pi * freq_hz * t_wave)

    common = dict(
        E_array=jnp.asarray(e_wave),
        t_max=t_window,
        nx=28,
        **_species_kwargs(),
    )
    clean_cfg = MultiPhysicsConfig(
        initial_theta=0.0,
        k_ads=0.0,
        k_des=0.0,
        k_reaction=0.0,
        k_clean=0.0,
        Rfilm_theta_max_ohm=1500.0,
        cdl_theta_fraction=0.8,
        area_floor_fraction=0.1,
        k0_theta_coeff=2.5,
    )
    fouled_cfg = MultiPhysicsConfig(
        initial_theta=0.8,
        k_ads=0.0,
        k_des=0.0,
        k_reaction=0.0,
        k_clean=0.0,
        Rfilm_theta_max_ohm=1500.0,
        cdl_theta_fraction=0.8,
        area_floor_fraction=0.1,
        k0_theta_coeff=2.5,
    )

    out_clean = simulate_multiphysics_electrochem(config=clean_cfg, **common)
    out_fouled = simulate_multiphysics_electrochem(config=fouled_cfg, **common)

    e_hist_clean = np.asarray(out_clean[3], dtype=float)
    i_hist_clean = np.asarray(out_clean[4], dtype=float)
    e_hist_fouled = np.asarray(out_fouled[3], dtype=float)
    i_hist_fouled = np.asarray(out_fouled[4], dtype=float)
    t_hist_clean = np.linspace(0.0, t_window, e_hist_clean.shape[0], endpoint=False, dtype=float)
    t_hist_fouled = np.linspace(0.0, t_window, e_hist_fouled.shape[0], endpoint=False, dtype=float)

    z_clean = estimate_impedance_from_trace(
        t_s=t_hist_clean,
        potential_v=e_hist_clean,
        current_mA=i_hist_clean,
        frequency_hz=freq_hz,
        discard_fraction=0.5,
    )
    z_fouled = estimate_impedance_from_trace(
        t_s=t_hist_fouled,
        potential_v=e_hist_fouled,
        current_mA=i_hist_fouled,
        frequency_hz=freq_hz,
        discard_fraction=0.5,
    )

    assert z_fouled["z_mag_ohm"] > z_clean["z_mag_ohm"] * 1.5
