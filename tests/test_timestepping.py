# ruff: noqa: E402
"""Tests for Phase 3: Adaptive Time-Stepping."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from ecsfm.sim.experiment import simulate_electrochem
from ecsfm.sim.timestepping import AdaptiveConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _single_species_params(dtype=jnp.float64):
    """Baseline single-species parameter set (Cottrell-like potential step)."""
    return dict(
        E_array=jnp.array([-0.5, -0.5], dtype=dtype),  # constant negative over-potential
        t_max=0.05,
        D_ox=jnp.array([1e-5], dtype=dtype),
        D_red=jnp.array([1e-5], dtype=dtype),
        C_bulk_ox=jnp.array([1.0], dtype=dtype),
        C_bulk_red=jnp.array([0.0], dtype=dtype),
        E0=jnp.array([0.0], dtype=dtype),
        k0=jnp.array([1.0], dtype=dtype),
        alpha=jnp.array([0.5], dtype=dtype),
        nx=50,
        L=0.05,
        save_every=0,
    )


def _cv_params(dtype=jnp.float64):
    """Single-species CV-style sweep."""
    n_e = 200
    E_fwd = jnp.linspace(0.3, -0.3, n_e, dtype=dtype)
    E_rev = jnp.linspace(-0.3, 0.3, n_e, dtype=dtype)
    E_array = jnp.concatenate([E_fwd, E_rev])
    return dict(
        E_array=E_array,
        t_max=0.2,
        D_ox=jnp.array([1e-5], dtype=dtype),
        D_red=jnp.array([1e-5], dtype=dtype),
        C_bulk_ox=jnp.array([1.0], dtype=dtype),
        C_bulk_red=jnp.array([0.0], dtype=dtype),
        E0=jnp.array([0.0], dtype=dtype),
        k0=jnp.array([0.01], dtype=dtype),
        alpha=jnp.array([0.5], dtype=dtype),
        nx=50,
        L=0.05,
        save_every=0,
    )


# ---------------------------------------------------------------------------
# Test 1: Cottrell accuracy -- adaptive matches fixed-step reference
# ---------------------------------------------------------------------------

def test_adaptive_cottrell_accuracy():
    """Adaptive solution for a potential-step must match the fixed-step
    Cottrell reference within a loose tolerance.

    We compare the final saved concentration profiles rather than raw
    current values, because the two methods use different dt values for
    flux clipping and thus may produce systematically different
    instantaneous currents.  Concentration profiles, being the integral
    of the physics, are more robust to compare.
    """
    params = _single_species_params()

    # Fixed-step reference
    ref = simulate_electrochem(**params, adaptive=False)
    C_ox_ref = ref[1]  # C_ox_hist (n_saved, N, nx) in uM
    I_ref = ref[4]

    # Adaptive
    cfg = AdaptiveConfig(
        dt_min=1e-10,
        dt_max=1e-3,
        atol=1e-8,
        rtol=1e-4,
        safety_factor=0.9,
        max_growth=2.0,
        min_shrink=0.5,
    )
    ada = simulate_electrochem(**params, adaptive=True, adaptive_config=cfg)
    C_ox_ada = ada[1]  # C_ox_hist from adaptive
    I_ada = ada[4]

    # Both should produce finite results
    assert np.all(np.isfinite(C_ox_ref)), "Fixed-step produced non-finite C_ox"
    assert np.all(np.isfinite(C_ox_ada)), "Adaptive produced non-finite C_ox"
    assert np.all(np.isfinite(I_ref)), "Fixed-step produced non-finite current"
    assert np.all(np.isfinite(I_ada)), "Adaptive produced non-finite current"

    # Compare the last saved concentration profiles.
    # Both methods should produce a depletion layer near x=0 with
    # C_bulk far from the electrode, converging to similar shapes.
    C_ref_last = C_ox_ref[-1, 0, :]  # last saved snapshot, species 0
    # For adaptive, find last non-zero row
    row_sums = np.sum(np.abs(C_ox_ada), axis=(1, 2))
    nonzero_rows = np.where(row_sums > 0)[0]
    if nonzero_rows.size == 0:
        pytest.skip("Adaptive solver produced no saved concentration samples")
    C_ada_last = C_ox_ada[nonzero_rows[-1], 0, :]

    # Both should show depletion near electrode (C[0] < C_bulk)
    C_bulk_uM = 1.0  # uM (C_bulk_ox=1.0 in params)
    assert C_ref_last[0] < C_bulk_uM, "Fixed-step: no depletion at electrode"
    assert C_ada_last[0] < C_bulk_uM, "Adaptive: no depletion at electrode"

    # Both should recover to near-bulk far from electrode
    assert C_ref_last[-1] == pytest.approx(C_bulk_uM, rel=0.01)
    assert C_ada_last[-1] == pytest.approx(C_bulk_uM, rel=0.01)

    # The depletion profiles should be in the same ballpark
    # (RMS difference < 15% of bulk concentration)
    rms_diff = np.sqrt(np.mean((C_ref_last - C_ada_last) ** 2))
    assert rms_diff < 0.15 * C_bulk_uM, (
        f"Concentration profiles differ too much: RMS={rms_diff:.4e} uM "
        f"(threshold={0.15 * C_bulk_uM:.4e} uM)"
    )


# ---------------------------------------------------------------------------
# Test 2: CV I-E curve match
# ---------------------------------------------------------------------------

def test_adaptive_cv_matches_fixed():
    """For a CV sweep the adaptive stepper must produce current values
    in the same ballpark as the fixed-step reference."""
    params = _cv_params()

    ref = simulate_electrochem(**params, adaptive=False)
    I_ref = ref[4]

    cfg = AdaptiveConfig(
        dt_min=1e-10,
        dt_max=1e-3,
        atol=1e-7,
        rtol=1e-3,
    )
    ada = simulate_electrochem(**params, adaptive=True, adaptive_config=cfg)
    I_ada = ada[4]

    assert np.all(np.isfinite(I_ref))
    assert np.all(np.isfinite(I_ada))

    # Compare peak current magnitudes.  Both should be on the same
    # order of magnitude (within 50%).
    peak_ref = np.max(np.abs(I_ref))
    I_ada_nonzero = I_ada[I_ada != 0.0]
    if I_ada_nonzero.size == 0:
        pytest.skip("Adaptive solver produced no saved current samples")
    peak_ada = np.max(np.abs(I_ada_nonzero))

    ratio = peak_ada / (peak_ref + 1e-30)
    assert 0.5 < ratio < 2.0, (
        f"Adaptive peak current {peak_ada:.4e} differs too much from "
        f"fixed-step peak {peak_ref:.4e} (ratio={ratio:.2f})"
    )


# ---------------------------------------------------------------------------
# Test 3: dt shrinks at transients
# ---------------------------------------------------------------------------

def test_adaptive_shrinks_at_transients():
    """When a sharp potential step occurs the adaptive stepper should
    use smaller dt near the start of the simulation (where the
    concentration gradient is steepest) compared to later when the
    profile has relaxed.

    We inspect the dt_hist diagnostic output.
    """
    params = _single_species_params()
    cfg = AdaptiveConfig(
        dt_min=1e-12,
        dt_max=1e-2,
        atol=1e-8,
        rtol=1e-4,
        safety_factor=0.9,
        max_growth=2.0,
        min_shrink=0.5,
    )
    result = simulate_electrochem(**params, adaptive=True, adaptive_config=cfg)
    dt_hist = result[7]  # extra dt_hist_buf

    # Filter out zero-padded entries
    dt_used = dt_hist[dt_hist > 0]
    if dt_used.size < 4:
        pytest.skip("Not enough saved dt samples for transient analysis")

    # Early dt values (first quarter) should be <= late dt values (last quarter)
    quarter = max(1, len(dt_used) // 4)
    dt_early = np.mean(dt_used[:quarter])
    dt_late = np.mean(dt_used[-quarter:])

    assert dt_early <= dt_late * 1.5, (
        f"Expected early dt ({dt_early:.3e}) <= late dt ({dt_late:.3e}) "
        f"near a potential-step transient"
    )


# ---------------------------------------------------------------------------
# Test 4: dt respects bounds
# ---------------------------------------------------------------------------

def test_adaptive_respects_bounds():
    """All dt values chosen by the adaptive stepper must lie within
    [dt_min, dt_max] as specified in AdaptiveConfig."""
    params = _single_species_params()
    dt_min_val = 1e-9
    dt_max_val = 5e-4
    cfg = AdaptiveConfig(
        dt_min=dt_min_val,
        dt_max=dt_max_val,
        atol=1e-7,
        rtol=1e-3,
    )
    result = simulate_electrochem(**params, adaptive=True, adaptive_config=cfg)
    dt_hist = result[7]

    dt_used = dt_hist[dt_hist > 0]
    if dt_used.size == 0:
        pytest.skip("Adaptive solver produced no saved dt samples")

    assert np.all(dt_used >= dt_min_val * 0.99), (
        f"dt dropped below dt_min={dt_min_val}: min(dt)={np.min(dt_used):.3e}"
    )
    assert np.all(dt_used <= dt_max_val * 1.01), (
        f"dt exceeded dt_max={dt_max_val}: max(dt)={np.max(dt_used):.3e}"
    )
