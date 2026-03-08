# ruff: noqa: E402
"""Tests for the electroneutrality and Poisson potential solvers."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from ecsfm.sim.mesh import Mesh1D
from ecsfm.sim.potential import ElectroneutralitySolver, PoissonSolver
from ecsfm.sim.experiment import simulate_electrochem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_mesh(nx: int = 201, L: float = 0.01) -> Mesh1D:
    return Mesh1D(x_min=0.0, x_max=L, n_points=nx, dtype=jnp.float64)


# Physical constants (same as production code)
_F = 96485.3321
_R = 8.314462618
_T = 298.15


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestElectroneutralityBinaryElectrolyte:
    """test_electroneutrality_binary_electrolyte:
    For a binary 1:1 electrolyte with a concentration gradient, the
    Henderson equation gives:

        E_j = (RT/F) * (D_+ - D_-) / (D_+ + D_-) * ln(C_high / C_low)

    The electroneutrality solver should reproduce this liquid-junction
    potential within a few percent on a fine mesh.
    """

    def test_henderson_equation(self):
        nx = 501
        L = 0.01  # 0.01 cm
        mesh = _uniform_mesh(nx=nx, L=L)

        # KCl-like: D+ ~ D- but slightly different
        D_K = 1.96e-5    # K+
        D_Cl = 2.03e-5   # Cl-
        D = jnp.array([D_K, D_Cl])
        z = jnp.array([1.0, -1.0])

        solver = ElectroneutralitySolver(mesh=mesh, z=z, D=D)

        # Concentration gradient from 10 mM to 1 mM (electroneutral: C_+ = C_-)
        C_high = 10e-6   # mol/cm^3  (10 mM)
        C_low = 1e-6     # mol/cm^3  (1 mM)
        C_profile = jnp.linspace(C_high, C_low, nx)
        C = jnp.stack([C_profile, C_profile])  # (2, nx): K+ and Cl- identical

        phi = solver.solve_phi(C)

        # Henderson equation for 1:1 electrolyte liquid junction potential.
        # The electroneutrality-derived potential difference between x=0
        # (high concentration) and x=L (low concentration, reference phi=0) is:
        #
        #   phi(high) - phi(low) = -(RT/F) * (D+ - D-) / (D+ + D-) * ln(C_high / C_low)
        #
        # This follows from dphi/dx = -(RT/F) * (z+*D+*dC+/dx + z-*D-*dC-/dx)
        #                                       / (z+^2*D+*C+ + z-^2*D-*C-)
        # For a 1:-1 electrolyte with C+ = C- = C(x):
        #   dphi/dx = -(RT/F) * (D+ - D-) / ((D+ + D-)*C) * dC/dx
        #           = -(RT/F) * (D+ - D-) / (D+ + D-) * d(ln C)/dx
        # Integrating from x=L (C_low) to x=0 (C_high):
        #   phi(0) - phi(L) = -(RT/F) * (D+ - D-) / (D+ + D-) * ln(C_high / C_low)
        RT_F = _R * _T / _F
        E_j_theory = -RT_F * (D_K - D_Cl) / (D_K + D_Cl) * np.log(C_high / C_low)

        # phi[0] - phi[-1] should match E_j_theory  (phi[-1] = 0 by convention)
        E_j_sim = float(phi[0])

        # Allow 5% relative error for the finite-difference discretization
        assert abs(E_j_sim - E_j_theory) < 0.05 * abs(E_j_theory) + 1e-6, (
            f"Henderson potential: sim={E_j_sim:.6e} V, theory={E_j_theory:.6e} V"
        )


class TestChargeBalancePreserved:
    """test_charge_balance_preserved:
    After solving phi from electroneutrality, sum(z_i * C_i) should
    remain approximately zero at all grid points (the solver does not
    modify C, but we verify that the input that satisfies
    electroneutrality still does after the phi computation).
    """

    def test_charge_balance(self):
        nx = 201
        mesh = _uniform_mesh(nx=nx)
        D = jnp.array([9.31e-5, 2.03e-5])  # H+, Cl-
        z = jnp.array([1.0, -1.0])

        solver = ElectroneutralitySolver(mesh=mesh, z=z, D=D)

        # Electroneutral profiles: C_+ = C_- everywhere
        C_profile = jnp.linspace(5e-6, 1e-6, nx)
        C = jnp.stack([C_profile, C_profile])

        # Charge balance: sum(z_i * C_i) should be ~0
        charge_density = jnp.sum(z[:, None] * C, axis=0)
        assert jnp.allclose(charge_density, 0.0, atol=1e-20), (
            f"Max charge imbalance: {float(jnp.max(jnp.abs(charge_density)))}"
        )

        # After solve_phi, phi should still be finite and well-behaved
        phi = solver.solve_phi(C)
        assert jnp.all(jnp.isfinite(phi))
        # phi[-1] = 0 by convention
        assert float(jnp.abs(phi[-1])) < 1e-15


class TestMigrationOffByDefault:
    """test_migration_off_by_default:
    Calling simulate_electrochem without z or enable_migration should
    give bit-identical output to the baseline (no migration code path).
    """

    def test_bit_identical_without_z(self):
        params = dict(
            E_array=jnp.linspace(-0.2, 0.2, 20, dtype=jnp.float64),
            t_max=0.05,
            D_ox=jnp.array([1e-5], dtype=jnp.float64),
            D_red=jnp.array([1e-5], dtype=jnp.float64),
            C_bulk_ox=jnp.array([1.0], dtype=jnp.float64),
            C_bulk_red=jnp.array([0.0], dtype=jnp.float64),
            E0=jnp.array([0.0], dtype=jnp.float64),
            k0=jnp.array([0.01], dtype=jnp.float64),
            alpha=jnp.array([0.5], dtype=jnp.float64),
            nx=32,
            save_every=0,
        )

        out_default = simulate_electrochem(**params)
        out_explicit = simulate_electrochem(**params, enable_migration=False, z=None)

        # Compare every element of the output tuple
        for idx, (a, b) in enumerate(zip(out_default, out_explicit)):
            np.testing.assert_array_equal(
                a, b, err_msg=f"Output element {idx} differs"
            )


class TestPoissonSolverStub:
    """The PoissonSolver should provide the same interface and currently
    delegate to ElectroneutralitySolver."""

    def test_same_result_as_electroneutrality(self):
        nx = 101
        mesh = _uniform_mesh(nx=nx)
        D = jnp.array([1e-5, 2e-5])
        z = jnp.array([1.0, -1.0])

        en = ElectroneutralitySolver(mesh=mesh, z=z, D=D)
        ps = PoissonSolver(mesh=mesh, z=z, D=D)

        C = jnp.stack([
            jnp.linspace(5e-6, 1e-6, nx),
            jnp.linspace(5e-6, 1e-6, nx),
        ])

        phi_en = en.solve_phi(C)
        phi_ps = ps.solve_phi(C)
        assert jnp.allclose(phi_en, phi_ps, atol=1e-15)


class TestPhiBulkReference:
    """phi should be zero at the bulk boundary (x_max) and generally
    non-zero elsewhere when there is a concentration gradient with
    asymmetric diffusion coefficients."""

    def test_phi_profile_shape(self):
        nx = 201
        mesh = _uniform_mesh(nx=nx)
        D = jnp.array([9.31e-5, 2.03e-5])  # H+, Cl- (very different D)
        z = jnp.array([1.0, -1.0])

        solver = ElectroneutralitySolver(mesh=mesh, z=z, D=D)

        C_profile = jnp.linspace(10e-6, 1e-6, nx)
        C = jnp.stack([C_profile, C_profile])

        phi = solver.solve_phi(C)

        # phi[-1] should be zero (bulk reference)
        assert abs(float(phi[-1])) < 1e-15

        # phi[0] should be non-zero because D_H+ >> D_Cl-
        assert abs(float(phi[0])) > 1e-5

        # phi should be monotonic (for this symmetric electroneutral case
        # with D+ > D-, the faster cation creates a positive junction
        # potential at the high-concentration side)
        dphi = jnp.diff(phi)
        # All increments should have the same sign
        signs = jnp.sign(dphi)
        # Most should agree (allow minor numerical noise at edges)
        dominant_sign = jnp.sign(jnp.sum(signs))
        assert float(jnp.mean(signs == dominant_sign)) > 0.95
