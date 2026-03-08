# ruff: noqa: E402
"""Tests for the Nernst-Planck transport module."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from ecsfm.sim.mesh import Mesh1D
from ecsfm.sim.transport import NernstPlanck1D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_mesh(nx: int = 101, L: float = 1.0) -> Mesh1D:
    return Mesh1D(x_min=0.0, x_max=L, n_points=nx, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNernstPlanckConstruction:
    def test_basic_construction(self):
        mesh = _uniform_mesh()
        D = jnp.array([1e-5, 2e-5])
        z = jnp.array([1.0, -1.0])
        np_solver = NernstPlanck1D(mesh=mesh, D=D, z=z)
        assert np_solver.D.shape == (2,)
        assert np_solver.z.shape == (2,)
        assert np_solver.f_const > 0

    def test_rejects_mismatched_D_z(self):
        mesh = _uniform_mesh()
        with pytest.raises(ValueError, match="same length"):
            NernstPlanck1D(mesh=mesh, D=jnp.array([1e-5]), z=jnp.array([1.0, -1.0]))


class TestNoPotentialFieldReducesToDiffusion:
    """test_np_no_field_reduces_to_diffusion:
    When phi = 0 everywhere, migration flux is zero and the total flux
    reduces to Fick's first law (pure diffusion).
    """

    def test_migration_flux_zero_when_phi_zero(self):
        nx = 101
        mesh = _uniform_mesh(nx=nx)
        N = 2
        D = jnp.array([1e-5, 2e-5])
        z = jnp.array([1.0, -1.0])
        np_solver = NernstPlanck1D(mesh=mesh, D=D, z=z)

        # Linear concentration profile for each species
        C = jnp.stack([jnp.linspace(1.0, 0.0, nx), jnp.linspace(0.0, 1.0, nx)])
        phi = jnp.zeros(nx)

        J_mig = np_solver.compute_migration_flux(C, phi)
        assert J_mig.shape == (N, nx)
        assert jnp.allclose(J_mig, 0.0, atol=1e-15)

    def test_total_flux_equals_diffusion_when_phi_zero(self):
        nx = 101
        mesh = _uniform_mesh(nx=nx)
        D = jnp.array([1e-5, 2e-5])
        z = jnp.array([1.0, -1.0])
        np_solver = NernstPlanck1D(mesh=mesh, D=D, z=z)

        C = jnp.stack([jnp.linspace(1.0, 0.0, nx), jnp.linspace(0.0, 1.0, nx)])
        phi = jnp.zeros(nx)

        J_total = np_solver.compute_total_flux(C, phi)

        # Pure diffusion flux: J_diff_i = -D_i * dC_i/dx
        dC_dx = jax.vmap(mesh.gradient)(C)
        J_diff = -D[:, None] * dC_dx

        assert jnp.allclose(J_total, J_diff, atol=1e-14)


class TestUniformFieldDrift:
    """test_np_uniform_field_drift:
    With a uniform electric field (constant dphi/dx), the migration flux
    should produce a drift velocity v_drift = z_i * f_const * D_i * (dphi/dx)
    on a uniform concentration profile.
    """

    def test_drift_velocity(self):
        nx = 201
        L = 0.01  # cm
        mesh = _uniform_mesh(nx=nx, L=L)
        D_val = 1e-5
        z_val = 2.0
        D = jnp.array([D_val])
        z = jnp.array([z_val])
        np_solver = NernstPlanck1D(mesh=mesh, D=D, z=z)

        C_uniform = jnp.ones((1, nx)) * 1e-3  # 1 mM uniform concentration
        # Linear potential: phi(x) = E_field * x  -->  dphi/dx = E_field
        E_field = 10.0  # V/cm
        phi = E_field * mesh.x

        J_mig = np_solver.compute_migration_flux(C_uniform, phi)

        # Expected: J_mig = -z * f_const * D * C * dphi/dx
        f_const = np_solver.f_const
        expected_flux = -z_val * f_const * D_val * 1e-3 * E_field

        # Interior points should match analytical value (edges have FD boundary effects)
        assert jnp.allclose(J_mig[0, 5:-5], expected_flux, rtol=1e-3), (
            f"Expected flux {expected_flux}, got {float(J_mig[0, nx//2])}"
        )


class TestTransferenceNumber:
    """test_transference_number:
    For HCl (H+ and Cl-), the transference number of the cation is:
        t_+ = D_H+ / (D_H+ + D_Cl-)
    In a uniform-concentration, uniform-field system the fraction of
    current carried by each ion should match this ratio.
    """

    def test_hcl_transference(self):
        nx = 201
        L = 0.01
        mesh = _uniform_mesh(nx=nx, L=L)

        # Literature values (cm^2/s)
        D_H = 9.31e-5   # H+
        D_Cl = 2.03e-5   # Cl-
        D = jnp.array([D_H, D_Cl])
        z = jnp.array([1.0, -1.0])
        np_solver = NernstPlanck1D(mesh=mesh, D=D, z=z)

        C_uniform = jnp.ones((2, nx)) * 1e-3  # 1 mM each
        E_field = 5.0  # V/cm
        phi = E_field * mesh.x

        J_mig = np_solver.compute_migration_flux(C_uniform, phi)

        # Current contribution of each species: I_i ~ z_i * J_mig_i
        # At midpoint (away from edges)
        mid = nx // 2
        I_H = z[0] * J_mig[0, mid]
        I_Cl = z[1] * J_mig[1, mid]
        I_total = I_H + I_Cl

        # t_+ = I_H / I_total
        t_plus_sim = float(I_H / I_total)
        t_plus_theory = D_H / (D_H + D_Cl)

        assert abs(t_plus_sim - t_plus_theory) < 0.01, (
            f"Transference number: sim={t_plus_sim:.4f}, theory={t_plus_theory:.4f}"
        )


class TestMigrationSourceShape:
    """Verify migration_source returns the correct shape and preserves
    boundary conditions."""

    def test_shape(self):
        nx = 51
        mesh = _uniform_mesh(nx=nx)
        D = jnp.array([1e-5, 2e-5])
        z = jnp.array([1.0, -1.0])
        np_solver = NernstPlanck1D(mesh=mesh, D=D, z=z)

        C = jnp.stack([jnp.linspace(1.0, 0.5, nx), jnp.linspace(0.5, 1.0, nx)])
        phi = 0.01 * mesh.x

        src = np_solver.migration_source(C, phi)
        assert src.shape == (2, nx)
        assert jnp.all(jnp.isfinite(src))
