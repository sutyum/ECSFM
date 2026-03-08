"""Nernst-Planck transport: diffusion + migration for charged species in 1D."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from ecsfm.sim.mesh import GradedMesh1D, Mesh1D

# Physical constants
_F = 96485.3321       # Faraday constant  (C mol^-1)
_R = 8.314462618      # Gas constant      (J mol^-1 K^-1)
_T_DEFAULT = 298.15   # Default temperature (K)


class NernstPlanck1D(eqx.Module):
    """1D Nernst-Planck flux computation for N charged species.

    The total molar flux for species *i* is:

        J_i = -D_i * dC_i/dx  -  z_i * f_const * D_i * C_i * dphi/dx

    where ``f_const = F / (R*T)``.

    Parameters
    ----------
    mesh : Mesh1D | GradedMesh1D
        Spatial mesh (uniform or graded).
    D : jax.Array, shape (N,)
        Diffusion coefficients for each species (cm^2 s^-1).
    z : jax.Array, shape (N,)
        Algebraic charge numbers (e.g. +1 for H+, -1 for Cl-).
    T : float
        Temperature in Kelvin (default 298.15 K).
    """

    mesh: Mesh1D | GradedMesh1D
    D: jax.Array       # (N,)
    z: jax.Array       # (N,)
    f_const: float     # F / (R*T)  [V^-1]

    def __init__(
        self,
        mesh: Mesh1D | GradedMesh1D,
        D: jax.Array,
        z: jax.Array,
        T: float = _T_DEFAULT,
    ):
        if D.ndim != 1:
            raise ValueError(f"D must be 1-D, got shape {D.shape}")
        if z.ndim != 1:
            raise ValueError(f"z must be 1-D, got shape {z.shape}")
        if D.shape[0] != z.shape[0]:
            raise ValueError(
                f"D and z must have the same length, got {D.shape[0]} and {z.shape[0]}"
            )
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        self.mesh = mesh
        self.D = D
        self.z = z
        self.f_const = _F / (_R * T)

    # ------------------------------------------------------------------
    # Core flux methods
    # ------------------------------------------------------------------

    def compute_migration_flux(self, C: jax.Array, phi: jax.Array) -> jax.Array:
        """Migration flux for every species at every grid point.

        J_mig_i(x) = -z_i * f_const * D_i * C_i(x) * dphi/dx(x)

        Parameters
        ----------
        C : jax.Array, shape (N, nx)
            Concentration profiles.
        phi : jax.Array, shape (nx,)
            Electrostatic potential profile.

        Returns
        -------
        jax.Array, shape (N, nx)
            Migration flux at each grid point for each species.
        """
        dphi_dx = self.mesh.gradient(phi)                       # (nx,)
        # z_i * f * D_i  -> (N,)  broadcast with C_i(x) -> (N, nx)
        coeff = self.z * self.f_const * self.D                  # (N,)
        J_mig = -coeff[:, None] * C * dphi_dx[None, :]         # (N, nx)
        return J_mig

    def compute_total_flux(self, C: jax.Array, phi: jax.Array) -> jax.Array:
        """Total Nernst-Planck flux (diffusion + migration).

        J_i(x) = -D_i * dC_i/dx  -  z_i * f * D_i * C_i * dphi/dx

        Parameters
        ----------
        C : jax.Array, shape (N, nx)
            Concentration profiles.
        phi : jax.Array, shape (nx,)
            Electrostatic potential profile.

        Returns
        -------
        jax.Array, shape (N, nx)
            Total flux at each grid point for each species.
        """
        # Diffusion component: -D_i * dC_i/dx  for each species
        # vmap gradient over species dimension
        dC_dx = jax.vmap(self.mesh.gradient)(C)                 # (N, nx)
        J_diff = -self.D[:, None] * dC_dx                      # (N, nx)
        J_mig = self.compute_migration_flux(C, phi)             # (N, nx)
        return J_diff + J_mig

    # ------------------------------------------------------------------
    # Divergence helper (for explicit source term)
    # ------------------------------------------------------------------

    def migration_source(self, C: jax.Array, phi: jax.Array) -> jax.Array:
        """Explicit source term from migration: -div(J_mig).

        Returns dC_i/dt|_mig = -dJ_mig_i/dx  for each species.

        Parameters
        ----------
        C : jax.Array, shape (N, nx)
            Concentration profiles.
        phi : jax.Array, shape (nx,)
            Electrostatic potential profile.

        Returns
        -------
        jax.Array, shape (N, nx)
            Rate of concentration change due to migration.
        """
        J_mig = self.compute_migration_flux(C, phi)             # (N, nx)
        # Divergence of flux: dJ/dx  (per species)
        dJ_dx = jax.vmap(self.mesh.gradient)(J_mig)             # (N, nx)
        return -dJ_dx
