"""Electrostatic potential solvers for coupled Nernst-Planck transport.

Two solvers are provided:

* ``ElectroneutralitySolver`` -- derives dphi/dx from the local
  electroneutrality condition sum(z_i * C_i) = 0 at every point.
  This avoids solving the Poisson equation and is the standard
  approach for dilute-solution theory.

* ``PoissonSolver`` -- (stub) placeholder for a future Poisson-based
  solver that resolves the electrical double layer explicitly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from ecsfm.sim.mesh import GradedMesh1D, Mesh1D

# Physical constants
_F = 96485.3321       # Faraday constant  (C mol^-1)
_R = 8.314462618      # Gas constant      (J mol^-1 K^-1)
_T_DEFAULT = 298.15   # Default temperature (K)


class ElectroneutralitySolver(eqx.Module):
    """Compute the electric potential profile phi(x) from electroneutrality.

    Under the electroneutrality approximation the gradient of the
    electrostatic potential is fully determined by the concentration
    gradients:

        dphi/dx = - (RT/F) * sum_i(z_i * D_i * dC_i/dx)
                             / sum_i(z_i^2 * D_i * C_i)

    The absolute potential is obtained by integrating from the bulk
    boundary (x = x_max) inward with the reference phi(x_max) = 0.

    Parameters
    ----------
    mesh : Mesh1D | GradedMesh1D
        Spatial mesh.
    z : jax.Array, shape (N,)
        Charge numbers.
    D : jax.Array, shape (N,)
        Diffusion coefficients (cm^2 s^-1).
    T : float
        Temperature (K).
    """

    mesh: Mesh1D | GradedMesh1D
    z: jax.Array       # (N,)
    D: jax.Array       # (N,)
    RT_over_F: float   # R*T / F  [V]

    def __init__(
        self,
        mesh: Mesh1D | GradedMesh1D,
        z: jax.Array,
        D: jax.Array,
        T: float = _T_DEFAULT,
    ):
        if z.ndim != 1:
            raise ValueError(f"z must be 1-D, got shape {z.shape}")
        if D.ndim != 1:
            raise ValueError(f"D must be 1-D, got shape {D.shape}")
        if z.shape[0] != D.shape[0]:
            raise ValueError(
                f"z and D must have the same length, got {z.shape[0]} and {D.shape[0]}"
            )
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        self.mesh = mesh
        self.z = z
        self.D = D
        self.RT_over_F = _R * T / _F

    def solve_phi(self, C: jax.Array) -> jax.Array:
        """Return the electrostatic potential profile phi(x).

        Parameters
        ----------
        C : jax.Array, shape (N, nx)
            Concentration profiles for all species.

        Returns
        -------
        jax.Array, shape (nx,)
            Electrostatic potential at each grid point, referenced to
            phi = 0 at the bulk boundary (x_max).
        """
        dtype = C.dtype
        RT_F = jnp.asarray(self.RT_over_F, dtype=dtype)
        eps = jnp.asarray(1e-30, dtype=dtype)

        # dC_i/dx  for every species: shape (N, nx)
        dC_dx = jax.vmap(self.mesh.gradient)(C)

        # Numerator:  sum_i( z_i * D_i * dC_i/dx )   -> (nx,)
        zD = self.z * self.D                            # (N,)
        numer = jnp.sum(zD[:, None] * dC_dx, axis=0)   # (nx,)

        # Denominator: sum_i( z_i^2 * D_i * C_i )     -> (nx,)
        z2D = self.z ** 2 * self.D                      # (N,)
        denom = jnp.sum(z2D[:, None] * C, axis=0)      # (nx,)
        denom = jnp.maximum(denom, eps)                 # avoid division by zero

        # dphi/dx = -(RT/F) * numer / denom            -> (nx,)
        dphi_dx = -RT_F * numer / denom

        # Integrate from bulk (x_max, index -1) inward.
        # phi[-1] = 0.  For each point going backward:
        #   phi[j] = phi[j+1] - dphi_dx_avg * (x[j+1] - x[j])
        # We flip, cumsum, then flip back.

        x = self.mesh.x.astype(dtype)
        dx_intervals = jnp.diff(x)                     # (nx-1,)

        # Average dphi/dx at each interval midpoint (trapezoidal rule)
        dphi_avg = (dphi_dx[:-1] + dphi_dx[1:]) / jnp.asarray(2.0, dtype=dtype)

        # Increments of phi from left to right:  delta_phi[j] = dphi_avg[j] * dx[j]
        delta_phi = dphi_avg * dx_intervals             # (nx-1,)

        # phi[0] = 0 as temporary, then cumsum gives phi at each node relative to x[0].
        phi_rel = jnp.concatenate([jnp.zeros(1, dtype=dtype), jnp.cumsum(delta_phi)])

        # Shift so that phi[-1] = 0  (bulk reference)
        phi = phi_rel - phi_rel[-1]
        return phi


class PoissonSolver(eqx.Module):
    """Stub: compute phi(x) from the Poisson equation.

    This class provides the same ``solve_phi`` interface as
    ``ElectroneutralitySolver`` but will eventually solve:

        d^2 phi / dx^2 = -(F / epsilon) * sum_i(z_i * C_i)

    For now it falls back to the electroneutrality approximation.

    Parameters
    ----------
    mesh : Mesh1D | GradedMesh1D
        Spatial mesh.
    z : jax.Array, shape (N,)
        Charge numbers.
    D : jax.Array, shape (N,)
        Diffusion coefficients (cm^2 s^-1).
    T : float
        Temperature (K).
    """

    _en_solver: ElectroneutralitySolver

    def __init__(
        self,
        mesh: Mesh1D | GradedMesh1D,
        z: jax.Array,
        D: jax.Array,
        T: float = _T_DEFAULT,
    ):
        self._en_solver = ElectroneutralitySolver(mesh=mesh, z=z, D=D, T=T)

    def solve_phi(self, C: jax.Array) -> jax.Array:
        """Return phi(x) -- currently delegates to ElectroneutralitySolver.

        Parameters
        ----------
        C : jax.Array, shape (N, nx)

        Returns
        -------
        jax.Array, shape (nx,)
        """
        return self._en_solver.solve_phi(C)
