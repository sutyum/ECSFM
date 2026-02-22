import jax
import jax.numpy as jnp
import equinox as eqx
from ecsfm.sim.mesh import Mesh1D


class Diffusion1D(eqx.Module):
    """Solves the 1D diffusion equation dC/dt = D * d2C/dx2."""

    mesh: Mesh1D
    D: float

    def __init__(self, mesh: Mesh1D, D: float):
        D = float(D)
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        self.mesh = mesh
        self.D = D

    def compute_rates(self, C: jax.Array) -> jax.Array:
        """Computes dC/dt for each grid point."""
        d2C = self.mesh.laplacian(C)
        dC_dt = self.D * d2C
        return dC_dt

    def step_euler(self, C: jax.Array, dt: float) -> jax.Array:
        """Simple forward Euler time step."""
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        dC_dt = self.compute_rates(C)
        return C + dt * dC_dt

    def step_implicit(self, C: jax.Array, dt: float, C_surf: float, C_bulk: float) -> jax.Array:
        """
        Implicit Backward Euler time step for 1D diffusion.
        Solves: (I - dt * D * Laplacian) C_next = C
        Using Thomas algorithm (tridiagonal matrix solver) for O(N) speed.
        
        Args:
            C: Current concentration array
            dt: Time step
            C_surf: Dirichlet boundary condition at x=0
            C_bulk: Dirichlet boundary condition at x=L
        """
        nx = C.shape[0]
        if nx < 2:
            raise ValueError(f"C must have at least 2 points, got {nx}")

        dtype = C.dtype
        dt = jnp.asarray(dt, dtype=dtype)
        dx = jnp.asarray(self.mesh.dx, dtype=dtype)
        D = jnp.asarray(self.D, dtype=dtype)
        C_surf = jnp.asarray(C_surf, dtype=dtype)
        C_bulk = jnp.asarray(C_bulk, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        two = jnp.asarray(2.0, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)

        # r = D * dt / dx^2
        r = D * dt / (dx**2)

        # Tridiagonal Bands:
        # -r * C_{i-1} + (1 + 2r) * C_i - r * C_{i+1} = C_current_i
        # Lower diagonal (a), main diagonal (b), upper diagonal (c)
        a = jnp.full((nx - 1,), -r, dtype=dtype)
        b = jnp.full((nx,), one + two * r, dtype=dtype)
        c = jnp.full((nx - 1,), -r, dtype=dtype)

        d = jnp.copy(C)

        # Apply Dirichlet boundary at x=0 (surface): (1) * C_0 = C_surf
        b = b.at[0].set(one)
        c = c.at[0].set(zero)
        d = d.at[0].set(C_surf)

        # Apply Dirichlet boundary at x=L (bulk): (1) * C_{nx-1} = C_bulk
        a = a.at[-1].set(zero)
        b = b.at[-1].set(one)
        d = d.at[-1].set(C_bulk)

        # JAX doesn't have a built-in tridiagonal solver easily exposed for jit.
        # We can write a fast custom Thomas algorithm using lax.scan, or use
        # a standard banded solve if we reshape, but Thomas is incredibly fast.

        # Thomas algorithm forward sweep
        def forward_sweep(carry, elems):
            c_prime_prev, d_prime_prev = carry
            a_i, b_i, c_i, d_i = elems

            denom = b_i - a_i * c_prime_prev
            c_prime = c_i / denom
            d_prime = (d_i - a_i * d_prime_prev) / denom

            return (c_prime, d_prime), (c_prime, d_prime)

        # Pad a list to match lengths for scan (a and c are nx-1, so pad them)
        a_padded = jnp.concatenate([jnp.array([zero], dtype=dtype), a])
        c_padded = jnp.concatenate([c, jnp.array([zero], dtype=dtype)])

        # Initial carry for backward sweep (c'_0, d'_0)
        denom_0 = b[0]
        c_prime_0 = c_padded[0] / denom_0
        d_prime_0 = d[0] / denom_0

        _, (c_prime_arr, d_prime_arr) = jax.lax.scan(
            forward_sweep,
            (c_prime_0, d_prime_0),
            (a_padded[1:], b[1:], c_padded[1:], d[1:])
        )

        c_prime_arr = jnp.concatenate([jnp.array([c_prime_0], dtype=dtype), c_prime_arr])
        d_prime_arr = jnp.concatenate([jnp.array([d_prime_0], dtype=dtype), d_prime_arr])

        # Thomas algorithm backward substitution
        def backward_sweep(x_next, elems):
            c_prime_i, d_prime_i = elems
            x_i = d_prime_i - c_prime_i * x_next
            return x_i, x_i

        _, x_rev = jax.lax.scan(
            backward_sweep,
            d_prime_arr[-1],
            (c_prime_arr[:-1][::-1], d_prime_arr[:-1][::-1])
        )

        x = jnp.concatenate([x_rev[::-1], jnp.array([d_prime_arr[-1]], dtype=dtype)])
        return x
