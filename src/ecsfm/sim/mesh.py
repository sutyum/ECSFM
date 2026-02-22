import jax
import jax.numpy as jnp
import equinox as eqx


class Mesh1D(eqx.Module):
    """1D spatial mesh for finite-difference operators."""

    x: jax.Array
    dx: jax.Array

    def __init__(
        self,
        x_min: float,
        x_max: float,
        n_points: int,
        dtype: jnp.dtype | None = None,
    ):
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        if x_max <= x_min:
            raise ValueError(f"x_max must be greater than x_min, got {x_max} <= {x_min}")

        self.x = jnp.linspace(x_min, x_max, n_points, dtype=dtype)
        self.dx = self.x[1] - self.x[0]

    def gradient(self, C: jax.Array) -> jax.Array:
        """Computes dC/dx with centered finite differences."""
        if C.ndim != 1:
            raise ValueError(f"C must be 1D, got shape {C.shape}")
        if C.shape[0] != self.x.shape[0]:
            raise ValueError(
                f"C length ({C.shape[0]}) must match mesh length ({self.x.shape[0]})"
            )

        # Pad with edge values (zero derivative at edges by default)
        C_padded = jnp.pad(C, (1, 1), mode="edge")

        # Central difference: (C_{i+1} - C_{i-1}) / 2dx
        dC = (C_padded[2:] - C_padded[:-2]) / (jnp.asarray(2.0, dtype=C.dtype) * self.dx)
        return dC

    def laplacian(self, C: jax.Array) -> jax.Array:
        """Computes d2C/dx2 with centered finite differences."""
        if C.ndim != 1:
            raise ValueError(f"C must be 1D, got shape {C.shape}")
        if C.shape[0] != self.x.shape[0]:
            raise ValueError(
                f"C length ({C.shape[0]}) must match mesh length ({self.x.shape[0]})"
            )

        # Pad with edge values (zero flux Neumann condition by default)
        C_padded = jnp.pad(C, (1, 1), mode="edge")

        # Second derivative: (C_{i-1} - 2C_i + C_{i+1}) / dx^2
        two = jnp.asarray(2.0, dtype=C.dtype)
        d2C = (C_padded[:-2] - two * C_padded[1:-1] + C_padded[2:]) / (self.dx**2)
        return d2C
