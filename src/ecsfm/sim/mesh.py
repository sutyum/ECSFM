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


class GradedMesh1D(eqx.Module):
    """1D non-uniform mesh with exponential grading near x=0 (electrode).

    Uses exponential distribution: x = x_min + (x_max-x_min) * (exp(g*xi) - 1) / (exp(g) - 1)
    where xi is uniform on [0, 1] and g is the grading_factor.

    Higher grading_factor = more refinement near x=0.
    grading_factor ~ 0 reduces to uniform spacing.
    """

    x: jax.Array
    dx: jax.Array          # dx_min for CFL reference
    dx_array: jax.Array    # per-interval spacings (n_points-1,)

    def __init__(
        self,
        x_min: float,
        x_max: float,
        n_points: int,
        grading_factor: float = 3.0,
        dtype: jnp.dtype | None = None,
    ):
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        if x_max <= x_min:
            raise ValueError(f"x_max must be greater than x_min, got {x_max} <= {x_min}")
        if grading_factor < 0:
            raise ValueError(f"grading_factor must be >= 0, got {grading_factor}")

        xi = jnp.linspace(0.0, 1.0, n_points, dtype=dtype)

        if grading_factor < 1e-10:
            # Near-zero grading -> uniform mesh
            x = jnp.linspace(x_min, x_max, n_points, dtype=dtype)
        else:
            g = jnp.asarray(grading_factor, dtype=dtype)
            x = jnp.asarray(x_min, dtype=dtype) + (
                jnp.asarray(x_max - x_min, dtype=dtype)
                * (jnp.exp(g * xi) - 1.0)
                / (jnp.exp(g) - 1.0)
            )

        self.x = x
        dx_arr = x[1:] - x[:-1]
        self.dx_array = dx_arr
        self.dx = dx_arr[0]  # smallest spacing (near electrode)

    def gradient(self, C: jax.Array) -> jax.Array:
        """Computes dC/dx with variable-spacing central differences."""
        if C.ndim != 1:
            raise ValueError(f"C must be 1D, got shape {C.shape}")
        if C.shape[0] != self.x.shape[0]:
            raise ValueError(
                f"C length ({C.shape[0]}) must match mesh length ({self.x.shape[0]})"
            )

        # Variable-spacing central difference for interior points:
        # dC/dx_i = (C_{i+1}*h_{i-1}^2 - C_{i-1}*h_i^2 + C_i*(h_i^2 - h_{i-1}^2))
        #           / (h_i * h_{i-1} * (h_i + h_{i-1}))
        # Simplified: use (C_{i+1} - C_{i-1}) / (x_{i+1} - x_{i-1}) for non-uniform
        # This is the standard non-uniform central difference.
        n = C.shape[0]
        h_fwd = self.dx_array  # (n-1,) = x[1:] - x[:-1]

        # For interior points i=1..n-2:
        # dC/dx_i ~ (h_{i-1}^2 * C_{i+1} - h_i^2 * C_{i-1} + (h_i^2 - h_{i-1}^2) * C_i)
        #           / (h_i * h_{i-1} * (h_i + h_{i-1}))
        h_left = h_fwd[:-1]   # h_{i-1} for i=1..n-2
        h_right = h_fwd[1:]   # h_i for i=1..n-2

        numer = (
            h_left**2 * C[2:]
            - h_right**2 * C[:-2]
            + (h_right**2 - h_left**2) * C[1:-1]
        )
        denom = h_left * h_right * (h_left + h_right)
        grad_interior = numer / denom

        # Edge values: forward/backward difference
        grad_0 = (C[1] - C[0]) / h_fwd[0]
        grad_n = (C[-1] - C[-2]) / h_fwd[-1]

        return jnp.concatenate([grad_0[None], grad_interior, grad_n[None]])

    def laplacian(self, C: jax.Array) -> jax.Array:
        """Computes d2C/dx2 with variable-spacing finite differences.

        Uses: d2C/dx2_i = 2*(C_{i+1}*h_{i-1} - C_i*(h_i+h_{i-1}) + C_{i-1}*h_i)
                          / (h_i * h_{i-1} * (h_i + h_{i-1}))
        """
        if C.ndim != 1:
            raise ValueError(f"C must be 1D, got shape {C.shape}")
        if C.shape[0] != self.x.shape[0]:
            raise ValueError(
                f"C length ({C.shape[0]}) must match mesh length ({self.x.shape[0]})"
            )

        h_fwd = self.dx_array
        h_left = h_fwd[:-1]   # h_{i-1} for i=1..n-2
        h_right = h_fwd[1:]   # h_i for i=1..n-2

        two = jnp.asarray(2.0, dtype=C.dtype)
        numer = two * (
            C[2:] * h_left
            - C[1:-1] * (h_right + h_left)
            + C[:-2] * h_right
        )
        denom = h_left * h_right * (h_left + h_right)
        lap_interior = numer / denom

        # Edge values: zero Neumann (same as uniform mesh edge padding)
        lap_0 = jnp.asarray(0.0, dtype=C.dtype)
        lap_n = jnp.asarray(0.0, dtype=C.dtype)

        return jnp.concatenate([lap_0[None], lap_interior, lap_n[None]])
