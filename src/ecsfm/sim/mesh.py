import jax
import jax.numpy as jnp
import equinox as eqx

class Mesh1D(eqx.Module):
    """
    1D Spatial Mesh for finite difference operations.
    """
    x: jax.Array
    dx: float

    def __init__(self, x_min: float, x_max: float, n_points: int):
        """
        Initializes the 1D mesh.
        
        Args:
            x_min: Start of the domain (typically 0, the electrode surface).
            x_max: End of the domain (bulk solution).
            n_points: Number of spatial points.
        """
        self.x = jnp.linspace(x_min, x_max, n_points)
        self.dx = self.x[1] - self.x[0]

    def gradient(self, C: jax.Array) -> jax.Array:
        """
        Computes the 1D gradient dC/dx using central finite differences.
        
        Returns:
            jax.Array of the same shape as C, with padded boundaries.
        """
        # Pad with edge values (zero derivative at edges by default)
        C_padded = jnp.pad(C, (1, 1), mode='edge')
        
        # Central difference: (C_{i+1} - C_{i-1}) / 2dx
        dC = (C_padded[2:] - C_padded[:-2]) / (2 * self.dx)
        return dC

    def laplacian(self, C: jax.Array) -> jax.Array:
        """
        Computes the 1D Laplacian (second derivative) d2C/dx2
        using central finite differences.
        
        Returns:
            jax.Array of the same shape as C, with padded boundaries.
        """
        # Pad with edge values (zero flux Neumann condition by default)
        C_padded = jnp.pad(C, (1, 1), mode='edge')
        
        # Second derivative: (C_{i-1} - 2C_i + C_{i+1}) / dx^2
        d2C = (C_padded[:-2] - 2 * C_padded[1:-1] + C_padded[2:]) / (self.dx ** 2)
        return d2C
