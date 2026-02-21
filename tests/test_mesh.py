import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ecsfm.sim.mesh import Mesh1D

def test_mesh1d_initialization():
    mesh = Mesh1D(x_min=0.0, x_max=1.0, n_points=11)
    assert mesh.x.shape == (11,)
    assert jnp.isclose(mesh.dx, 0.1)
    assert jnp.isclose(mesh.x[0], 0.0)
    assert jnp.isclose(mesh.x[-1], 1.0)

def test_mesh1d_gradient():
    mesh = Mesh1D(x_min=0.0, x_max=1.0, n_points=101)
    # Test linear function: C(x) = 3x, then dC/dx = 3
    C_linear = 3.0 * mesh.x
    grad_linear = mesh.gradient(C_linear)
    assert jnp.allclose(grad_linear[1:-1], 3.0, atol=1e-5)
    
    # Test quadratic function: C(x) = x^2, then dC/dx = 2x
    C_quad = mesh.x**2
    grad_quad = mesh.gradient(C_quad)
    expected_grad = 2.0 * mesh.x
    assert jnp.allclose(grad_quad[1:-1], expected_grad[1:-1], atol=1e-3)

def test_mesh1d_laplacian():
    mesh = Mesh1D(x_min=0.0, x_max=1.0, n_points=101)
    # Test quadratic function: C(x) = x^2, then d2C/dx2 = 2
    C_quad = mesh.x**2
    lap_quad = mesh.laplacian(C_quad)
    assert jnp.allclose(lap_quad[1:-1], 2.0, atol=1e-4)
    
    # Test cubic function: C(x) = x^3, then d2C/dx2 = 6x
    C_cubic = mesh.x**3
    lap_cubic = mesh.laplacian(C_cubic)
    expected_lap = 6.0 * mesh.x
    assert jnp.allclose(lap_cubic[1:-1], expected_lap[1:-1], atol=1e-3)
