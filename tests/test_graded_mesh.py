# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from ecsfm.sim.mesh import GradedMesh1D, Mesh1D
from ecsfm.sim.physics import Diffusion1D


class TestGradedMesh1D:
    def test_monotonicity_and_density(self):
        """x values are monotonically increasing and denser near x=0."""
        mesh = GradedMesh1D(x_min=0.0, x_max=1.0, n_points=101, grading_factor=3.0)
        assert mesh.x.shape == (101,)
        assert jnp.all(jnp.diff(mesh.x) > 0)
        # First interval should be smaller than last interval
        assert float(mesh.dx_array[0]) < float(mesh.dx_array[-1])

    def test_graded_reduces_to_uniform(self):
        """grading_factor ~ 0 matches Mesh1D."""
        graded = GradedMesh1D(x_min=0.0, x_max=1.0, n_points=51, grading_factor=0.0)
        uniform = Mesh1D(x_min=0.0, x_max=1.0, n_points=51)
        assert jnp.allclose(graded.x, uniform.x, atol=1e-12)

    def test_gradient_polynomial(self):
        """Gradient of x^2 matches 2x on interior points."""
        mesh = GradedMesh1D(x_min=0.0, x_max=1.0, n_points=201, grading_factor=3.0)
        C = mesh.x**2
        grad = mesh.gradient(C)
        expected = 2.0 * mesh.x
        # Check interior points (skip first and last)
        assert jnp.allclose(grad[2:-2], expected[2:-2], atol=1e-3)

    def test_laplacian_polynomial(self):
        """Laplacian of x^3 matches 6x on interior points."""
        mesh = GradedMesh1D(x_min=0.0, x_max=1.0, n_points=201, grading_factor=3.0)
        C = mesh.x**3
        lap = mesh.laplacian(C)
        expected = 6.0 * mesh.x
        # Check interior points (skip boundaries)
        assert jnp.allclose(lap[2:-2], expected[2:-2], atol=0.1)

    def test_laplacian_quadratic(self):
        """Laplacian of x^2 should be constant 2."""
        mesh = GradedMesh1D(x_min=0.0, x_max=1.0, n_points=201, grading_factor=3.0)
        C = mesh.x**2
        lap = mesh.laplacian(C)
        assert jnp.allclose(lap[1:-1], 2.0, atol=1e-6)

    def test_endpoints(self):
        """x[0] = x_min, x[-1] = x_max."""
        mesh = GradedMesh1D(x_min=0.0, x_max=0.05, n_points=100, grading_factor=4.0)
        assert jnp.isclose(mesh.x[0], 0.0)
        assert jnp.isclose(mesh.x[-1], 0.05)

    def test_invalid_grading_factor(self):
        with pytest.raises(ValueError):
            GradedMesh1D(x_min=0.0, x_max=1.0, n_points=10, grading_factor=-1.0)


class TestCottrellGradedMesh:
    def test_cottrell_graded_mesh(self):
        """Graded mesh with nx=200 should match Cottrell within 2%."""
        D = 1e-5
        C_bulk = 1.0
        L = 0.05
        nx = 200

        mesh = GradedMesh1D(x_min=0.0, x_max=L, n_points=nx, grading_factor=4.0)
        diffusion = Diffusion1D(mesh, D)

        C = jnp.full(nx, C_bulk)
        C = C.at[0].set(0.0)
        dt = float(0.01 * mesh.dx**2 / D)

        @jax.jit
        def step(C_current):
            C_next = diffusion.step_implicit(C_current, dt, C_surf=0.0, C_bulk=C_bulk)
            return C_next

        time_to_sample = 0.1
        steps = int(time_to_sample / dt)

        C_final = jax.lax.fori_loop(0, steps, lambda i, c: step(c), C)

        # Flux at electrode using forward difference with non-uniform spacing
        flux_sim = D * (C_final[1] - C_final[0]) / mesh.dx_array[0]
        flux_analytical = C_bulk * jnp.sqrt(D / (jnp.pi * time_to_sample))

        rel_error = jnp.abs(flux_sim - flux_analytical) / flux_analytical
        assert rel_error < 0.02, f"Error {float(rel_error)*100:.2f}%: sim={float(flux_sim)}, analytical={float(flux_analytical)}"


class TestSimulateElectrochemGraded:
    def test_graded_output_shapes(self):
        """simulate_electrochem with grading produces same output shapes, finite values."""
        from ecsfm.sim.experiment import simulate_electrochem

        out = simulate_electrochem(
            E_array=jnp.linspace(-0.2, 0.2, 20, dtype=jnp.float32),
            t_max=0.2,
            D_ox=jnp.array([1e-5], dtype=jnp.float32),
            D_red=jnp.array([1e-5], dtype=jnp.float32),
            C_bulk_ox=jnp.array([1.0], dtype=jnp.float32),
            C_bulk_red=jnp.array([0.0], dtype=jnp.float32),
            E0=jnp.array([0.0], dtype=jnp.float32),
            k0=jnp.array([0.01], dtype=jnp.float32),
            alpha=jnp.array([0.5], dtype=jnp.float32),
            nx=32,
            save_every=0,
            grading_factor=3.0,
        )
        x, c_ox, c_red, e_hist, i_hist, e_vis, i_vis = out
        assert x.shape == (32,)
        assert not np.isnan(i_hist).any()
        assert not np.isnan(c_ox).any()
        # x should be non-uniformly spaced
        dx = np.diff(x)
        assert dx[0] < dx[-1]  # denser near electrode
