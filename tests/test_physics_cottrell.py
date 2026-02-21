import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ecsfm.sim.mesh import Mesh1D
from ecsfm.sim.physics import Diffusion1D

def test_cottrell_flux():
    """
    Verifies the finite difference diffusion equation against the
    analytical Cottrell equation for planar diffusion.
    """
    D = 1e-5     # diffusion coefficient (cm^2/s)
    C_bulk = 1.0 # bulk concentration
    
    # Mesh needs to be fine enough to resolve the steep concentration 
    # gradient near the electrode.
    L = 0.05 # distance into bulk (cm)
    nx = 2000
    mesh = Mesh1D(x_min=0.0, x_max=L, n_points=nx)
    diffusion = Diffusion1D(mesh, D)
    
    # Diffusive relaxation requires D * dt / dx^2 <= 0.5 for stability
    # dx = 0.05 / 2000 = 2.5e-5. dx^2 = 6.25e-10.
    # D/dx^2 = 1.6e4. dt <= 0.5 / 1.6e4 = 3.125e-5. Let's use 1e-5.
    dt = 1e-5 
    
    # Initial state
    C = jnp.full(nx, C_bulk)
    # The potential step drops C(0) immediately to 0
    C = C.at[0].set(0.0)
    
    @jax.jit
    def step(C_current):
        rates = diffusion.compute_rates(C_current)
        C_next = C_current + dt * rates
        
        # Enforce boundary: x=0 is consumed, x=L is bulk
        C_next = C_next.at[0].set(0.0)
        C_next = C_next.at[-1].set(C_bulk)
        return C_next

    time_to_sample = 0.1
    steps_to_sample = int(time_to_sample / dt)
    
    # Use jax.lax.fori_loop for compiled speed
    C_final = jax.lax.fori_loop(0, steps_to_sample, lambda i, c: step(c), C)
    
    # Flux at electrode J = D * dC/dx
    # Using simple forward difference: J = D * (C_final[1] - C_final[0]) / dx
    flux_sim = D * (C_final[1] - C_final[0]) / mesh.dx
    
    # Analytical Cottrell flux: J = C_bulk * sqrt(D / (pi * t))
    flux_analytical = C_bulk * jnp.sqrt(D / (jnp.pi * time_to_sample))
    
    rel_error = jnp.abs(flux_sim - flux_analytical) / flux_analytical
    
    # The error should be quite low (<2% for a fine mesh)
    assert rel_error < 0.02, f"Error {rel_error*100:.2f}%: sim J={flux_sim}, analytical J={flux_analytical}"
