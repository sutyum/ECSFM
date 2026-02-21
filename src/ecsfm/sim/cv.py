import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from ecsfm.sim.mesh import Mesh1D
from ecsfm.sim.physics import Diffusion1D
from ecsfm.sim.kinetics import ButlerVolmer

def simulate_cv(
    D_ox: float = 1e-5,
    D_red: float = 1e-5,
    C_bulk_ox: float = 1.0,
    C_bulk_red: float = 0.0,
    E0: float = 0.0,
    k0: float = 0.01,
    alpha: float = 0.5,
    scan_rate: float = 0.1,
    E_start: float = 0.5,
    E_vertex: float = -0.5,
    L: float = 0.05,
    nx: int = 200,
    save_every: int = 1
):
    """
    Simulates a Cyclic Voltammetry experiment.
    
    Args:
        D_ox: Diffusion coefficient of the oxidized species (cm^2/s).
        D_red: Diffusion coefficient of the reduced species (cm^2/s).
        C_bulk_ox: Bulk concentration of the oxidized species (mM).
        C_bulk_red: Bulk concentration of the reduced species (mM).
        E0: Formal reduction potential (V).
        k0: Standard rate constant (cm/s).
        alpha: Charge transfer coefficient.
        scan_rate: Potential scan rate (V/s).
        E_start: Starting potential (V).
        E_vertex: Vertex potential (V) where the scan reverses.
        L: Length of the simulation domain (cm).
        nx: Number of spatial grid points.
        save_every: Save state frequency (1 = every step, >1 = thinner array).
        
    Returns:
        tuple: (x, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis)
               x is the spatial mesh array, all other arrays are time-series outputs.
    """
    # Force float64 for numerical precision
    jax.config.update("jax_enable_x64", True)

    # We need to standardize units to cm and mol to compute real current
    # 1 mM = 1e-6 mol/cm^3
    C_bulk_ox_mol_cm3 = C_bulk_ox * 1e-6
    C_bulk_red_mol_cm3 = C_bulk_red * 1e-6
    
    mesh = Mesh1D(x_min=0.0, x_max=L, n_points=nx)
    
    diff_ox = Diffusion1D(mesh, D_ox)
    diff_red = Diffusion1D(mesh, D_red)
    kinetics = ButlerVolmer(k0=k0, alpha=alpha, E0=E0)
    
    # Explicit diffusion stability requires D * dt / dx^2 < 0.5
    max_D = max(D_ox, D_red)
    
    # We need a much smaller dt to stabilize the *kinetics*, not just diffusion
    # When kinetics rate > diffusion rate, the boundary bin empties instantly and oscillates.
    dt_max_stable = 0.1 * (mesh.dx ** 2) / max_D 
    dt = dt_max_stable / 10.0 # Force a tiny dt
    
    t_max = 2 * abs(E_start - E_vertex) / scan_rate
    n_steps = int(t_max / dt)
    
    if save_every is None or save_every <= 0:
        save_every = max(1, n_steps // 200)
    
    @jax.jit
    def step_fn(state, i):
        C_ox, C_red = state
        t = i * dt
        
        # Calculate potential E(t)
        # Triangular wave
        half_time = t_max / 2
        
        # JAX conditional for triangle wave
        E_t = jnp.where(t < half_time,
                        E_start - scan_rate * t,
                        E_vertex + scan_rate * (t - half_time))

        # 2. Semi-Implicit Boundary Step at x=0
        # Instead of projecting C_surf explicitly (which causes NaN if flux is too high),
        # we step the boundary bin explicitly first, utilizing the Fickian limit
        
        # Calculate flux from current state
        flux = kinetics.flux(E_t, C_red[0], C_ox[0])
        
        # We cap the flux to the absolute maximum that could be supplied by the adjacent bin 
        # in this timestep, preventing negative concentrations natively.
        max_ox_flux = (C_ox[0] * mesh.dx) / dt        # Can't consume more Ox than exists
        max_red_flux = (C_red[0] * mesh.dx) / dt      # Can't consume more Red than exists
        
        # If flux > 0 (consuming Red), cap at max_red_flux
        # If flux < 0 (consuming Ox), cap at -max_ox_flux
        flux = jnp.clip(flux, -max_ox_flux, max_red_flux)
        
        # Explicitly step the first bin using Fickian mass balance
        # dC/dt = D * d2C/dx2 + Flux / dx
        # For the boundary bin, diffusion rate is D * (C[1] - C[0]) / dx^2
        rate_ox_0 = D_ox * (C_ox[1] - C_ox[0]) / (mesh.dx**2) + flux / mesh.dx
        rate_red_0 = D_red * (C_red[1] - C_red[0]) / (mesh.dx**2) - flux / mesh.dx
        
        C_ox_surf = C_ox[0] + dt * rate_ox_0
        C_red_surf = C_red[0] + dt * rate_red_0
        
        # 3. Implicit bulk stepping
        C_ox_next = diff_ox.step_implicit(C_ox, dt, C_ox_surf, C_bulk_ox_mol_cm3)
        C_red_next = diff_red.step_implicit(C_red, dt, C_red_surf, C_bulk_red_mol_cm3)
        
        # Non-negativity constraint
        C_ox_next = jax.nn.relu(C_ox_next)
        C_red_next = jax.nn.relu(C_red_next)
        
        # Real Macroscopic Current Generation (I = n * F * J)
        # Using the *capped* diffusion flux to represent actual physical turnover rates!
        # Multiplied by 1000 to convert from A/cm^2 to mA/cm^2
        I_t = kinetics.n * kinetics.F * flux * 1000.0
        
        # Return state and things we want to save
        # jax.lax.scan expects (carry, y) = f(carry, x)
        new_state = (C_ox_next, C_red_next)
        history_frame = (C_ox_next, C_red_next, E_t, I_t)
        return new_state, history_frame

    # Initial concentrations
    C_ox_init = jnp.full(nx, C_bulk_ox_mol_cm3)
    C_red_init = jnp.full(nx, C_bulk_red_mol_cm3)
    
    # lax.scan is perfect for this: collects history and pushes state forward
    _, (C_ox_hist, C_red_hist, E_hist, I_hist) = jax.lax.scan(
        step_fn, 
        (C_ox_init, C_red_init), 
        jnp.arange(n_steps)
    )
    
    # Thin the history down for visualization
    C_ox_hist = C_ox_hist[::save_every]
    C_red_hist = C_red_hist[::save_every]
    E_hist_vis = E_hist[::save_every]
    I_hist_vis = I_hist[::save_every]
    
    return np.array(mesh.x), np.array(C_ox_hist), np.array(C_red_hist), np.array(E_hist), np.array(I_hist), np.array(E_hist_vis)
