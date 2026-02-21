import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from ecsfm.sim.mesh import Mesh1D

def simulate_electrochem(
    E_array: jax.Array,
    t_max: float,
    D_ox: jax.Array,
    D_red: jax.Array,
    C_bulk_ox: jax.Array,
    C_bulk_red: jax.Array,
    E0: jax.Array,
    k0: jax.Array,
    alpha: jax.Array,
    L: float = 0.05,
    nx: int = 200,
    save_every: int = 1
):
    """
    Simulates a generic 1D electrochemical experiment with N independent species.
    
    Args:
        E_array: 1D array of applied potentials (V), shape (seq_len,)
        t_max: Total duration of the experiment in seconds
        D_ox: Diffusion coefficients of oxidized species (cm^2/s), shape (N,)
        D_red: Diffusion coefficients of reduced species (cm^2/s), shape (N,)
        C_bulk_ox: Bulk concentrations of oxidized species (mM), shape (N,)
        C_bulk_red: Bulk concentrations of reduced species (mM), shape (N,)
        E0: Formal reduction potentials (V), shape (N,)
        k0: Standard rate constants (cm/s), shape (N,)
        alpha: Charge transfer coefficients, shape (N,)
        L: Length of the simulation domain (cm)
        nx: Number of spatial grid points
        save_every: Save state frequency
        
    Returns:
        tuple: (x, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis, I_hist_vis)
               C_hist shape: (time, N, nx)
               I_hist shape: (time,) - this is the combined total macroscopic current
    """
    jax.config.update("jax_enable_x64", True)

    D_ox = jnp.asarray(D_ox)
    D_red = jnp.asarray(D_red)
    C_bulk_ox = jnp.asarray(C_bulk_ox)
    C_bulk_red = jnp.asarray(C_bulk_red)
    E0 = jnp.asarray(E0)
    k0 = jnp.asarray(k0)
    alpha = jnp.asarray(alpha)
    
    N = D_ox.shape[0]

    C_bulk_ox_mol = C_bulk_ox * 1e-6
    C_bulk_red_mol = C_bulk_red * 1e-6
    
    mesh = Mesh1D(x_min=0.0, x_max=L, n_points=nx)
    dx = mesh.dx
    
    max_D = jnp.max(jnp.maximum(D_ox, D_red))
    dt_max_stable = 0.1 * (dx ** 2) / max_D 
    dt = dt_max_stable / 10.0 
    
    n_steps = int(t_max / dt)
    
    if save_every is None or save_every <= 0:
        save_every = max(1, n_steps // 200)
    
    E_times = jnp.linspace(0.0, t_max, len(E_array))
    
    def build_dense_matrix(D):
        r = D * dt / (dx ** 2)
        main_diag = jnp.full(nx, 1.0 + 2.0 * r)
        upper_diag = jnp.full(nx - 1, -r)
        lower_diag = jnp.full(nx - 1, -r)
        
        main_diag = main_diag.at[0].set(1.0)
        main_diag = main_diag.at[-1].set(1.0)

        # Boundary rules override diagonals to 0 so we just enforce constants
        upper_diag = upper_diag.at[0].set(0.0)
        lower_diag = lower_diag.at[-1].set(0.0)
        
        M = jnp.diag(main_diag) + jnp.diag(upper_diag, k=1) + jnp.diag(lower_diag, k=-1)
        return M
        
    vmap_matrices = jax.vmap(build_dense_matrix)
    M_ox = vmap_matrices(D_ox) # Shape (N, nx, nx)
    M_red = vmap_matrices(D_red)
    
    # We will map the standard linear solver over the N independent species systems
    vmap_solve = jax.vmap(jnp.linalg.solve, in_axes=(0, 0))
    
    @jax.jit
    def step_fn(state, i):
        C_ox_state, C_red_state = state # Shape: (N, nx)
        t = i * dt
        E_t = jnp.interp(t, E_times, E_array)

        f_constant = 38.92 # F / (R * T) at 298.15K
        k_red = k0 * jnp.exp(-alpha * f_constant * (E_t - E0))
        k_ox = k0 * jnp.exp((1.0 - alpha) * f_constant * (E_t - E0))
        
        flux = k_ox * C_red_state[:, 0] - k_red * C_ox_state[:, 0] # (N,) array
        
        max_ox_flux = (C_ox_state[:, 0] * dx) / dt
        max_red_flux = (C_red_state[:, 0] * dx) / dt
        flux = jnp.clip(flux, -max_ox_flux, max_red_flux)
        
        rate_ox_0 = D_ox * (C_ox_state[:, 1] - C_ox_state[:, 0]) / (dx**2) + flux / dx
        rate_red_0 = D_red * (C_red_state[:, 1] - C_red_state[:, 0]) / (dx**2) - flux / dx
        
        C_ox_surf = C_ox_state[:, 0] + dt * rate_ox_0
        C_red_surf = C_red_state[:, 0] + dt * rate_red_0
        
        d_ox = C_ox_state.copy()
        d_ox = d_ox.at[:, 0].set(C_ox_surf)
        d_ox = d_ox.at[:, -1].set(C_bulk_ox_mol) # (N,) broadcasted to (N, 1) properly via JAX
        
        d_red = C_red_state.copy()
        d_red = d_red.at[:, 0].set(C_red_surf)
        d_red = d_red.at[:, -1].set(C_bulk_red_mol)
        
        C_ox_next = vmap_solve(M_ox, d_ox)
        C_red_next = vmap_solve(M_red, d_red)
        
        C_ox_next = jax.nn.relu(C_ox_next)
        C_red_next = jax.nn.relu(C_red_next)
        
        F = 96485.3321
        n_elec = 1.0 # 1 electron transfer per molecule
        I_species = n_elec * F * flux * 1000.0 # (N,), mA/cm^2
        I_t = jnp.sum(I_species) # Scalar superimposed total current
        
        new_state = (C_ox_next, C_red_next)
        history_frame = (C_ox_next, C_red_next, E_t, I_t)
        return new_state, history_frame

    # Initial concentrations
    # broadcast (N,) -> (N, nx)
    C_ox_init = jnp.broadcast_to(C_bulk_ox_mol[:, None], (N, nx))
    C_red_init = jnp.broadcast_to(C_bulk_red_mol[:, None], (N, nx))
    
    _, (C_ox_hist, C_red_hist, E_hist, I_hist) = jax.lax.scan(
        step_fn, 
        (C_ox_init, C_red_init), 
        jnp.arange(n_steps)
    )
    
    C_ox_hist = C_ox_hist[::save_every] * 1e6
    C_red_hist = C_red_hist[::save_every] * 1e6
    E_hist_vis = E_hist[::save_every]
    I_hist_vis = I_hist[::save_every]
    
    return np.array(mesh.x), np.array(C_ox_hist), np.array(C_red_hist), np.array(E_hist), np.array(I_hist), np.array(E_hist_vis), np.array(I_hist_vis)
