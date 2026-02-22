import jax
import jax.numpy as jnp
import numpy as np

from ecsfm.sim.mesh import Mesh1D


def _validate_vector(name: str, arr: jax.Array) -> None:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")


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
    save_every: int | None = 0,
):
    """Simulates a generic 1D electrochemical experiment with N species.

    Returns:
        tuple: (x, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis, I_hist_vis)
    """
    if nx < 2:
        raise ValueError(f"nx must be >= 2, got {nx}")
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    E_array = jnp.asarray(E_array)
    D_ox = jnp.asarray(D_ox)
    D_red = jnp.asarray(D_red)
    C_bulk_ox = jnp.asarray(C_bulk_ox)
    C_bulk_red = jnp.asarray(C_bulk_red)
    E0 = jnp.asarray(E0)
    k0 = jnp.asarray(k0)
    alpha = jnp.asarray(alpha)

    _validate_vector("E_array", E_array)
    _validate_vector("D_ox", D_ox)
    _validate_vector("D_red", D_red)
    _validate_vector("C_bulk_ox", C_bulk_ox)
    _validate_vector("C_bulk_red", C_bulk_red)
    _validate_vector("E0", E0)
    _validate_vector("k0", k0)
    _validate_vector("alpha", alpha)

    if E_array.shape[0] < 2:
        raise ValueError(f"E_array must have at least 2 points, got {E_array.shape[0]}")

    N = D_ox.shape[0]
    if N < 1:
        raise ValueError("At least one species is required.")

    for name, arr in (
        ("D_red", D_red),
        ("C_bulk_ox", C_bulk_ox),
        ("C_bulk_red", C_bulk_red),
        ("E0", E0),
        ("k0", k0),
        ("alpha", alpha),
    ):
        if arr.shape[0] != N:
            raise ValueError(
                f"{name} length ({arr.shape[0]}) must match D_ox length ({N})."
            )

    if bool(jnp.any(D_ox <= 0)) or bool(jnp.any(D_red <= 0)):
        raise ValueError("All diffusion coefficients must be positive.")
    if bool(jnp.any(k0 <= 0)):
        raise ValueError("All k0 values must be positive.")
    if bool(jnp.any((alpha < 0) | (alpha > 1))):
        raise ValueError("All alpha values must lie in [0, 1].")

    dtype = jnp.result_type(
        E_array,
        D_ox,
        D_red,
        C_bulk_ox,
        C_bulk_red,
        E0,
        k0,
        alpha,
        jnp.asarray(t_max),
        jnp.asarray(L),
    )
    dtype = jnp.promote_types(dtype, jnp.float32)

    E_array = E_array.astype(dtype)
    D_ox = D_ox.astype(dtype)
    D_red = D_red.astype(dtype)
    C_bulk_ox = C_bulk_ox.astype(dtype)
    C_bulk_red = C_bulk_red.astype(dtype)
    E0 = E0.astype(dtype)
    k0 = k0.astype(dtype)
    alpha = alpha.astype(dtype)

    to_mol_cm3 = jnp.asarray(1e-6, dtype=dtype)
    to_mA = jnp.asarray(1000.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    ten = jnp.asarray(10.0, dtype=dtype)
    tenth = jnp.asarray(0.1, dtype=dtype)
    F = jnp.asarray(96485.3321, dtype=dtype)
    f_constant = jnp.asarray(96485.3321 / (8.314462618 * 298.15), dtype=dtype)

    C_bulk_ox_mol = C_bulk_ox * to_mol_cm3
    C_bulk_red_mol = C_bulk_red * to_mol_cm3

    mesh = Mesh1D(x_min=0.0, x_max=L, n_points=nx, dtype=dtype)
    dx = jnp.asarray(mesh.dx, dtype=dtype)

    max_D = jnp.max(jnp.maximum(D_ox, D_red))
    dt = (tenth * (dx**2) / max_D) / ten
    dt_f = float(dt)
    if not np.isfinite(dt_f) or dt_f <= 0:
        raise ValueError(f"Computed invalid timestep dt={dt_f}")

    n_steps = int(float(t_max) / dt_f)
    if n_steps < 1:
        raise ValueError(
            f"Simulation would run zero steps (n_steps={n_steps}). Increase t_max or reduce dt."
        )

    if save_every is None or save_every <= 0:
        save_every = max(1, n_steps // 200)

    n_saved = (n_steps + save_every - 1) // save_every

    E_times = jnp.linspace(0.0, float(t_max), E_array.shape[0], dtype=dtype)
    dt_arr = jnp.asarray(dt_f, dtype=dtype)

    def build_dense_matrix(D):
        r = D * dt_arr / (dx**2)
        main_diag = jnp.full((nx,), jnp.asarray(1.0, dtype=dtype) + two * r, dtype=dtype)
        upper_diag = jnp.full((nx - 1,), -r, dtype=dtype)
        lower_diag = jnp.full((nx - 1,), -r, dtype=dtype)

        main_diag = main_diag.at[0].set(jnp.asarray(1.0, dtype=dtype))
        main_diag = main_diag.at[-1].set(jnp.asarray(1.0, dtype=dtype))
        upper_diag = upper_diag.at[0].set(jnp.asarray(0.0, dtype=dtype))
        lower_diag = lower_diag.at[-1].set(jnp.asarray(0.0, dtype=dtype))

        return jnp.diag(main_diag) + jnp.diag(upper_diag, k=1) + jnp.diag(lower_diag, k=-1)

    vmap_matrices = jax.vmap(build_dense_matrix)
    M_ox = vmap_matrices(D_ox)
    M_red = vmap_matrices(D_red)

    vmap_solve = jax.vmap(jnp.linalg.solve, in_axes=(0, 0))

    @jax.jit
    def step_fn(carry, i):
        C_ox_state, C_red_state, C_ox_samples, C_red_samples = carry
        t = i.astype(dtype) * dt_arr
        E_t = jnp.interp(t, E_times, E_array)

        k_red = k0 * jnp.exp(-alpha * f_constant * (E_t - E0))
        k_ox = k0 * jnp.exp((jnp.asarray(1.0, dtype=dtype) - alpha) * f_constant * (E_t - E0))

        flux = k_ox * C_red_state[:, 0] - k_red * C_ox_state[:, 0]

        max_ox_flux = (C_ox_state[:, 0] * dx) / dt_arr
        max_red_flux = (C_red_state[:, 0] * dx) / dt_arr
        flux = jnp.clip(flux, -max_ox_flux, max_red_flux)

        rate_ox_0 = D_ox * (C_ox_state[:, 1] - C_ox_state[:, 0]) / (dx**2) + flux / dx
        rate_red_0 = D_red * (C_red_state[:, 1] - C_red_state[:, 0]) / (dx**2) - flux / dx

        C_ox_surf = C_ox_state[:, 0] + dt_arr * rate_ox_0
        C_red_surf = C_red_state[:, 0] + dt_arr * rate_red_0

        d_ox = C_ox_state.at[:, 0].set(C_ox_surf)
        d_ox = d_ox.at[:, -1].set(C_bulk_ox_mol)

        d_red = C_red_state.at[:, 0].set(C_red_surf)
        d_red = d_red.at[:, -1].set(C_bulk_red_mol)

        C_ox_next = vmap_solve(M_ox, d_ox)
        C_red_next = vmap_solve(M_red, d_red)

        C_ox_next = jax.nn.relu(C_ox_next)
        C_red_next = jax.nn.relu(C_red_next)

        I_species = F * flux * to_mA
        I_t = jnp.sum(I_species)

        save_idx = i // save_every
        should_save = (i % save_every) == 0

        def _store(samples):
            ox_samples, red_samples = samples
            ox_samples = ox_samples.at[save_idx].set(C_ox_next)
            red_samples = red_samples.at[save_idx].set(C_red_next)
            return ox_samples, red_samples

        C_ox_samples, C_red_samples = jax.lax.cond(
            should_save,
            _store,
            lambda samples: samples,
            (C_ox_samples, C_red_samples),
        )

        return (C_ox_next, C_red_next, C_ox_samples, C_red_samples), (E_t, I_t)

    C_ox_init = jnp.broadcast_to(C_bulk_ox_mol[:, None], (N, nx))
    C_red_init = jnp.broadcast_to(C_bulk_red_mol[:, None], (N, nx))
    C_ox_samples_init = jnp.zeros((n_saved, N, nx), dtype=dtype)
    C_red_samples_init = jnp.zeros((n_saved, N, nx), dtype=dtype)

    final_carry, (E_hist, I_hist) = jax.lax.scan(
        step_fn,
        (C_ox_init, C_red_init, C_ox_samples_init, C_red_samples_init),
        jnp.arange(n_steps, dtype=jnp.int32),
    )
    _, _, C_ox_hist, C_red_hist = final_carry

    C_ox_hist = C_ox_hist * jnp.asarray(1e6, dtype=dtype)
    C_red_hist = C_red_hist * jnp.asarray(1e6, dtype=dtype)
    E_hist_vis = E_hist[::save_every]
    I_hist_vis = I_hist[::save_every]

    return (
        np.array(mesh.x),
        np.array(C_ox_hist),
        np.array(C_red_hist),
        np.array(E_hist),
        np.array(I_hist),
        np.array(E_hist_vis),
        np.array(I_hist_vis),
    )
