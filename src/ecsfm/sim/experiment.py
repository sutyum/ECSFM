from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ecsfm.sim.mesh import GradedMesh1D, Mesh1D
from ecsfm.sim.potential import ElectroneutralitySolver
from ecsfm.sim.timestepping import AdaptiveConfig
from ecsfm.sim.transport import NernstPlanck1D


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
    grading_factor: float = 0.0,
    mesh: Mesh1D | GradedMesh1D | None = None,
    z: jax.Array | None = None,
    enable_migration: bool = False,
    adaptive: bool = False,
    adaptive_config: AdaptiveConfig | None = None,
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

    if mesh is not None:
        _mesh = mesh
    elif grading_factor > 0:
        _mesh = GradedMesh1D(x_min=0.0, x_max=L, n_points=nx, grading_factor=grading_factor, dtype=dtype)
    else:
        _mesh = Mesh1D(x_min=0.0, x_max=L, n_points=nx, dtype=dtype)

    dx = jnp.asarray(_mesh.dx, dtype=dtype)
    is_graded = hasattr(_mesh, 'dx_array')

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

    if is_graded:
        h = _mesh.dx_array.astype(dtype)  # (nx-1,)

        def build_dense_matrix(D):
            h_left = h[:-1]    # h_{i-1} for i=1..nx-2
            h_right = h[1:]    # h_i for i=1..nx-2

            a_int = two * D * dt_arr / (h_left * (h_left + h_right))
            c_int = two * D * dt_arr / (h_right * (h_left + h_right))
            b_int = jnp.asarray(1.0, dtype=dtype) + a_int + c_int

            main_diag = jnp.ones(nx, dtype=dtype)
            main_diag = main_diag.at[1:-1].set(b_int)

            lower_diag = jnp.zeros(nx - 1, dtype=dtype)
            lower_diag = lower_diag.at[:len(a_int)].set(-a_int)

            upper_diag = jnp.zeros(nx - 1, dtype=dtype)
            upper_diag = upper_diag.at[1:].set(-c_int)

            return jnp.diag(main_diag) + jnp.diag(upper_diag, k=1) + jnp.diag(lower_diag, k=-1)
    else:
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

    # ------------------------------------------------------------------
    # Migration setup (Nernst-Planck + electroneutrality)
    # ------------------------------------------------------------------
    if enable_migration:
        if z is None:
            raise ValueError("z (charge numbers) must be provided when enable_migration=True")
        z_arr = jnp.asarray(z, dtype=dtype)
        _validate_vector("z", z_arr)
        # z must supply charges for ALL species (ox + red interleaved: N_ox, N_red)
        if z_arr.shape[0] != 2 * N:
            raise ValueError(
                f"z must have 2*N = {2*N} entries (N_ox charges then N_red charges), "
                f"got {z_arr.shape[0]}"
            )
        z_ox = z_arr[:N]
        z_red = z_arr[N:]

        # Build transport helpers  (all species stacked: ox then red)
        D_all = jnp.concatenate([D_ox, D_red])         # (2N,)
        z_all = jnp.concatenate([z_ox, z_red])          # (2N,)
        np_solver = NernstPlanck1D(mesh=_mesh, D=D_all, z=z_all)
        en_solver = ElectroneutralitySolver(mesh=_mesh, z=z_all, D=D_all)

    @jax.jit
    def step_fn(carry, i):
        C_ox_state, C_red_state, C_ox_samples, C_red_samples = carry
        t = i.astype(dtype) * dt_arr
        E_t = jnp.interp(t, E_times, E_array)

        # ---- Step 1: Surface kinetics (Butler-Volmer flux) ----
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

        # ---- Steps 2-3 (migration, only when enabled) ----
        if enable_migration:
            # Stack concentrations: (2N, nx)
            C_all = jnp.concatenate([d_ox, d_red], axis=0)

            # Step 2: compute phi(x) from electroneutrality
            phi = en_solver.solve_phi(C_all)

            # Step 3: explicit migration source -> C* = C^n + dt * (-div(J_mig))
            mig_src = np_solver.migration_source(C_all, phi)   # (2N, nx)
            C_all_star = C_all + dt_arr * mig_src
            C_all_star = jax.nn.relu(C_all_star)               # keep non-negative

            # Pin boundary values after migration
            d_ox = C_all_star[:N]
            d_ox = d_ox.at[:, 0].set(C_ox_surf)
            d_ox = d_ox.at[:, -1].set(C_bulk_ox_mol)
            d_red = C_all_star[N:]
            d_red = d_red.at[:, 0].set(C_red_surf)
            d_red = d_red.at[:, -1].set(C_bulk_red_mol)

        # ---- Step 4: Implicit diffusion solve ----
        C_ox_next = vmap_solve(M_ox, d_ox)
        C_red_next = vmap_solve(M_red, d_red)

        # ---- Step 5: Non-negativity + conservation rescaling ----
        C_ox_next = jax.nn.relu(C_ox_next)
        C_red_next = jax.nn.relu(C_red_next)

        # Mass conservation: enforce C_ox + C_red = C_bulk_ox + C_bulk_red for interior points
        C_total_bulk = C_bulk_ox_mol + C_bulk_red_mol  # (N,)
        C_total = C_ox_next[:, 1:-1] + C_red_next[:, 1:-1]  # (N, nx-2)
        scale = C_total_bulk[:, None] / jnp.maximum(C_total, jnp.asarray(1e-30, dtype=dtype))
        C_ox_next = C_ox_next.at[:, 1:-1].set(C_ox_next[:, 1:-1] * scale)
        C_red_next = C_red_next.at[:, 1:-1].set(C_red_next[:, 1:-1] * scale)

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

    # ------------------------------------------------------------------ #
    # Fixed-step path (default) -- original jax.lax.scan                  #
    # ------------------------------------------------------------------ #
    if not adaptive:
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
            np.array(_mesh.x),
            np.array(C_ox_hist),
            np.array(C_red_hist),
            np.array(E_hist),
            np.array(I_hist),
            np.array(E_hist_vis),
            np.array(I_hist_vis),
        )

    # ------------------------------------------------------------------ #
    # Adaptive time-stepping path -- jax.lax.while_loop                   #
    # ------------------------------------------------------------------ #
    cfg = adaptive_config if adaptive_config is not None else AdaptiveConfig()

    t_max_jnp = jnp.asarray(float(t_max), dtype=dtype)
    dt_min_jnp = jnp.asarray(cfg.dt_min, dtype=dtype)
    dt_max_jnp = jnp.asarray(cfg.dt_max, dtype=dtype)
    atol_jnp = jnp.asarray(cfg.atol, dtype=dtype)
    rtol_jnp = jnp.asarray(cfg.rtol, dtype=dtype)
    safety_jnp = jnp.asarray(cfg.safety_factor, dtype=dtype)
    max_growth_jnp = jnp.asarray(cfg.max_growth, dtype=dtype)
    min_shrink_jnp = jnp.asarray(cfg.min_shrink, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)

    # Build implicit-diffusion matrix for an *arbitrary* dt_sub.
    if is_graded:
        h_arr = _mesh.dx_array.astype(dtype)

        def _build_matrix_dt(D_val, dt_sub):
            h_left = h_arr[:-1]
            h_right = h_arr[1:]
            a_int = two * D_val * dt_sub / (h_left * (h_left + h_right))
            c_int = two * D_val * dt_sub / (h_right * (h_left + h_right))
            b_int = one + a_int + c_int
            main_diag = jnp.ones(nx, dtype=dtype).at[1:-1].set(b_int)
            lower_diag = jnp.zeros(nx - 1, dtype=dtype).at[:a_int.shape[0]].set(-a_int)
            upper_diag = jnp.zeros(nx - 1, dtype=dtype).at[1:].set(-c_int)
            return (
                jnp.diag(main_diag)
                + jnp.diag(upper_diag, k=1)
                + jnp.diag(lower_diag, k=-1)
            )
    else:
        def _build_matrix_dt(D_val, dt_sub):
            r = D_val * dt_sub / (dx ** 2)
            main_diag = jnp.full((nx,), one + two * r, dtype=dtype)
            upper_diag = jnp.full((nx - 1,), -r, dtype=dtype)
            lower_diag = jnp.full((nx - 1,), -r, dtype=dtype)
            main_diag = main_diag.at[0].set(one).at[-1].set(one)
            upper_diag = upper_diag.at[0].set(jnp.asarray(0.0, dtype=dtype))
            lower_diag = lower_diag.at[-1].set(jnp.asarray(0.0, dtype=dtype))
            return (
                jnp.diag(main_diag)
                + jnp.diag(upper_diag, k=1)
                + jnp.diag(lower_diag, k=-1)
            )

    def _build_all_matrices(dt_sub):
        build_v = jax.vmap(lambda D: _build_matrix_dt(D, dt_sub))
        return build_v(D_ox), build_v(D_red)

    def _take_one_step(C_ox_state, C_red_state, t_now, dt_sub):
        """One explicit-surface + implicit-diffusion step of size dt_sub."""
        E_t = jnp.interp(t_now, E_times, E_array)

        k_red = k0 * jnp.exp(-alpha * f_constant * (E_t - E0))
        k_ox = k0 * jnp.exp((one - alpha) * f_constant * (E_t - E0))

        flux = k_ox * C_red_state[:, 0] - k_red * C_ox_state[:, 0]

        max_ox_flux = (C_ox_state[:, 0] * dx) / dt_sub
        max_red_flux = (C_red_state[:, 0] * dx) / dt_sub
        flux = jnp.clip(flux, -max_ox_flux, max_red_flux)

        rate_ox_0 = D_ox * (C_ox_state[:, 1] - C_ox_state[:, 0]) / (dx ** 2) + flux / dx
        rate_red_0 = D_red * (C_red_state[:, 1] - C_red_state[:, 0]) / (dx ** 2) - flux / dx

        C_ox_surf = C_ox_state[:, 0] + dt_sub * rate_ox_0
        C_red_surf = C_red_state[:, 0] + dt_sub * rate_red_0

        d_ox = C_ox_state.at[:, 0].set(C_ox_surf).at[:, -1].set(C_bulk_ox_mol)
        d_red = C_red_state.at[:, 0].set(C_red_surf).at[:, -1].set(C_bulk_red_mol)

        M_ox_sub, M_red_sub = _build_all_matrices(dt_sub)
        C_ox_next = vmap_solve(M_ox_sub, d_ox)
        C_red_next = vmap_solve(M_red_sub, d_red)

        C_ox_next = jax.nn.relu(C_ox_next)
        C_red_next = jax.nn.relu(C_red_next)

        # Mass conservation
        C_total_bulk = C_bulk_ox_mol + C_bulk_red_mol
        C_total = C_ox_next[:, 1:-1] + C_red_next[:, 1:-1]
        scale = C_total_bulk[:, None] / jnp.maximum(
            C_total, jnp.asarray(1e-30, dtype=dtype)
        )
        C_ox_next = C_ox_next.at[:, 1:-1].set(C_ox_next[:, 1:-1] * scale)
        C_red_next = C_red_next.at[:, 1:-1].set(C_red_next[:, 1:-1] * scale)

        I_species = F * flux * to_mA
        I_t = jnp.sum(I_species)

        return C_ox_next, C_red_next, E_t, I_t

    # Save interval in simulation-time units
    save_dt = jnp.asarray(float(save_every) * dt_f, dtype=dtype)

    # Safety cap on iterations to prevent infinite loops inside JIT
    max_while_iters = n_steps * 10

    dt_init = jnp.clip(dt_arr, dt_min_jnp, dt_max_jnp)
    E_hist_buf = jnp.zeros((n_saved,), dtype=dtype)
    I_hist_buf = jnp.zeros((n_saved,), dtype=dtype)
    dt_hist_buf = jnp.zeros((n_saved,), dtype=dtype)

    init_state = (
        jnp.asarray(0.0, dtype=dtype),        # 0: t
        dt_init,                                # 1: dt_cur
        C_ox_init.copy(),                       # 2: C_ox
        C_red_init.copy(),                      # 3: C_red
        jnp.asarray(0, dtype=jnp.int32),       # 4: step_count
        jnp.asarray(0, dtype=jnp.int32),       # 5: save_count
        C_ox_samples_init,                      # 6: C_ox_samples
        C_red_samples_init,                     # 7: C_red_samples
        E_hist_buf,                             # 8: E_hist_buf
        I_hist_buf,                             # 9: I_hist_buf
        dt_hist_buf,                            # 10: dt_hist_buf
        save_dt,                                # 11: next_save_t
    )

    def _cond_fn(state):
        t = state[0]
        step_count = state[4]
        return (t < t_max_jnp) & (step_count < max_while_iters)

    def _body_fn(state):
        (t, dt_cur, C_ox_s, C_red_s, step_count, save_count,
         C_ox_samp, C_red_samp, E_buf, I_buf, dt_buf, next_save_t) = state

        # Clamp dt so we do not overshoot t_max
        dt_cur = jnp.minimum(dt_cur, t_max_jnp - t)
        dt_cur = jnp.clip(dt_cur, dt_min_jnp, dt_max_jnp)

        # Full step of size dt_cur
        C_ox_full, C_red_full, _, _ = _take_one_step(
            C_ox_s, C_red_s, t, dt_cur
        )

        # Two half-steps of size dt_cur / 2
        dt_half = dt_cur * half
        C_ox_h1, C_red_h1, _, _ = _take_one_step(
            C_ox_s, C_red_s, t, dt_half
        )
        C_ox_h2, C_red_h2, _, _ = _take_one_step(
            C_ox_h1, C_red_h1, t + dt_half, dt_half
        )

        # Richardson error estimate
        err_ox = jnp.max(jnp.abs(C_ox_full - C_ox_h2))
        err_red = jnp.max(jnp.abs(C_red_full - C_red_h2))
        err = jnp.maximum(err_ox, err_red)

        sol_scale = jnp.maximum(
            jnp.max(jnp.abs(C_ox_h2)),
            jnp.max(jnp.abs(C_red_h2)),
        )
        tol = atol_jnp + rtol_jnp * sol_scale

        accept = err <= tol

        ratio = tol / jnp.maximum(err, jnp.asarray(1e-30, dtype=dtype))
        factor = safety_jnp * jnp.sqrt(ratio)

        dt_new_accept = dt_cur * jnp.minimum(factor, max_growth_jnp)
        dt_new_reject = dt_cur * jnp.maximum(factor, min_shrink_jnp)
        dt_new = jnp.where(accept, dt_new_accept, dt_new_reject)
        dt_new = jnp.clip(dt_new, dt_min_jnp, dt_max_jnp)

        # Use the more accurate two-half-step result on accept
        C_ox_next = jnp.where(accept, C_ox_h2, C_ox_s)
        C_red_next = jnp.where(accept, C_red_h2, C_red_s)
        t_next = jnp.where(accept, t + dt_cur, t)
        step_next = jnp.where(accept, step_count + 1, step_count)

        # Compute current at the accepted state (with same flux clamping
        # as _take_one_step to avoid unphysical values).
        E_t = jnp.interp(t, E_times, E_array)
        k_red_s = k0 * jnp.exp(-alpha * f_constant * (E_t - E0))
        k_ox_s = k0 * jnp.exp((one - alpha) * f_constant * (E_t - E0))
        flux_save = k_ox_s * C_red_s[:, 0] - k_red_s * C_ox_s[:, 0]
        max_ox_flux_s = (C_ox_s[:, 0] * dx) / jnp.maximum(dt_cur, dt_min_jnp)
        max_red_flux_s = (C_red_s[:, 0] * dx) / jnp.maximum(dt_cur, dt_min_jnp)
        flux_save = jnp.clip(flux_save, -max_ox_flux_s, max_red_flux_s)
        I_t = jnp.sum(F * flux_save * to_mA)

        # Save logic: store when we cross next_save_t
        should_save = accept & (t_next >= next_save_t) & (save_count < n_saved)
        safe_idx = jnp.minimum(save_count, n_saved - 1)

        C_ox_samp = jnp.where(
            should_save,
            C_ox_samp.at[safe_idx].set(C_ox_next),
            C_ox_samp,
        )
        C_red_samp = jnp.where(
            should_save,
            C_red_samp.at[safe_idx].set(C_red_next),
            C_red_samp,
        )
        E_buf = jnp.where(
            should_save,
            E_buf.at[safe_idx].set(E_t),
            E_buf,
        )
        I_buf = jnp.where(
            should_save,
            I_buf.at[safe_idx].set(I_t),
            I_buf,
        )
        dt_buf = jnp.where(
            should_save,
            dt_buf.at[safe_idx].set(dt_cur),
            dt_buf,
        )
        save_count = jnp.where(should_save, save_count + 1, save_count)
        next_save_t = jnp.where(should_save, next_save_t + save_dt, next_save_t)

        return (
            t_next, dt_new, C_ox_next, C_red_next, step_next, save_count,
            C_ox_samp, C_red_samp, E_buf, I_buf, dt_buf, next_save_t,
        )

    final = jax.lax.while_loop(_cond_fn, _body_fn, init_state)
    (_, _, _, _, _, _,
     C_ox_hist_ad, C_red_hist_ad, E_hist_ad, I_hist_ad, dt_hist_ad, _) = final

    C_ox_hist_ad = C_ox_hist_ad * jnp.asarray(1e6, dtype=dtype)
    C_red_hist_ad = C_red_hist_ad * jnp.asarray(1e6, dtype=dtype)

    return (
        np.array(_mesh.x),
        np.array(C_ox_hist_ad),
        np.array(C_red_hist_ad),
        np.array(E_hist_ad),
        np.array(I_hist_ad),
        np.array(E_hist_ad),   # E_hist_vis = E_hist for adaptive
        np.array(I_hist_ad),   # I_hist_vis = I_hist for adaptive
        np.array(dt_hist_ad),  # extra: dt history for diagnostics
    )
