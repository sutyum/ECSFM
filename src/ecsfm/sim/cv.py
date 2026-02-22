import jax
import jax.numpy as jnp
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
    save_every: int | None = 0,
):
    """Simulates a cyclic voltammetry experiment in 1D.

    Returns:
        tuple: (x, C_ox_hist, C_red_hist, E_hist, I_hist, E_hist_vis)
    """
    if nx < 2:
        raise ValueError(f"nx must be >= 2, got {nx}")
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    if scan_rate <= 0:
        raise ValueError(f"scan_rate must be positive, got {scan_rate}")
    if D_ox <= 0 or D_red <= 0:
        raise ValueError(f"Diffusion coefficients must be positive, got D_ox={D_ox}, D_red={D_red}")
    if k0 <= 0:
        raise ValueError(f"k0 must be positive, got {k0}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if E_start == E_vertex:
        raise ValueError("E_start and E_vertex must differ to produce a scan.")

    dtype = jnp.result_type(
        jnp.asarray(D_ox),
        jnp.asarray(D_red),
        jnp.asarray(C_bulk_ox),
        jnp.asarray(C_bulk_red),
        jnp.asarray(E0),
        jnp.asarray(k0),
        jnp.asarray(alpha),
        jnp.asarray(scan_rate),
        jnp.asarray(E_start),
        jnp.asarray(E_vertex),
        jnp.asarray(L),
    )
    dtype = jnp.promote_types(dtype, jnp.float32)

    D_ox = jnp.asarray(D_ox, dtype=dtype)
    D_red = jnp.asarray(D_red, dtype=dtype)
    C_bulk_ox = jnp.asarray(C_bulk_ox, dtype=dtype)
    C_bulk_red = jnp.asarray(C_bulk_red, dtype=dtype)
    E0 = jnp.asarray(E0, dtype=dtype)
    k0 = jnp.asarray(k0, dtype=dtype)
    alpha = jnp.asarray(alpha, dtype=dtype)
    scan_rate = jnp.asarray(scan_rate, dtype=dtype)
    E_start = jnp.asarray(E_start, dtype=dtype)
    E_vertex = jnp.asarray(E_vertex, dtype=dtype)

    to_mol_cm3 = jnp.asarray(1e-6, dtype=dtype)
    to_mA = jnp.asarray(1000.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    ten = jnp.asarray(10.0, dtype=dtype)
    tenth = jnp.asarray(0.1, dtype=dtype)

    C_bulk_ox_mol_cm3 = C_bulk_ox * to_mol_cm3
    C_bulk_red_mol_cm3 = C_bulk_red * to_mol_cm3

    mesh = Mesh1D(x_min=0.0, x_max=L, n_points=nx, dtype=dtype)
    dx = jnp.asarray(mesh.dx, dtype=dtype)

    diff_ox = Diffusion1D(mesh, float(D_ox))
    diff_red = Diffusion1D(mesh, float(D_red))
    kinetics = ButlerVolmer(k0=float(k0), alpha=float(alpha), E0=float(E0))

    max_D = jnp.maximum(D_ox, D_red)
    dt = (tenth * (dx**2) / max_D) / ten
    t_max = two * jnp.abs(E_start - E_vertex) / scan_rate

    dt_f = float(dt)
    t_max_f = float(t_max)
    if not np.isfinite(dt_f) or dt_f <= 0:
        raise ValueError(f"Computed invalid timestep dt={dt_f}")
    if not np.isfinite(t_max_f) or t_max_f <= 0:
        raise ValueError(f"Computed invalid duration t_max={t_max_f}")

    n_steps = int(t_max_f / dt_f)
    if n_steps < 1:
        raise ValueError(
            f"Simulation would run zero steps (n_steps={n_steps}). Increase t_max or reduce dt."
        )

    if save_every is None or save_every <= 0:
        save_every = max(1, n_steps // 200)

    n_saved = (n_steps + save_every - 1) // save_every
    dt_arr = jnp.asarray(dt_f, dtype=dtype)
    half_time = jnp.asarray(t_max_f / 2.0, dtype=dtype)

    @jax.jit
    def step_fn(carry, i):
        C_ox, C_red, C_ox_samples, C_red_samples = carry
        t = i.astype(dtype) * dt_arr

        E_t = jnp.where(
            t < half_time,
            E_start - scan_rate * t,
            E_vertex + scan_rate * (t - half_time),
        )

        flux = kinetics.flux(E_t, C_red[0], C_ox[0])

        max_ox_flux = (C_ox[0] * dx) / dt_arr
        max_red_flux = (C_red[0] * dx) / dt_arr
        flux = jnp.clip(flux, -max_ox_flux, max_red_flux)

        rate_ox_0 = D_ox * (C_ox[1] - C_ox[0]) / (dx**2) + flux / dx
        rate_red_0 = D_red * (C_red[1] - C_red[0]) / (dx**2) - flux / dx

        C_ox_surf = C_ox[0] + dt_arr * rate_ox_0
        C_red_surf = C_red[0] + dt_arr * rate_red_0

        C_ox_next = diff_ox.step_implicit(C_ox, dt_f, C_ox_surf, C_bulk_ox_mol_cm3)
        C_red_next = diff_red.step_implicit(C_red, dt_f, C_red_surf, C_bulk_red_mol_cm3)

        C_ox_next = jax.nn.relu(C_ox_next)
        C_red_next = jax.nn.relu(C_red_next)

        current_scale = jnp.asarray(kinetics.n * kinetics.F, dtype=dtype)
        I_t = current_scale * flux * to_mA

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

    C_ox_init = jnp.full((nx,), C_bulk_ox_mol_cm3, dtype=dtype)
    C_red_init = jnp.full((nx,), C_bulk_red_mol_cm3, dtype=dtype)
    C_ox_samples_init = jnp.zeros((n_saved, nx), dtype=dtype)
    C_red_samples_init = jnp.zeros((n_saved, nx), dtype=dtype)

    final_carry, (E_hist, I_hist) = jax.lax.scan(
        step_fn,
        (C_ox_init, C_red_init, C_ox_samples_init, C_red_samples_init),
        jnp.arange(n_steps, dtype=jnp.int32),
    )
    _, _, C_ox_hist, C_red_hist = final_carry

    C_ox_hist = C_ox_hist * jnp.asarray(1e6, dtype=dtype)
    C_red_hist = C_red_hist * jnp.asarray(1e6, dtype=dtype)
    E_hist_vis = E_hist[::save_every]

    return (
        np.array(mesh.x),
        np.array(C_ox_hist),
        np.array(C_red_hist),
        np.array(E_hist),
        np.array(I_hist),
        np.array(E_hist_vis),
    )
