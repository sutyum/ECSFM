import jax
import jax.numpy as jnp


def apply_sensor_model(
    t: jnp.ndarray,
    E_app: jnp.ndarray,
    I_f_mA: jnp.ndarray,
    Cdl: float = 1e-5,
    Ru: float = 100.0,
    noise_std_mA: float = 0.0,
    key: jax.Array | None = None,
):
    """Applies a first-order Ru-Cdl potentiostat/sensor response model.

    Returns:
        tuple: (E_real, I_total_mA, I_cap_mA)
    """
    if Cdl <= 0:
        raise ValueError(f"Cdl must be positive, got {Cdl}")
    if Ru < 0:
        raise ValueError(f"Ru must be non-negative, got {Ru}")
    if noise_std_mA < 0:
        raise ValueError(f"noise_std_mA must be non-negative, got {noise_std_mA}")

    t = jnp.asarray(t)
    E_app = jnp.asarray(E_app)
    I_f_mA = jnp.asarray(I_f_mA)

    if t.ndim != 1 or E_app.ndim != 1 or I_f_mA.ndim != 1:
        raise ValueError(
            f"t, E_app, and I_f_mA must all be 1D. Got {t.shape}, {E_app.shape}, {I_f_mA.shape}."
        )
    if not (t.shape[0] == E_app.shape[0] == I_f_mA.shape[0]):
        raise ValueError(
            f"t, E_app, and I_f_mA must have matching lengths. Got {t.shape[0]}, {E_app.shape[0]}, {I_f_mA.shape[0]}."
        )
    if t.shape[0] < 2:
        raise ValueError(f"Need at least 2 time points, got {t.shape[0]}")

    dts = jnp.diff(t)
    if bool(jnp.any(dts <= 0)):
        raise ValueError("t must be strictly increasing.")

    dtype = jnp.result_type(t, E_app, I_f_mA, jnp.asarray(Cdl), jnp.asarray(Ru))
    dtype = jnp.promote_types(dtype, jnp.float32)

    t = t.astype(dtype)
    E_app = E_app.astype(dtype)
    I_f_mA = I_f_mA.astype(dtype)
    Cdl = jnp.asarray(Cdl, dtype=dtype)
    Ru = jnp.asarray(Ru, dtype=dtype)
    tau = Ru * Cdl

    to_A = jnp.asarray(1000.0, dtype=dtype)
    to_mA = jnp.asarray(1000.0, dtype=dtype)
    eps_dt = jnp.asarray(1e-12, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)

    # Convert Faradaic current to Amperes for physics equations.
    I_f_A = I_f_mA / to_A

    def step_fn(carry, x):
        E_real_prev, E_app_prev, I_f_A_prev = carry
        dt, E_app_curr, I_f_A_curr = x

        V_eff_prev = E_app_prev - I_f_A_prev * Ru
        V_eff_curr = E_app_curr - I_f_A_curr * Ru

        term = tau / jnp.maximum(dt, eps_dt)
        E_real_curr = (
            E_real_prev * (term - half) + half * (V_eff_curr + V_eff_prev)
        ) / (term + half)

        dE_real = E_real_curr - E_real_prev
        I_cap_A = Cdl * dE_real / jnp.maximum(dt, eps_dt)
        I_total_A = I_cap_A + I_f_A_curr

        new_carry = (E_real_curr, E_app_curr, I_f_A_curr)
        output = (E_real_curr, I_total_A, I_cap_A)
        return new_carry, output

    V_eff_0 = E_app[0] - I_f_A[0] * Ru
    E_real_init = V_eff_0
    I_cap_init_A = jnp.asarray(0.0, dtype=dtype)
    I_total_init_A = I_f_A[0]

    carry_init = (E_real_init, E_app[0], I_f_A[0])

    E_app_seq = E_app[1:]
    I_f_A_seq = I_f_A[1:]

    _, (E_real_rest, I_total_rest_A, I_cap_rest_A) = jax.lax.scan(
        step_fn,
        carry_init,
        (dts.astype(dtype), E_app_seq, I_f_A_seq),
    )

    E_real = jnp.concatenate([jnp.array([E_real_init], dtype=dtype), E_real_rest])
    I_total_A = jnp.concatenate([jnp.array([I_total_init_A], dtype=dtype), I_total_rest_A])
    I_cap_A = jnp.concatenate([jnp.array([I_cap_init_A], dtype=dtype), I_cap_rest_A])

    I_total_mA = I_total_A * to_mA
    I_cap_mA = I_cap_A * to_mA

    if noise_std_mA > 0.0:
        if key is None:
            raise ValueError("A PRNG key is required when noise_std_mA > 0.")
        noise_std = jnp.asarray(noise_std_mA, dtype=dtype)
        noise = jax.random.normal(key, I_total_mA.shape, dtype=dtype) * noise_std
        I_total_mA = I_total_mA + noise

    return E_real, I_total_mA, I_cap_mA
