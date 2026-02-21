import jax
import jax.numpy as jnp

def apply_sensor_model(
    t: jnp.ndarray,
    E_app: jnp.ndarray,
    I_f_mA: jnp.ndarray,
    Cdl: float = 1e-5,  # Farads
    Ru: float = 100.0,  # Ohms
    noise_std_mA: float = 0.0,
    key: jax.random.PRNGKey = None
):
    """
    Applies an RC sensor model capturing EIS response to Faradaic currents and applied potentials.
    
    This simulates the uncompensated resistance (Ru) and double-layer capacitance (Cdl) 
    found in real electrochemical cells. It calculates the RC filtered real potential 
    acting on the electrode, and calculates the total current including capacitive charging.
    
    Args:
        t: Time array (seconds).
        E_app: Applied potential array (Volts).
        I_f_mA: Faradaic current array (mA).
        Cdl: Double-layer capacitance (Farads).
        Ru: Uncompensated resistance (Ohms).
        noise_std_mA: Standard deviation of Gaussian measurement noise (mA).
        key: JAX PRNGKey for noise generation.
        
    Returns:
        tuple: (E_real, I_total_mA, I_cap_mA)
               E_real: The actual RC-filtered potential at the interface (V).
               I_total_mA: The total measured current (mA).
               I_cap_mA: The capacitive current (mA).
    """
    tau = Ru * Cdl
    
    # Convert Faradaic current to Amperes for physics equations
    I_f_A = I_f_mA / 1000.0
    
    def step_fn(carry, x):
        E_real_prev, E_app_prev, I_f_A_prev = carry
        dt, E_app_curr, I_f_A_curr = x
        
        # V_eff = E_app - I_f * Ru
        V_eff_prev = E_app_prev - I_f_A_prev * Ru
        V_eff_curr = E_app_curr - I_f_A_curr * Ru
        
        # Discretization of the ODE: tau * dE/dt + E_real = V_eff
        # using Trapezoidal Rule (Crank-Nicolson)
        term = tau / jnp.maximum(dt, 1e-12)
        
        E_real_curr = (E_real_prev * (term - 0.5) + 0.5 * (V_eff_curr + V_eff_prev)) / (term + 0.5)
        
        # dE/dt across the step
        dE_real = E_real_curr - E_real_prev
        I_cap_A = Cdl * dE_real / jnp.maximum(dt, 1e-12)
        
        I_total_A = I_cap_A + I_f_A_curr
        
        new_carry = (E_real_curr, E_app_curr, I_f_A_curr)
        output = (E_real_curr, I_total_A, I_cap_A)
        return new_carry, output

    # Assume steady state initially
    V_eff_0 = E_app[0] - I_f_A[0] * Ru
    E_real_init = V_eff_0
    I_cap_init_A = 0.0
    I_total_init_A = I_f_A[0]
    
    carry_init = (E_real_init, E_app[0], I_f_A[0])
    
    # We scan starting from the second step
    dts = jnp.diff(t)
    E_app_seq = E_app[1:]
    I_f_A_seq = I_f_A[1:]
    
    _, (E_real_rest, I_total_rest_A, I_cap_rest_A) = jax.lax.scan(
        step_fn,
        carry_init,
        (dts, E_app_seq, I_f_A_seq)
    )
    
    # Concatenate initial state
    E_real = jnp.concatenate([jnp.array([E_real_init]), E_real_rest])
    I_total_A = jnp.concatenate([jnp.array([I_total_init_A]), I_total_rest_A])
    I_cap_A = jnp.concatenate([jnp.array([I_cap_init_A]), I_cap_rest_A])
    
    # Convert Amperes back to mA
    I_total_mA = I_total_A * 1000.0
    I_cap_mA = I_cap_A * 1000.0
    
    if noise_std_mA > 0.0 and key is not None:
        noise = jax.random.normal(key, I_total_mA.shape) * noise_std_mA
        I_total_mA = I_total_mA + noise
        
    return E_real, I_total_mA, I_cap_mA
