import jax
import jax.numpy as jnp
import numpy as np
from ecsfm.sim.sensor import apply_sensor_model

def test_sensor_cv_baseline():
    """
    Test 1: Constant scan rate (CV) should produce a constant baseline capacitive current.
    I_cap = C_dl * v
    """
    jax.config.update("jax_enable_x64", True)
    
    v = 0.1 # V/s
    t_max = 10.0
    dt = 0.001
    t = jnp.arange(0, t_max, dt)
    E_app = v * t
    I_f_mA = jnp.zeros_like(t)
    
    Cdl = 1e-5 # 10 uF
    Ru = 100.0 # 100 Ohms
    
    # Expected I_cap in mA at steady state:
    # I_cap (A) = Cdl * v = 1e-5 * 0.1 = 1e-6 A = 1e-3 mA
    expected_I_cap_mA = 1e-3
    
    E_real, I_total_mA, I_cap_mA = apply_sensor_model(t, E_app, I_f_mA, Cdl, Ru)
    
    # Check the last 10% of the scan to ensure it reached steady state
    steady_state_I_cap = I_cap_mA[-int(len(t) * 0.1):]
    
    np.testing.assert_allclose(steady_state_I_cap, expected_I_cap_mA, rtol=1e-3, atol=1e-6)
    
def test_sensor_ca_exponential():
    """
    Test 2: Step potential (CA) should produce an exponential decay in current.
    I_cap(t) = (V_step / Ru) * exp(-t / (Ru * Cdl))
    """
    jax.config.update("jax_enable_x64", True)
    
    t_max = 0.05 # 50 ms
    dt = 1e-5    # fine dt to capture fast RC
    t = jnp.arange(0, t_max, dt)
    
    # To properly simulate a step from 0 to 1 at t=0, we must set E_app[0]=0, E_app[t>0]=1
    # Actually, E_app[0] sets the initial steady state (so E_real_init = 0).
    E_app = jnp.ones_like(t)
    E_app = E_app.at[0].set(0.0)
    
    I_f_mA = jnp.zeros_like(t)
    
    Cdl = 1e-5 # 10 uF
    Ru = 100.0 # 100 Ohms
    tau = Ru * Cdl # 1 ms
    
    E_real, I_total_mA, I_cap_mA = apply_sensor_model(t, E_app, I_f_mA, Cdl, Ru)
    
    # Expected analytical solution for t > 0
    # I_cap_A(t) = (1V / 100 Ohms) * exp(-t / tau) = 0.01 * exp(-t / tau)
    # I_cap_mA(t) = 10.0 * exp(-t / tau)
    t_eval = t[1:] # t > 0
    expected_I_cap_mA = 10.0 * jnp.exp(-t_eval / tau)
    
    # Compare from index 2 to end to skip the numerical stepping artifact exactly at the step edge
    np.testing.assert_allclose(I_cap_mA[2:], expected_I_cap_mA[1:], rtol=1e-2, atol=1e-3)

def test_sensor_eis_impedance():
    """
    Test 3: Sine wave (EIS) should produce the correct impedance magnitude and phase.
    Z = Ru + 1 / (j * omega * Cdl)
    """
    jax.config.update("jax_enable_x64", True)
    
    freq = 100.0 # Hz
    omega = 2.0 * jnp.pi * freq
    t_max = 0.1 # 10 cycles
    dt = 1e-5
    t = jnp.arange(0, t_max, dt)
    
    amp = 0.01 # 10 mV
    E_app = amp * jnp.sin(omega * t)
    
    I_f_mA = jnp.zeros_like(t)
    
    Cdl = 1e-5
    Ru = 100.0
    
    E_real, I_total_mA, I_cap_mA = apply_sensor_model(t, E_app, I_f_mA, Cdl, Ru)
    
    # Theoretical impedance
    # Z = Ru - j / (omega * Cdl)
    Z_real = Ru
    Z_imag = -1.0 / (omega * Cdl)
    Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
    Z_phase = np.arctan2(Z_imag, Z_real)
    
    # Expected current I = E_app / Z
    I_mag_A = amp / Z_mag
    I_mag_mA = I_mag_A * 1000.0
    
    # Current phase leads voltage by -Z_phase
    # E_app was pure sine (phase 0). Current = amp/|Z| * sin(omega*t - Z_phase)
    expected_I_total_mA = I_mag_mA * jnp.sin(omega * t - Z_phase)
    
    # The transient from t=0 decays away. Let's check the last 5 cycles
    start_idx = int( len(t) / 2 )
    
    np.testing.assert_allclose(
        I_total_mA[start_idx:], 
        expected_I_total_mA[start_idx:], 
        rtol=1e-2, 
        atol=1e-3
    )
