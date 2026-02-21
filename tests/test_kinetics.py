import jax
import jax.numpy as jnp
import pytest
from ecsfm.sim.kinetics import ButlerVolmer

def test_butler_volmer_equilibrium():
    """
    At equilibrium (net current = 0), the Butler-Volmer equation 
    must reduce to the Nernst equation:
    E = E0 + (RT/nF) * ln(C_ox / C_red)
    """
    # Simple one-electron system
    E0 = 0.0
    bv = ButlerVolmer(k0=1.0, alpha=0.5, E0=E0, n=1, T=298.15)
    
    # Let's test several equilibrium situations
    test_cases = [
        # (C_red, C_ox)
        (1.0, 1.0),
        (0.1, 1.0),
        (1.0, 0.1),
        (0.5, 0.5)
    ]
    
    for C_red, C_ox in test_cases:
        # Nernst potential
        E_nernst = E0 + (bv.R * bv.T) / (bv.n * bv.F) * jnp.log(C_ox / C_red)
        
        # Calculate current density at the Nernst potential
        # It should be exactly zero
        current = bv.current_density(E_nernst, float(C_red), float(C_ox))
        
        # Float64 or precision might lead to tiny values instead of exactly 0
        assert jnp.abs(current) < 1e-6, f"Current {current} is not 0 at Nernst potential {E_nernst}"

def test_butler_volmer_driven():
    """
    Tests that applying an overpotential drives the current in the expected direction.
    """
    E0 = 0.0
    bv = ButlerVolmer(k0=1.0, alpha=0.5, E0=E0)
    
    # At C_red = C_ox = 1.0, E=0 is equilibrium
    current_eq = bv.current_density(0.0, 1.0, 1.0)
    assert jnp.abs(current_eq) < 1e-6
    
    # Applying positive overpotential (E > E0) should cause net oxidation (positive current)
    current_ox = bv.current_density(0.1, 1.0, 1.0)
    assert current_ox > 0
    
    # Applying negative overpotential (E < E0) should cause net reduction (negative current)
    current_red = bv.current_density(-0.1, 1.0, 1.0)
    assert current_red < 0

def test_alpha_symmetry():
    """
    If alpha = 0.5, C_red = C_ox = 1.0, then current density should be 
    perfectly antisymmetric with respect to overpotential.
    """
    E0 = 0.0
    bv = ButlerVolmer(k0=1.0, alpha=0.5, E0=E0)
    
    I_plus = bv.current_density(0.2, 1.0, 1.0)
    I_minus = bv.current_density(-0.2, 1.0, 1.0)
    
    assert jnp.allclose(I_plus, -I_minus)
