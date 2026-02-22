import jax.numpy as jnp
import equinox as eqx

class ButlerVolmer(eqx.Module):
    """
    Computes heterogeneous electron transfer rates at an electrode surface 
    using the Butler-Volmer kinetic model.
    """
    k0: float    # Standard rate constant (cm/s)
    alpha: float # Charge transfer coefficient (typically 0.5)
    E0: float    # Formal potential of the redox couple (V)
    n: int = 1   # Number of electrons transferred
    T: float = 298.15 # Temperature (K)
    
    # Constants
    F: float = 96485.332 # Faraday constant (C/mol)
    R: float = 8.31446   # Universal gas constant (J/(mol*K))

    def __init__(self, k0: float, alpha: float, E0: float, n: int = 1, T: float = 298.15):
        self.k0 = k0
        self.alpha = alpha
        self.E0 = E0
        self.n = n
        self.T = T

    def rate_constants(self, E: float):
        """
        Calculates the forward (oxidation) and backward (reduction) 
        rate constants for a given applied potential E.
        """
        f = (self.n * self.F) / (self.R * self.T)
        overpotential = E - self.E0
        
        # Oxidation rate constant
        k_ox = self.k0 * jnp.exp((1 - self.alpha) * f * overpotential)
        
        # Reduction rate constant
        k_red = self.k0 * jnp.exp(-self.alpha * f * overpotential)
        
        return k_ox, k_red

    def flux(self, E: float, C_red: float, C_ox: float) -> float:
        """
        Calculates the net molar flux at the electrode surface purely from kinetics.
        Positive flux = net oxidation (current is positive).
        """
        k_ox, k_red = self.rate_constants(E)
        
        # Net flux J = k_ox * C_red - k_red * C_ox
        net_flux = k_ox * C_red - k_red * C_ox
        return net_flux
        
    def current_density(self, E: float, C_red: float, C_ox: float) -> float:
        """
        Calculates the current density (A/cm^2) for the given state.
        """
        # i = n * F * J
        J = self.flux(E, C_red, C_ox)
        return self.n * self.F * J
