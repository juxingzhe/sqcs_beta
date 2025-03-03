# physicalconstants.py
import numpy as np

class PhysicalConstants:
    """Physical constants used in superconducting quantum circuit simulation."""
    
    # Reduced Planck constant (J·s)
    H = 1.054571817e-34
    
    # Planck constant (J·s)
    h = 6.62607015e-34
    
    # Elementary charge (C)
    E = 1.602176634e-19
    
    # Flux quantum (Wb)
    Phi_0 = h / (2 * E)
    
    # Boltzmann constant (J/K)
    k_B = 1.380649e-23
    
    # Vacuum permittivity (F/m)
    epsilon_0 = 8.8541878128e-12
    
    # Speed of light in vacuum (m/s)
    c = 299792458
    
    # Temperature conversion
    @staticmethod
    def kelvin_to_Hz(temp_K):
        """Convert temperature from Kelvin to frequency in Hz."""
        return temp_K * PhysicalConstants.k_B / PhysicalConstants.h
    
    @staticmethod
    def thermal_photon_number(freq_Hz, temp_K):
        """Calculate thermal photon number for a mode at given frequency and temperature."""
        if temp_K == 0:
            return 0
        x = PhysicalConstants.h * freq_Hz / (PhysicalConstants.k_B * temp_K)
        return 1 / (np.exp(x) - 1)