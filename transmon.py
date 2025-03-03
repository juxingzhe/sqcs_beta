# transmon.py
import numpy as np
from physicalconstants import PhysicalConstants as PC
import matplotlib.pyplot as plt

class QuantumElement:
    """Base class for quantum elements in superconducting circuits."""
    
    def __init__(self, name, index=None):
        self.name = name
        self.index = index
        self.parameters = {}
        self.Operator = None
    
    def __getitem__(self, key):
        """Access element parameters."""
        return self.parameters.get(key)
    
    def __setitem__(self, key, value):
        """Set element parameters."""
        if key in self.ALLOWED_KEYS:
            self.parameters[key] = value
        else:
            raise KeyError(f"'{key}' is not an allowed parameter for {self.__class__.__name__}")
    
    def __str__(self):
        """String representation of the element."""
        formatted_str = f"{self.__class__.__name__}: {self.name}\n"
        for key, value in self.parameters.items():
            formatted_str += f"  {key}: {value}\n"
        return formatted_str

class Transmon(QuantumElement):
    """Transmon qubit class."""
    
    ALLOWED_KEYS = {
        'C',            # Capacitance (F)
        'I1',           # Critical current of JJ1 (A)
        'I2',           # Critical current of JJ2 (A)
        'Phase_bias',   # External phase bias (rad)
        'Phase',        # Phase operator
        'Ec',           # Charging energy (J)
        'Ej',           # Josephson energy (J)
        'frequency',    # Qubit frequency (Hz)
        'anharmonicity', # Anharmonicity (Hz)
        'Signal_x',     # X-direction signal
        'Signal_y',     # Y-direction signal
        'Signal_z',     # Z-direction signal
        'T1',           # Energy relaxation time (s)
        'T2',           # Dephasing time (s)
        'Temp'          # Temperature (K)
    }
    
    def __init__(self, name, index=None):
        """Initialize a Transmon qubit."""
        super().__init__(name, index)
        
        # Default parameters
        self.parameters = {
            'C': 100e-15,                  # 100 fF
            'I1': 10e-9,                   # 10 nA
            'I2': 10e-9,                   # 10 nA
            'Phase_bias': 0.0,             # No bias
            'Phase': ConstantFunction(0),  # Default zero phase
            'Signal_x': ConstantFunction(0),
            'Signal_y': ConstantFunction(0),
            'Signal_z': ConstantFunction(0),
            'T1': 100e-6,                  # 100 μs
            'T2': 50e-6,                   # 50 μs
            'Temp': 0.020                  # 20 mK
        }
        
    def Ej(self, t=None):
        """Calculate Josephson energy at time t."""
        I1 = self['I1']
        I2 = self['I2']
        phase = self['Phase'].sequence(t) if t is not None else 0
        
        # Josephson energy from the SQUID configuration
        return (PC.Phi_0 / (2 * np.pi)) * (I1 + I2) * np.sqrt(1 - 
               ((I1 - I2) / (I1 + I2)) ** 2 * np.sin(phase) ** 2)
        
    def get_frequency(self):
        """Calculate transmon frequency based on Ec and Ej."""
        if 'Ec' in self.parameters and 'Ej' in self.parameters:
            # For typical transmon regime where Ej >> Ec
            omega_p = np.sqrt(8 * self['Ec'] * self['Ej'](0)) / PC.h
            return omega_p
        else:
            return None
        
    def get_anharmonicity(self):
        """Calculate transmon anharmonicity."""
        if 'Ec' in self.parameters:
            # Anharmonicity is approximately -Ec in transmon regime
            return -self['Ec'] / PC.h
        else:
            return None
        
    def update_operator(self, hilbert_dim=10):
        """Update the quantum operators for this transmon."""
        from operators import TransmonOperator
        self.Operator = TransmonOperator(self, hilbert_dim)
        
class Coupler(Transmon):
    """Coupler element based on tunable transmon."""
    
    ALLOWED_KEYS = Transmon.ALLOWED_KEYS.union({
        'g01',       # Coupling strength (Hz)
        'connected_qubits'  # List of connected qubit names
    })
    
    def __init__(self, name, index=None):
        """Initialize a coupler element."""
        super().__init__(name, index)
        self.parameters['connected_qubits'] = []
        
    def Ej(self, t=None):
        """Calculate Josephson energy at time t, possibly with control signals."""
        # Base Josephson energy calculation
        base_ej = super().Ej(t)
        
        # Modify with z-signal for flux tuning
        if 'Signal_z' in self.parameters and t is not None:
            flux_signal = self['Signal_z'].sequence(t)
            # Assume the signal modifies the effective phase bias
            phase_mod = np.pi * flux_signal  # Normalized to Phi_0/2
            I1 = self['I1']
            I2 = self['I2']
            
            # Recalculate Ej with the modified phase
            return (PC.Phi_0 / (2 * np.pi)) * (I1 + I2) * np.sqrt(1 - 
                   ((I1 - I2) / (I1 + I2)) ** 2 * np.sin(self['Phase_bias'] + phase_mod) ** 2)
        else:
            return base_ej

class ConstantFunction:
    """A function that returns a constant value."""
    
    def __init__(self, value):
        self.value = value
        
    def sequence(self, t=None):
        """Return the constant value regardless of time."""
        return self.value