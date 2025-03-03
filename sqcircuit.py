# sqcircuits.py
from transmon import Transmon, Coupler, ConstantFunction
from physicalconstants import PhysicalConstants as PC
import numpy as np
from signals import ConstantSignal
import matplotlib.pyplot as plt
from operators import TransmonOperator

class SQCircuit:
    """Superconducting Quantum Circuit class for circuit dynamics simulation."""
    
    def __init__(self, qubits_num, couplers_num, resonators_num=0, coupler_names=None, hilbert_dim=10):  
        """
        Initialize a superconducting quantum circuit with specified elements.
        
        Args:
            qubits_num: Number of qubits
            couplers_num: Number of couplers
            resonators_num: Number of resonators (default: 0)
            coupler_names: List of coupler names (if None, default naming scheme is used)
            hilbert_dim: Hilbert space dimension for transmon elements (default: 10)
        """
        self.qubits_num = qubits_num
        self.couplers_num = couplers_num
        self.resonators_num = resonators_num
        self.quantum_elements_num = qubits_num + couplers_num
        self.hilbert_dim = hilbert_dim
        
        # Generate qubits and couplers
        self.qubits = self.qubits_generator(qubits_num)
        self.couplers = self.couplers_generator(couplers_num, coupler_names)
        
        # Signals and other elements
        self.signals = []
        self.resonators = []
        self.quantum_elements = self.qubits + self.couplers
        
        # Initialize operators for each quantum element
        for qe in self.quantum_elements:
            qe.update_operator(hilbert_dim=hilbert_dim)
        
        # Generate mutual inductance and capacitance dictionaries
        self.MC_dict = self.generate_MC_dict()
        self.ML_dict = self.generate_ML_dict()
        
        # Calculate capacitive and inductive energy matrices
        self.M_Ec = self.M_Ec_generator()
        
        # Initialize quantum elements' parameters
        for index, quantum_element in enumerate(self.quantum_elements):
            # Set charging energy
            quantum_element['Ec'] = self.M_Ec[index][index]
            
            # Set callable Josephson energy
            quantum_element['Ej'] = quantum_element.Ej
    
    def qubits_generator(self, qubits_num):
        """
        Generate qubits list based on qubits_num.
        
        Args:
            qubits_num: Number of qubits to generate
            
        Returns:
            List of Transmon instances
        """
        qubit_names = [f'Q{i+1}' for i in range(qubits_num)]
        qubits = [
            Transmon(name=qubit_name, index=i) for i, qubit_name in enumerate(qubit_names)
        ]
        return qubits
    
    def couplers_generator(self, couplers_num, coupler_names=None):
        """
        Generate couplers list based on couplers_num and coupler_names.
        
        Args:
            couplers_num: Number of couplers
            coupler_names: List of coupler names (if None, default naming is used)
            
        Returns:
            List of Coupler instances
        """
        if couplers_num == 0:
            return []
            
        if coupler_names is None:
            # Default naming: linear chain with nearest-neighbor connections
            coupler_names = [f'C{self.qubits[i].name}{self.qubits[i+1].name}' 
                            for i in range(min(self.qubits_num-1, couplers_num))]
            
            # If more couplers than n-1 connections, add additional connections
            if couplers_num > self.qubits_num - 1:
                # Connect non-adjacent qubits
                idx = 0
                for i in range(self.qubits_num):
                    for j in range(i+2, self.qubits_num):
                        if len(coupler_names) < couplers_num:
                            coupler_names.append(f'C{self.qubits[i].name}{self.qubits[j].name}')
                            idx += 1
                        else:
                            break
                    if len(coupler_names) >= couplers_num:
                        break
        
        if couplers_num != len(coupler_names):
            raise ValueError(f"couplers_num ({couplers_num}) doesn't match the number of coupler names ({len(coupler_names)})")
        
        couplers = [
            Coupler(name=coupler_name, index=self.qubits_num + i) 
            for i, coupler_name in enumerate(coupler_names)
        ]
            
        return couplers
    
    def generate_MC_dict(self):
        """
        Generate the Mutual Capacitance Dictionary based on coupler names.
        
        Returns:
            Dictionary of mutual capacitance values between circuit elements
        """
        MCD = {}
        default_qqmc = 6E-16  # Default qubit-qubit mutual capacitance
        default_qcmc = 10.11E-15  # Default qubit-coupler mutual capacitance
        
        # Initialize dictionary for all quantum elements
        for qe in self.quantum_elements:
            MCD[qe.name] = {}
        
        # Set mutual capacitance values based on coupler connections
        for coupler in self.couplers:
            qubit1, qubit0 = self.split_coupler_string(coupler.name)
            
            # Set mutual capacitances between coupler and qubits
            MCD[coupler.name][qubit1] = default_qcmc
            MCD[coupler.name][qubit0] = default_qcmc
            MCD[qubit1][coupler.name] = default_qcmc
            MCD[qubit0][coupler.name] = default_qcmc

            # Set mutual capacitance between connected qubits
            MCD[qubit1][qubit0] = default_qqmc
            MCD[qubit0][qubit1] = default_qqmc
            
            # Track connected qubits for the coupler
            if 'connected_qubits' in coupler.parameters:
                coupler.parameters['connected_qubits'] = [qubit0, qubit1]
        
        return MCD
    
    def generate_ML_dict(self):
        """
        Generate the Mutual Inductance Dictionary based on coupler names.
        
        Returns:
            Dictionary of mutual inductance values between circuit elements
        """
        MLD = {}
        default_qqml = 1.0  # Default qubit-qubit mutual inductance
        default_qcml = 1.0  # Default qubit-coupler mutual inductance
        
        # Initialize dictionary for all quantum elements
        for qe in self.quantum_elements:
            MLD[qe.name] = {}
        
        # Set mutual inductance values based on coupler connections
        for coupler in self.couplers:
            qubit1, qubit0 = self.split_coupler_string(coupler.name)
            
            # Set mutual inductances between coupler and qubits
            MLD[coupler.name][qubit1] = default_qcml
            MLD[coupler.name][qubit0] = default_qcml
            MLD[qubit1][coupler.name] = default_qcml
            MLD[qubit0][coupler.name] = default_qcml

            # Set mutual inductance between connected qubits
            MLD[qubit1][qubit0] = default_qqml
            MLD[qubit0][qubit1] = default_qqml
        
        return MLD
    
    def M_Ec_generator(self):
        """
        Calculate capacitive energy matrix.

        Returns:
            np.array: Capacitor energy matrix
        """
        MCM = self._MCM_generator()
        M_Ec = np.zeros_like(MCM)
        
        # Calculate the capacitive energy matrix
        for i in range(self.quantum_elements_num):
            for j in range(self.quantum_elements_num):
                if i == j:
                    M_Ec[i][i] = np.sum(MCM[:, i])
                else:
                    M_Ec[i][j] = -MCM[i][j]
        
        # Convert to charging energy
        M_Ec = 0.5 * PC.E**2 * np.linalg.pinv(M_Ec)
        return M_Ec
        
    def _MCM_generator(self):
        """
        Calculate capacitance matrix.

        Returns:
            np.array: Capacitance matrix
        """
        MCM = np.zeros((self.quantum_elements_num, self.quantum_elements_num))
        
        # Set self-capacitance values
        for quantum_element in self.quantum_elements:
            MCM[quantum_element.index][quantum_element.index] = quantum_element['C']
        
        # Set mutual capacitance values from the dictionary
        for outer_key, inner_dict in self.MC_dict.items():
            for inner_key, inner_value in inner_dict.items():
                outer_key_index = self[outer_key].index
                inner_key_index = self[inner_key].index
                MCM[outer_key_index][inner_key_index] = inner_value
                MCM[inner_key_index][outer_key_index] = inner_value
        
        return MCM

    def M_Ej_generator(self, t):
        """
        Calculate Josephson energy matrix at time t.

        Args:
            t: Time point

        Returns:
            np.array: Josephson energy matrix
        """
        MLM_temp = PC.H**2/(4*np.pi**2*4*PC.E**2)/self._MLM_generator(t)
        M_Ej = np.zeros_like(MLM_temp)
        
        # Calculate the Josephson energy matrix
        for i in range(self.quantum_elements_num):
            for j in range(self.quantum_elements_num):
                if i == j:
                    M_Ej[i][i] = np.sum(MLM_temp[:, i])
                else:
                    M_Ej[i][j] = -MLM_temp[i][j]
        
        return M_Ej
    
    def _MLM_generator(self, t):
        """
        Calculate inductance matrix at time t.

        Args:
            t: Time point

        Returns:
            np.array: Inductance matrix
        """
        MLM = np.ones((self.quantum_elements_num, self.quantum_elements_num))
        
        # Set mutual inductance values from the dictionary
        for outer_key, inner_dict in self.ML_dict.items():
            for inner_key, inner_value in inner_dict.items():
                outer_key_index = self[outer_key].index
                inner_key_index = self[inner_key].index
                MLM[outer_key_index][inner_key_index] = inner_value
                MLM[inner_key_index][outer_key_index] = inner_value
        
        # Set self-inductance values based on Josephson junction parameters
        for quantum_element in self.quantum_elements:
            idx = quantum_element.index
            I1 = quantum_element['I1']
            I2 = quantum_element['I2']
            phase = quantum_element['Phase'].sequence(t)
            
            # Calculate effective inductance at time t
            MLM[idx][idx] = PC.Phi_0/(2*np.pi) / np.sqrt(I1**2 + I2**2 + 2*I1*I2*np.cos(2*phase))
        
        return MLM

    def __getitem__(self, key):
        """
        Get quantum element by name or index.

        Args:
            key: Quantum element name or index
            
        Returns:
            Quantum element (Transmon or Coupler)
        """
        if isinstance(key, int):
            if 0 <= key < self.quantum_elements_num:
                return self.quantum_elements[key]
            else:
                raise IndexError(f"Index {key} is out of range for circuit with {self.quantum_elements_num} elements")
        elif isinstance(key, str):
            for element in self.quantum_elements:
                if element.name == key:
                    return element
            raise KeyError(f"No quantum element named '{key}' in the circuit")
        else:
            raise TypeError(f"Key must be int or str, not {type(key)}")