# operators.py
import numpy as np
from physicalconstants import PhysicalConstants as PC

class QuantumOperator:
    """Base class for quantum operators."""
    
    def __init__(self, hilbert_dim=2):
        self.hilbert_dim = hilbert_dim
        self.I = np.eye(hilbert_dim)
        
    def tensor_product(self, *operators):
        """Compute tensor product of multiple operators."""
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        return result

class TransmonOperator(QuantumOperator):
    """Quantum operators for transmon qubits."""
    
    def __init__(self, transmon, hilbert_dim=10):
        """Initialize operators for a transmon qubit."""
        super().__init__(hilbert_dim)
        self.transmon = transmon
        self.tensor_element_truncation_dim = hilbert_dim
        
        # Create ladder operators
        self.anhilation_operator = self._create_anhilation_operator()
        self.creation_operator = self._create_creation_operator()
        
        # Number operator
        self.number_operator = np.matmul(self.creation_operator, self.anhilation_operator)
        
    def _create_anhilation_operator(self):
        """Create annihilation operator for the specified Hilbert dimension."""
        a = np.zeros((self.hilbert_dim, self.hilbert_dim))
        for i in range(1, self.hilbert_dim):
            a[i-1, i] = np.sqrt(i)
        return a
    
    def _create_creation_operator(self):
        """Create creation operator for the specified Hilbert dimension."""
        return self._create_anhilation_operator().T.conj()
    
    def n_operator(self, t=None):
        """Number operator."""
        return self.number_operator
    
    def phi_operator(self, t=None):
        """Phase operator (conjugate to number operator)."""
        a = self.anhilation_operator
        adag = self.creation_operator
        
        # phi operator is proportional to i(a† - a)
        phi = 1j * (adag - a) / np.sqrt(2)
        return phi
    
    def x_operator(self):
        """Pauli X operator in the qubit subspace."""
        x = np.zeros((self.hilbert_dim, self.hilbert_dim))
        x[0, 1] = x[1, 0] = 1
        return x
    
    def y_operator(self):
        """Pauli Y operator in the qubit subspace."""
        y = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=complex)
        y[0, 1] = -1j
        y[1, 0] = 1j
        return y
    
    def z_operator(self):
        """Pauli Z operator in the qubit subspace."""
        z = np.zeros((self.hilbert_dim, self.hilbert_dim))
        z[0, 0] = 1
        z[1, 1] = -1
        return z
    
    def get_matrix_element(self, operator, i, j):
        """Get the matrix element <i|operator|j>."""
        return operator[i, j]
    
    def project_to_computational_subspace(self, operator):
        """Project an operator into the computational subspace (|0⟩ and |1⟩)."""
        computational_subspace = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                computational_subspace[i, j] = operator[i, j]
        return computational_subspace