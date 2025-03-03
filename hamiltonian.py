# hamiltonian.py
import numpy as np
from physicalconstants import PhysicalConstants as PC
import scipy.linalg as la
from scipy.integrate import solve_ivp

class Hamiltonian:
    """Class for constructing and manipulating Hamiltonians of superconducting quantum circuits."""
    
    def __init__(self, circuit):
        """
        Initialize Hamiltonian for a quantum circuit.
        
        Args:
            circuit: SQCircuit instance
        """
        self.circuit = circuit
        self.hilbert_dim = circuit.hilbert_dim
        self.total_dim = self.hilbert_dim ** circuit.qubits_num
        
        # Store the identity operators for convenience
        self.identities = {}
        for qe in circuit.quantum_elements:
            self.identities[qe.name] = qe.Operator.I
    
    def get_bare_hamiltonian(self, t=None):
        """
        Construct the bare Hamiltonian without couplings.
        
        Args:
            t: Time point (for time-dependent parameters)
            
        Returns:
            np.array: Bare Hamiltonian matrix
        """
        H_bare = 0
        
        # Add transmon terms
        for qubit in self.circuit.qubits:
            # Get operators
            n_op = qubit.Operator.n_operator(t)
            n_op_sq = np.matmul(n_op, n_op)
            
            # Get parameters
            Ec = qubit['Ec']
            Ej = qubit['Ej'](t)
            
            # Transmon Hamiltonian: 4Ec(n-ng)² - Ej*cos(phi)
            # For simplicity, we use second order expansion of cos(phi)
            H_qubit = 4 * Ec * n_op_sq - Ej * (
                self.identities[qubit.name] - 
                qubit.Operator.phi_operator(t)**2 / 2
            )
            
            # Extend to the multi-qubit space
            H_term = H_qubit
            for other_qubit in self.circuit.qubits:
                if other_qubit.name != qubit.name:
                    H_term = np.kron(H_term, self.identities[other_qubit.name])
            
            H_bare += H_term
        
        # Add coupler terms
        for coupler in self.circuit.couplers:
            # Get operators
            n_op = coupler.Operator.n_operator(t)
            n_op_sq = np.matmul(n_op, n_op)
            
            # Get parameters
            Ec = coupler['Ec']
            Ej = coupler['Ej'](t)
            
            # Coupler Hamiltonian (same form as transmon)
            H_coupler = 4 * Ec * n_op_sq - Ej * (
                self.identities[coupler.name] - 
                coupler.Operator.phi_operator(t)**2 / 2
            )
            
            # Extend to the multi-qubit space
            H_term = H_coupler
            for qubit in self.circuit.qubits:
                H_term = np.kron(H_term, self.identities[qubit.name])
            
            H_bare += H_term
        
        return H_bare
    
    def get_coupling_hamiltonian(self, t=None):
        """
        Construct the coupling Hamiltonian between qubits and couplers.
        
        Args:
            t: Time point (for time-dependent parameters)
            
        Returns:
            np.array: Coupling Hamiltonian matrix
        """
        H_coupling = 0
        
        # Process each coupler
        for coupler in self.circuit.couplers:
            if not coupler['connected_qubits']:
                continue
                
            # Get connected qubits
            qubit1_name = coupler['connected_qubits'][0]
            qubit2_name = coupler['connected_qubits'][1]
            qubit1 = self.circuit[qubit1_name]
            qubit2 = self.circuit[qubit2_name]
            
            # Calculate coupling strength (proportional to sqrt(Ej*Ec))
            Ej_coupler = coupler['Ej'](t)
            Ec_coupler = coupler['Ec']
            g_factor = np.sqrt(Ej_coupler * Ec_coupler) / PC.h
            
            # Coupling term between qubits via coupler: g * (a1† + a1) * (a2† + a2)
            # Use phi operators which are proportional to (a† + a)
            phi1 = qubit1.Operator.phi_operator(t)
            phi2 = qubit2.Operator.phi_operator(t)
            
            # Construct coupling term in the full Hilbert space
            coupling_term = g_factor * np.kron(phi1, phi2)
            
            # Extend to the multi-qubit space
            for qubit in self.circuit.qubits:
                if qubit.name not in [qubit1.name, qubit2.name]:
                    coupling_term = np.kron(coupling_term, self.identities[qubit.name])
            
            H_coupling += coupling_term
        
        return H_coupling
    
    def get_drive_hamiltonian(self, t):
        """
        Construct the drive Hamiltonian based on control signals.
        
        Args:
            t: Time point
            
        Returns:
            np.array: Drive Hamiltonian matrix
        """
        H_drive = 0
        
        # Process each qubit
        for qubit in self.circuit.qubits:
            # X drive
            if qubit['Signal_x'] and callable(qubit['Signal_x'].sequence):
                amp_x = qubit['Signal_x'].sequence(t)
                if amp_x != 0:
                    # Use x_operator from TransmonOperator
                    x_op = qubit.Operator.x_operator()
                    
                    # Extend to the multi-qubit space
                    H_term = amp_x * x_op
                    for other_qubit in self.circuit.qubits:
                        if other_qubit.name != qubit.name:
                            H_term = np.kron(H_term, self.identities[other_qubit.name])
                    
                    H_drive += H_term
            
            # Y drive
            if qubit['Signal_y'] and callable(qubit['Signal_y'].sequence):
                amp_y = qubit['Signal_y'].sequence(t)
                if amp_y != 0:
                    # Use y_operator from TransmonOperator
                    y_op = qubit.Operator.y_operator()
                    
                    # Extend to the multi-qubit space
                    H_term = amp_y * y_op
                    for other_qubit in self.circuit.qubits:
                        if other_qubit.name != qubit.name:
                            H_term = np.kron(H_term, self.identities[other_qubit.name])
                    
                    H_drive += H_term
            
            # Z drive
            if qubit['Signal_z'] and callable(qubit['Signal_z'].sequence):
                amp_z = qubit['Signal_z'].sequence(t)
                if amp_z != 0:
                    # Use z_operator from TransmonOperator
                    z_op = qubit.Operator.z_operator()
                    
                    # Extend to the multi-qubit space
                    H_term = amp_z * z_op
                    for other_qubit in self.circuit.qubits:
                        if other_qubit.name != qubit.name:
                            H_term = np.kron(H_term, self.identities[other_qubit.name])
                    
                    H_drive += H_term
        
        return H_drive
    
    def get_total_hamiltonian(self, t=None):
        """
        Construct the total Hamiltonian.
        
        Args:
            t: Time point (for time-dependent parameters)
            
        Returns:
            np.array: Total Hamiltonian matrix
        """
        H_bare = self.get_bare_hamiltonian(t)
        H_coupling = self.get_coupling_hamiltonian(t)
        
        H_total = H_bare + H_coupling
        
        # Add drive terms if time is provided
        if t is not None:
            H_drive = self.get_drive_hamiltonian(t)
            H_total += H_drive
        
        return H_total
    
    def evolve_state(self, initial_state, tspan, num_steps=100, method='RK45'):
        """
        Evolve a quantum state under the time-dependent Hamiltonian.
        
        Args:
            initial_state: Initial quantum state vector
            tspan: Time span for evolution (t_start, t_end)
            num_steps: Number of time steps
            method: ODE solver method
            
        Returns:
            times: Array of time points
            states: Array of state vectors at each time
        """
        # Time points for evaluation
        times = np.linspace(tspan[0], tspan[1], num_steps)
        
        # Define the Schrödinger equation for the ODE solver
        def schrodinger_eq(t, psi):
            # Get Hamiltonian at time t
            H = self.get_total_hamiltonian(t)
            
            # Calculate -i/ħ * H * psi
            dpsi = -1j * H.dot(psi) / PC.H
            
            return dpsi
        
        # Solve the ODE
        solution = solve_ivp(
            schrodinger_eq,
            tspan,
            initial_state,
            method=method,
            t_eval=times,
            rtol=1e-10,
            atol=1e-10
        )
        
        return solution.t, solution.y.T