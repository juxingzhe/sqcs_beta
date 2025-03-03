# simulator.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from physicalconstants import PhysicalConstants as PC

class DensityMatrixSimulator:
    """Simulator for density matrix evolution of superconducting quantum circuits."""
    
    def __init__(self, hamiltonian):
        """
        Initialize the simulator.
        
        Args:
            hamiltonian: Hamiltonian instance for the quantum circuit
        """
        self.hamiltonian = hamiltonian
        self.circuit = hamiltonian.circuit
        
    def get_collapse_operators(self):
        """
        Generate collapse operators for Lindblad master equation.
        
        Returns:
            list: Collapse operators for relaxation and dephasing
        """
        collapse_ops = []
        
        # Process each qubit
        for qubit in self.circuit.qubits:
            # Get relaxation and dephasing rates from T1 and T2
            gamma1 = 1.0 / qubit['T1'] if qubit['T1'] > 0 else 0  # Relaxation rate
            gamma2 = 1.0 / qubit['T2'] if qubit['T2'] > 0 else 0  # Pure dephasing rate
            
            # Ensure gamma2 >= gamma1/2 (physics constraint)
            gamma_phi = max(0, gamma2 - gamma1/2)  # Pure dephasing contribution
            
            # Get operators for this qubit
            a = qubit.Operator.anhilation_operator  # Lowering operator
            n = qubit.Operator.number_operator  # Number operator
            
            # Extend to the multi-qubit space
            a_full = a
            n_full = n
            for other_qubit in self.circuit.qubits:
                if other_qubit.name != qubit.name:
                    a_full = np.kron(a_full, np.eye(self.hamiltonian.hilbert_dim))
                    n_full = np.kron(n_full, np.eye(self.hamiltonian.hilbert_dim))
            
            # Relaxation operator with rate sqrt(gamma1)
            if gamma1 > 0:
                collapse_ops.append((np.sqrt(gamma1), a_full))
            
            # Dephasing operator with rate sqrt(gamma_phi)
            if gamma_phi > 0:
                collapse_ops.append((np.sqrt(gamma_phi), n_full))
        
        return collapse_ops
    
    def lindblad_rhs(self, t, rho_vec):
        """
        Right-hand side of the Lindblad master equation.
        
        Args:
            t: Time point
            rho_vec: Vectorized density matrix
            
        Returns:
            np.array: Time derivative of vectorized density matrix
        """
        # Reshape vector to matrix
        dim = int(np.sqrt(len(rho_vec)))
        rho = rho_vec.reshape(dim, dim)
        
        # Get Hamiltonian at time t
        H = self.hamiltonian.get_total_hamiltonian(t)
        
        # Coherent evolution: -i[H, rho]
        drho = -1j * (H @ rho - rho @ H) / PC.H
        
        # Dissipative evolution from collapse operators
        for rate, c_op in self.get_collapse_operators():
            c_dag = c_op.conj().T
            c_dag_c = c_dag @ c_op
            
            # Lindblad term: L[rho] = c*rho*c† - 0.5*(c†*c*rho + rho*c†*c)
            drho += rate * (c_op @ rho @ c_dag - 0.5 * c_dag_c @ rho - 0.5 * rho @ c_dag_c)
        
        # Vectorize result
        return drho.reshape(-1)
    
    def simulate_density_matrix(self, initial_state, tspan, num_steps=100):
        """
        Simulate the evolution of the density matrix.
        
        Args:
            initial_state: Initial state vector or density matrix
            tspan: Time span (t_start, t_end)
            num_steps: Number of time steps
            
        Returns:
            times: Array of time points
            rho_list: List of density matrices at each time
        """
        # Convert initial state to density matrix if needed
        if initial_state.ndim == 1:
            rho_0 = np.outer(initial_state, initial_state.conj())
        else:
            rho_0 = initial_state
            
        # Vectorize the initial density matrix
        rho_0_vec = rho_0.reshape(-1)
        
        # Time points for evaluation
        times = np.linspace(tspan[0], tspan[1], num_steps)
        
        # Solve the master equation
        solution = solve_ivp(
            self.lindblad_rhs,
            tspan,
            rho_0_vec,
            method='RK45',
            t_eval=times,
            rtol=1e-8,
            atol=1e-8
        )
        
        # Convert back to density matrices
        dim = int(np.sqrt(len(rho_0_vec)))
        rho_list = [sol.reshape(dim, dim) for sol in solution.y.T]
        
        return solution.t, rho_list
    
    def get_expectation_value(self, rho, operator):
        """
        Calculate expectation value of an operator.
        
        Args:
            rho: Density matrix
            operator: Quantum operator
            
        Returns:
            float: Expectation value Tr(rho * operator)
        """
        return np.trace(rho @ operator)
    
    def plot_population(self, times, rho_list, qubit_index=0, levels=2):
        """
        Plot population of energy levels for a specific qubit.
        
        Args:
            times: Array of time points
            rho_list: List of density matrices
            qubit_index: Index of the qubit to analyze
            levels: Number of energy levels to plot
        """
        populations = np.zeros((len(times), levels))
        
        # Get the qubit
        qubit = self.circuit.qubits[qubit_index]
        
        # Get projection operators for each level of this qubit
        proj_ops = []
        for i in range(levels):
            # Create projection operator |i⟩⟨i|
            proj = np.zeros((self.hamiltonian.hilbert_dim, self.hamiltonian.hilbert_dim))
            proj[i, i] = 1.0
            
            # Extend to the multi-qubit space
            proj_full = proj
            for other_qubit in self.circuit.qubits:
                if other_qubit.name != qubit.name:
                    proj_full = np.kron(proj_full, np.eye(self.hamiltonian.hilbert_dim))
            
            proj_ops.append(proj_full)
        
        # Calculate populations for each time
        for t_idx, rho in enumerate(rho_list):
            for level in range(levels):
                populations[t_idx, level] = self.get_expectation_value(rho, proj_ops[level])
        
        # Plot
        plt.figure(figsize=(10, 6))
        for level in range(levels):
            plt.plot(times * 1e9, populations[:, level], label=f'|{level}⟩')
        
        plt.xlabel('Time (ns)')
        plt.ylabel('Population')
        plt.title(f'Energy Level Populations for {qubit.name}')
        plt.legend()
        plt.grid(True)
        return plt.gcf()