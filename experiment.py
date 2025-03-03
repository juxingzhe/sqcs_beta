# experiments.py
import numpy as np
import matplotlib.pyplot as plt
from signals import GaussianPulse, SquarePulse, DRAG, SinusoidalSignal, PulseSequence

class QuantumExperiment:
    """Base class for quantum experiments on superconducting circuits."""
    
    def __init__(self, circuit, simulator):
        """
        Initialize a quantum experiment.
        
        Args:
            circuit: SQCircuit instance
            simulator: DensityMatrixSimulator instance
        """
        self.circuit = circuit
        self.simulator = simulator
        
    def prepare_initial_state(self):
        """
        Prepare the initial state for the experiment.
        Default is ground state |0⟩ for all qubits.
        
        Returns:
            np.array: Initial state vector
        """
        # Ground state |0⟩ for each qubit
        state = np.zeros(self.simulator.hamiltonian.hilbert_dim ** self.circuit.qubits_num)
        state[0] = 1.0  # |00...0⟩
        return state
    
    def run(self):
        """Run the experiment (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement this method")

class RabiExperiment(QuantumExperiment):
    """Rabi oscillation experiment."""
    
    def __init__(self, circuit, simulator, qubit_name, drive_amplitudes, pulse_duration=50e-9):
        """
        Initialize Rabi experiment.
        
        Args:
            circuit: SQCircuit instance
            simulator: DensityMatrixSimulator instance
            qubit_name: Name of the target qubit
            drive_amplitudes: List of drive amplitudes to test
            pulse_duration: Duration of each pulse in seconds
        """
        super().__init__(circuit, simulator)
        self.qubit_name = qubit_name
        self.qubit = circuit[qubit_name]
        self.drive_amplitudes = drive_amplitudes
        self.pulse_duration = pulse_duration
        
    def run(self):
        """
        Run the Rabi experiment with varying drive amplitudes.
        
        Returns:
            dict: Results including amplitudes and final |1⟩ state populations
        """
        results = {
            'amplitudes': self.drive_amplitudes,
            'populations': []
        }
        
        # Define the projection operator to |1⟩ state
        proj_1 = np.zeros((self.simulator.hamiltonian.hilbert_dim, self.simulator.hamiltonian.hilbert_dim))
        proj_1[1, 1] = 1.0
        
        # Extend to multi-qubit space
        for other_qubit in self.circuit.qubits:
            if other_qubit.name != self.qubit.name:
                proj_1 = np.kron(proj_1, np.eye(self.simulator.hamiltonian.hilbert_dim))
        
        # Run experiment for each amplitude
        for amplitude in self.drive_amplitudes:
            # Set X drive signal
            pulse = SquarePulse(amplitude, 0, self.pulse_duration, f"Rabi_{amplitude}")
            self.qubit['Signal_x'] = pulse
            
            # Clear other signals
            self.qubit['Signal_y'] = None
            self.qubit['Signal_z'] = None
            
            # Prepare initial state and run simulation
            initial_state = self.prepare_initial_state()
            times, rho_list = self.simulator.simulate_density_matrix(
                initial_state, 
                (0, self.pulse_duration),
                num_steps=100
            )
            
            # Get final state population in |1⟩
            final_rho = rho_list[-1]
            p1 = self.simulator.get_expectation_value(final_rho, proj_1)
            results['populations'].append(p1)
        
        return results
    
    def plot_results(self, results=None):
        """
        Plot the results of the Rabi experiment.
        
        Args:
            results: Results from the run method (if None, run the experiment)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if results is None:
            results = self.run()
            
        # Convert to NumPy arrays
        amplitudes = np.array(results['amplitudes'])
        populations = np.array(results['populations'])
        
        # Plot Rabi oscillation
        plt.figure(figsize=(10, 6))
        plt.plot(amplitudes, populations, 'o-')
        plt.xlabel('Drive Amplitude')
        plt.ylabel('|1⟩ State Population')
        plt.title(f'Rabi Experiment - {self.qubit_name}')
        plt.grid(True)
        
        return plt.gcf()

class RamseyExperiment(QuantumExperiment):
    """Ramsey interferometry experiment."""
    
    def __init__(self, circuit, simulator, qubit_name, delay_times, 
                 detuning=0.0, pulse_duration=10e-9, pulse_amplitude=1.0):
        """
        Initialize Ramsey experiment.
        
        Args:
            circuit: SQCircuit instance
            simulator: DensityMatrixSimulator instance
            qubit_name: Name of the target qubit
            delay_times: List of delay times between π/2 pulses
            detuning: Frequency detuning from resonance
            pulse_duration: Duration of π/2 pulses
            pulse_amplitude: Amplitude of π/2 pulses
        """
        super().__init__(circuit, simulator)
        self.qubit_name = qubit_name
        self.qubit = circuit[qubit_name]
        self.delay_times = delay_times
        self.detuning = detuning
        self.pulse_duration = pulse_duration
        self.pulse_amplitude = pulse_amplitude
        
    def run(self):
        """
        Run the Ramsey experiment with varying delay times.
        
        Returns:
            dict: Results including delay times and final |1⟩ state populations
        """
        results = {
            'delay_times': self.delay_times,
            'populations': []
        }
        
        # Define the projection operator to |1⟩ state
        proj_1 = np.zeros((self.simulator.hamiltonian.hilbert_dim, self.simulator.hamiltonian.hilbert_dim))
        proj_1[1, 1] = 1.0
        
        # Extend to multi-qubit space
        for other_qubit in self.circuit.qubits:
            if other_qubit.name != self.qubit.name:
                proj_1 = np.kron(proj_1, np.eye(self.simulator.hamiltonian.hilbert_dim))
        
        # Run experiment for each delay time
        for delay in self.delay_times:
            # Create pulse sequence: π/2 - delay - π/2
            first_pulse = GaussianPulse(self.pulse_amplitude, self.pulse_duration/4, self.pulse_duration/2, "First π/2")
            second_pulse = GaussianPulse(self.pulse_amplitude, self.pulse_duration/2 + delay + self.pulse_duration/2, 
                                         self.pulse_duration/4, "Second π/2")
            
            pulse_seq = PulseSequence([first_pulse, second_pulse], "Ramsey Sequence")
            
            # Set qubit drive signals
            self.qubit['Signal_x'] = pulse_seq
            
            # Add detuning if specified
            if self.detuning != 0.0:
                detuning_signal = SinusoidalSignal(1.0, self.detuning, 0, 0, 
                                                  self.pulse_duration + delay + self.pulse_duration,
                                                  "Detuning")
                self.qubit['Signal_z'] = detuning_signal
            else:
                self.qubit['Signal_z'] = None
                
            self.qubit['Signal_y'] = None
            
            # Prepare initial state and run simulation
            initial_state = self.prepare_initial_state()
            times, rho_list = self.simulator.simulate_density_matrix(
                initial_state, 
                (0, self.pulse_duration + delay + self.pulse_duration),
                num_steps=200
            )
            
            # Get final state population in |1⟩
            final_rho = rho_list[-1]
            p1 = self.simulator.get_expectation_value(final_rho, proj_1)
            results['populations'].append(p1)
        
        return results
    
    def plot_results(self, results=None):
        """
        Plot the results of the Ramsey experiment.
        
        Args:
            results: Results from the run method (if None, run the experiment)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if results is None:
            results = self.run()
            
        # Convert to NumPy arrays
        delay_times = np.array(results['delay_times'])
        populations = np.array(results['populations'])
        
        # Plot Ramsey oscillation
        plt.figure(figsize=(10, 6))
        plt.plot(delay_times * 1e9, populations, 'o-')
        plt.xlabel('Delay Time (ns)')
        plt.ylabel('|1⟩ State Population')
        plt.title(f'Ramsey Experiment - {self.qubit_name}')
        plt.grid(True)
        
        return plt.gcf()

class CouplerStudyExperiment(QuantumExperiment):
    """Study of coupler bias effect on qubit-qubit interaction."""
    
    def __init__(self, circuit, simulator, coupler_name, bias_values, 
                 measurement_time=100e-9):
        """
        Initialize coupler study experiment.
        
        Args:
            circuit: SQCircuit instance
            simulator: DensityMatrixSimulator instance
            coupler_name: Name of the target coupler
            bias_values: List of bias values to test
            measurement_time: Time for each measurement
        """
        super().__init__(circuit, simulator)
        self.coupler_name = coupler_name
        self.coupler = circuit[coupler_name]
        self.bias_values = bias_values
        self.measurement_time = measurement_time
        
        # Identify the connected qubits
        self.connected_qubits = [circuit[name] for name in self.coupler['connected_qubits']]
        if len(self.connected_qubits) != 2:
            raise ValueError(f"Coupler {coupler_name} must connect exactly 2 qubits for this experiment")
        
    def run(self):
        """
        Run the coupler study experiment with varying bias values.
        
        Returns:
            dict: Results including bias values and effective coupling strengths
        """
        results = {
            'bias_values': self.bias_values,
            'frequencies': [],
            'coupling_strengths': []
        }
        
        # Run experiment for each bias value
        for bias in self.bias_values:
            # Set coupler bias signal
            bias_signal = ConstantSignal(bias, f"Bias_{bias}")
            self.coupler['Signal_z'] = bias_signal
            
            # Get bare Hamiltonian and find eigenvalues
            H = self.simulator.hamiltonian.get_total_hamiltonian(0)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            
            # Extract qubit frequencies from the eigenvalues
            # Assuming the first 4 eigenvalues correspon