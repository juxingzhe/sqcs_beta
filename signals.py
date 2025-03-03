# signals.py
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

class Signal:
    """Base class for quantum control signals."""
    
    def __init__(self, name=""):
        self.name = name
    
    def sequence(self, t):
        """Return signal amplitude at time t."""
        return 0.0
    
    def plot(self, trange, num_points=1000):
        """Plot the signal over a time range."""
        t = np.linspace(trange[0], trange[1], num_points)
        signal = np.array([self.sequence(ti) for ti in t])
        
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Signal: {self.name}')
        plt.grid(True)
        plt.show()

class ConstantSignal(Signal):
    """Constant amplitude signal."""
    
    def __init__(self, amplitude=0.0, name="Constant"):
        super().__init__(name)
        self.amplitude = amplitude
    
    def sequence(self, t):
        """Return constant amplitude regardless of time."""
        return self.amplitude

class PulseSequence(Signal):
    """Sequence of multiple pulses."""
    
    def __init__(self, pulses=None, name="Pulse Sequence"):
        super().__init__(name)
        self.pulses = pulses if pulses is not None else []
    
    def add_pulse(self, pulse):
        """Add a pulse to the sequence."""
        self.pulses.append(pulse)
    
    def sequence(self, t):
        """Sum of all pulse amplitudes at time t."""
        return sum(pulse.sequence(t) for pulse in self.pulses)

class GaussianPulse(Signal):
    """Gaussian shaped pulse."""
    
    def __init__(self, amplitude, sigma, t_center, name="Gaussian"):
        super().__init__(name)
        self.amplitude = amplitude
        self.sigma = sigma
        self.t_center = t_center
    
    def sequence(self, t):
        """Gaussian pulse amplitude at time t."""
        return self.amplitude * np.exp(-0.5 * ((t - self.t_center) / self.sigma) ** 2)

class SquarePulse(Signal):
    """Square pulse with constant amplitude."""
    
    def __init__(self, amplitude, t_start, t_end, name="Square"):
        super().__init__(name)
        self.amplitude = amplitude
        self.t_start = t_start
        self.t_end = t_end
    
    def sequence(self, t):
        """Square pulse amplitude at time t."""
        if isinstance(t, (list, np.ndarray)):
            result = np.zeros_like(t, dtype=float)
            mask = (t >= self.t_start) & (t <= self.t_end)
            result[mask] = self.amplitude
            return result
        else:
            return self.amplitude if self.t_start <= t <= self.t_end else 0.0

class DRAG(Signal):
    """Derivative Removal by Adiabatic Gate (DRAG) pulse for leakage reduction."""
    
    def __init__(self, amplitude, sigma, t_center, beta=0.5, name="DRAG"):
        super().__init__(name)
        self.amplitude = amplitude
        self.sigma = sigma
        self.t_center = t_center
        self.beta = beta  # DRAG correction parameter
    
    def gaussian(self, t):
        """Gaussian component."""
        return self.amplitude * np.exp(-0.5 * ((t - self.t_center) / self.sigma) ** 2)
    
    def derivative(self, t):
        """Derivative of the Gaussian component."""
        return -self.gaussian(t) * (t - self.t_center) / (self.sigma ** 2)
    
    def sequence_x(self, t):
        """X component of DRAG pulse."""
        return self.gaussian(t)
    
    def sequence_y(self, t):
        """Y component of DRAG pulse (derivative scaled by beta)."""
        return self.beta * self.derivative(t)
    
    def sequence(self, t):
        """Default is the X component."""
        return self.sequence_x(t)

class ArbitraryWaveform(Signal):
    """Arbitrary waveform defined by time-amplitude pairs with interpolation."""
    
    def __init__(self, times, amplitudes, interpolation='linear', name="Arbitrary"):
        super().__init__(name)
        self.times = np.array(times)
        self.amplitudes = np.array(amplitudes)
        
        # Create interpolation function
        if interpolation == 'cubic':
            self.interp_func = interpolate.CubicSpline(self.times, self.amplitudes)
        else:  # default to linear
            self.interp_func = interpolate.interp1d(
                self.times, self.amplitudes, 
                bounds_error=False, fill_value=0.0
            )
    
    def sequence(self, t):
        """Interpolated amplitude at time t."""
        if isinstance(t, (list, np.ndarray)):
            # For values outside the range, return zeros
            result = np.zeros_like(t, dtype=float)
            mask = (t >= self.times[0]) & (t <= self.times[-1])
            result[mask] = self.interp_func(t[mask])
            return result
        else:
            if self.times[0] <= t <= self.times[-1]:
                return self.interp_func(t)
            return 0.0

class SinusoidalSignal(Signal):
    """Sinusoidal signal with frequency, amplitude, and phase."""
    
    def __init__(self, amplitude, frequency, phase=0, t_start=None, t_end=None, name="Sinusoidal"):
        super().__init__(name)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.t_start = t_start
        self.t_end = t_end
    
    def sequence(self, t):
        """Sinusoidal amplitude at time t."""
        # Check if t is within the active time window if specified
        if (self.t_start is not None and t < self.t_start) or \
           (self.t_end is not None and t > self.t_end):
            return 0.0
        
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)