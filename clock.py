"""
AntClock: Discrete Riemann Geometry
Core implementation of curvature clock walker and mathematical operators.
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Any
from scipy.special import comb
import matplotlib.pyplot as plt


class CurvatureClockWalker:
    """
    Fundamental dynamical system that drives the AntClock framework.

    Walks through integers carrying a curvature clock, revealing how symmetry
    breaks in digit shells mirror the critical line structure of ζ(s).
    """

    def __init__(self, x_0: float = 1.0, chi_feg: float = 0.638):
        """
        Initialize the curvature clock walker.

        Args:
            x_0: Starting position (typically 1.0)
            chi_feg: FEG coupling constant (field equation gauge)
        """
        self.x_0 = x_0
        self.chi_feg = chi_feg
        self.x_current = x_0
        self.phase_accumulated = 0.0
        self.history = []
        self.geometry_x = []
        self.geometry_y = []

    def pascal_curvature(self, n: int) -> float:
        """
        Compute Pascal curvature κ_n = r_{n+1} - 2r_n + r_{n-1}
        where r_n = log(C(n, floor(n/2)))

        Args:
            n: Row index in Pascal's triangle

        Returns:
            Curvature value
        """
        if n < 2:
            return 0.0

        # Central binomial coefficient
        def c(n): return math.log(comb(n, n//2, exact=True)) if n >= 0 else 0.0

        r_n_minus_1 = c(n-1)
        r_n = c(n)
        r_n_plus_1 = c(n+1)

        return r_n_plus_1 - 2*r_n + r_n_minus_1

    def digit_mirror(self, d: int) -> int:
        """
        Digit mirror operator μ_7(d) = d^7 mod 10

        Involution on oscillating pairs 2↔8, 3↔7
        Fixed sector: {0,1,4,5,6,9}

        Args:
            d: Digit (0-9)

        Returns:
            Mirrored digit
        """
        return pow(d, 7, 10)

    def count_digits(self, x: float, digit: int) -> int:
        """
        Count occurrences of a specific digit in the decimal representation.

        Args:
            x: Number to analyze
            digit: Digit to count (0-9)

        Returns:
            Count of digit occurrences
        """
        if x == 0:
            return 1 if digit == 0 else 0

        count = 0
        num_str = str(int(x))
        for char in num_str:
            if int(char) == digit:
                count += 1
        return count

    def nine_eleven_charge(self, x: float) -> float:
        """
        Compute 9/11 charge Q(x) = N_9(x) / (N_0(x) + 1)

        Tension metric measuring digit shell stability.

        Args:
            x: Current position

        Returns:
            9/11 charge value
        """
        n9 = self.count_digits(x, 9)
        n0 = self.count_digits(x, 0)
        return n9 / (n0 + 1)

    def angular_coordinate(self, n: int) -> float:
        """
        Angular coordinate θ(n) = (π/2) × (n mod 4)

        Mirror-phase shells at θ = 3π/2 are discrete tangent singularities.

        Args:
            n: Integer index

        Returns:
            Angular coordinate in radians
        """
        return (math.pi / 2) * (n % 4)

    def clock_rate(self, x: float) -> float:
        """
        Compute clock rate R(x) = χ_FEG · κ_{d(x)} · (1 + Q_{9/11}(x))

        Args:
            x: Current position

        Returns:
            Clock rate at current position
        """
        digit_count = len(str(int(x))) if x > 0 else 1
        kappa = self.pascal_curvature(digit_count)
        q_charge = self.nine_eleven_charge(x)
        return self.chi_feg * kappa * (1 + q_charge)

    def evolve_step(self, x: float) -> Tuple[float, float]:
        """
        Evolve one step in the curvature clock dynamics.

        Args:
            x: Current position

        Returns:
            Tuple of (new_x, phase_increment)
        """
        rate = self.clock_rate(x)
        phase_increment = rate

        # Simple Euler step (can be made more sophisticated)
        new_x = x + 1.0  # Unit step for now

        return new_x, phase_increment

    def evolve(self, steps: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Evolve the walker through digit shells for specified number of steps.

        Args:
            steps: Number of evolution steps

        Returns:
            Tuple of (history_list, summary_dict)
        """
        self.history = []
        self.geometry_x = []
        self.geometry_y = []

        x = self.x_0
        phase_total = 0.0

        for step in range(steps):
            # Record current state
            theta = self.angular_coordinate(int(x))
            state = {
                'step': step,
                'x': x,
                'theta': theta,
                'phase_accumulated': phase_total,
                'digit_shell': len(str(int(x))) if x > 0 else 1,
                'clock_rate': self.clock_rate(x),
                'nine_eleven_charge': self.nine_eleven_charge(x),
                'pascal_curvature': self.pascal_curvature(len(str(int(x))) if x > 0 else 1)
            }

            self.history.append(state)

            # Update geometry for visualization
            self.geometry_x.append(math.cos(theta))
            self.geometry_y.append(math.sin(theta))

            # Evolve one step
            x, phase_inc = self.evolve_step(x)
            phase_total += phase_inc

        # Compute bifurcation index (simplified version)
        bifurcation_index = max([h['pascal_curvature'] for h in self.history[-10:]] or [0])

        summary = {
            'total_steps': steps,
            'final_x': x,
            'total_phase': phase_total,
            'bifurcation_index': bifurcation_index,
            'max_digit_shell': max([h['digit_shell'] for h in self.history]),
            'mirror_phase_transitions': len([h for h in self.history
                                           if h['theta'] == 3*math.pi/2])
        }

        return self.history, summary

    def get_geometry(self) -> Tuple[List[float], List[float]]:
        """
        Get geometry coordinates for visualization.

        Returns:
            Tuple of (x_coords, y_coords) for unit circle plotting
        """
        return self.geometry_x, self.geometry_y

    def plot_geometry(self, save_path: str = None):
        """
        Plot the unit circle geometry with phase transitions.

        Args:
            save_path: Optional path to save the plot
        """
        x_coords, y_coords = self.get_geometry()

        plt.figure(figsize=(10, 10))
        plt.plot(x_coords, y_coords, 'b-', alpha=0.7, linewidth=2)

        # Plot unit circle
        theta_circle = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', alpha=0.3)

        # Mark mirror-phase shells (θ = 3π/2)
        mirror_indices = [i for i, h in enumerate(self.history)
                         if h['theta'] == 3*math.pi/2]
        if mirror_indices:
            mirror_x = [self.geometry_x[i] for i in mirror_indices]
            mirror_y = [self.geometry_y[i] for i in mirror_indices]
            plt.scatter(mirror_x, mirror_y, c='red', s=50, alpha=0.8,
                       label='Mirror-phase shells')

        plt.axis('equal')
        plt.title('AntClock: Unit Circle Geometry with Phase Transitions')
        plt.xlabel('cos(θ)')
        plt.ylabel('sin(θ)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Geometry plot saved to {save_path}")
        else:
            plt.show()


def compute_enhanced_betti_numbers(shell_index: int) -> List[int]:
    """
    Compute enhanced Betti numbers for a digit shell.

    This is a simplified version - full implementation would use
    persistent homology on Pascal row simplicial complexes.

    Args:
        shell_index: Digit shell index (number of digits)

    Returns:
        List of Betti numbers [β₀, β₁, β₂, ...]
    """
    # Simplified Betti numbers based on shell structure
    if shell_index == 1:
        return [1, 0, 0]  # Single point
    elif shell_index == 2:
        return [1, 1, 0]  # Circle-like
    elif shell_index == 7:  # Mirror shell
        return [1, 3, 1]  # More complex topology
    else:
        # General case - increasing complexity
        beta0 = 1
        beta1 = shell_index % 4  # Mirror symmetry breaking
        beta2 = 1 if shell_index % 4 == 3 else 0  # Mirror shells have holes
        return [beta0, beta1, beta2]


# Convenience functions for demos
def create_walker(x_0: float = 1.0, chi_feg: float = 0.638) -> CurvatureClockWalker:
    """Create a new CurvatureClockWalker instance."""
    return CurvatureClockWalker(x_0=x_0, chi_feg=chi_feg)


def run_basic_demo(steps: int = 1000) -> Dict[str, Any]:
    """
    Run a basic demonstration of the curvature clock walker.

    Args:
        steps: Number of evolution steps

    Returns:
        Summary dictionary
    """
    walker = create_walker()
    history, summary = walker.evolve(steps)
    return summary
