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

    # Class-level cache for expensive computations
    _curvature_cache: Dict[int, float] = {}

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

        Uses caching for performance on repeated computations.

        Args:
            n: Row index in Pascal's triangle

        Returns:
            Curvature value
        """
        if n in self._curvature_cache:
            return self._curvature_cache[n]

        if n < 2:
            curvature = 0.0
        else:
            # Central binomial coefficient with error handling
            def c(n_val):
                try:
                    return math.log(comb(n_val, n_val//2, exact=True)) if n_val >= 0 else 0.0
                except (ValueError, OverflowError):
                    # Fallback for very large n where exact computation fails
                    return n_val * math.log(2)  # Approximation

            r_n_minus_1 = c(n-1)
            r_n = c(n)
            r_n_plus_1 = c(n+1)

            curvature = r_n_plus_1 - 2*r_n + r_n_minus_1

        # Cache the result
        self._curvature_cache[n] = curvature
        return curvature

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
        Uses efficient integer arithmetic instead of string conversion.

        Args:
            x: Number to analyze
            digit: Digit to count (0-9)

        Returns:
            Count of digit occurrences
        """
        if x == 0:
            return 1 if digit == 0 else 0

        n = int(x)
        if n == 0:
            return 1 if digit == 0 else 0

        count = 0
        while n > 0:
            if n % 10 == digit:
                count += 1
            n //= 10
        return count

    def digit_shell_tension(self, x: float) -> float:
        """
        Compute digit shell tension T(x) - a measure of how "tense" the digit
        representation is, grounded in the number of carry operations needed
        when incrementing by 1.

        T(x) = Σ_{d=1}^{9} (d/9) * N_d(x) / len(digits)

        This measures the "pressure" toward digit transitions - higher values
        indicate digits closer to 9 and thus higher tension toward shell boundaries.

        Uses efficient integer arithmetic for performance.

        Args:
            x: Current position

        Returns:
            Digit shell tension value in [0, 1]
        """
        if x <= 0:
            return 0.0

        n = int(x)
        if n == 0:
            return 0.0

        total_tension = 0.0
        digit_count = 0

        # Count digits and compute tension using integer arithmetic
        temp = n
        while temp > 0:
            digit = temp % 10
            digit_count += 1
            # Higher digits contribute more tension (closer to carry-over)
            tension_contribution = (digit / 9.0)
            total_tension += tension_contribution
            temp //= 10

        if digit_count == 0:
            return 0.0

        # Normalize by digit length
        return total_tension / digit_count

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
        Compute clock rate R(x) = χ_FEG · κ_{d(x)} · (1 + T(x))

        Where T(x) is the digit shell tension measuring carry-over pressure.

        Args:
            x: Current position

        Returns:
            Clock rate at current position
        """
        digit_count = len(str(int(x))) if x > 0 else 1
        kappa = self.pascal_curvature(digit_count)
        tension = self.digit_shell_tension(x)
        return self.chi_feg * kappa * (1 + tension)

    def evolve_step(self, x: float, dt: float = 0.01) -> Tuple[float, float]:
        """
        Evolve one step using curvature-driven dynamics.

        The position evolves according to the curvature flow:
        dx/dt = κ(x) * (1 + Q_{9/11}(x)) * χ_FEG
        dθ/dt = κ(x) * χ_FEG

        This creates a geodesic flow in the discrete geometry where
        curvature drives both position and phase evolution.

        Args:
            x: Current position
            dt: Time step for numerical integration

        Returns:
            Tuple of (new_x, phase_increment)
        """
        # Get curvature field at current position
        digit_shell = len(str(int(x))) if x > 0 else 1
        kappa = self.pascal_curvature(digit_shell)
        tension = self.digit_shell_tension(x)

        # Curvature-driven velocity field
        # Position evolves with curvature magnitude and tension
        velocity = kappa * (1 + tension) * self.chi_feg

        # Phase increment based on curvature magnitude
        phase_increment = kappa * self.chi_feg * dt

        # Euler integration: x_{n+1} = x_n + v(x_n) * dt
        new_x = x + velocity * dt

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
                'digit_shell_tension': self.digit_shell_tension(x),
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

