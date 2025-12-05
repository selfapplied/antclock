"""
AntClock: Discrete Riemann Geometry
Core implementation of curvature clock walker and mathematical operators.
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Any, Iterator
from scipy.special import comb
import matplotlib.pyplot as plt
from .definitions import (
    Digit, RowIndex, ShellIndex, CE1GeometryPoint,
    AntClockStep, AntClockSummary, FlowField,
    ContinuedFraction, GaussSystem, Triangulation,
    UniversalClock, validate_digit, validate_shell_index
)


class CurvatureClockWalker:
    """
    Fundamental dynamical system that drives the AntClock framework.

    Walks through integers carrying a curvature clock, revealing how symmetry
    breaks in digit shells mirror the critical line structure of ζ(s).
    """

    # Class-level cache for expensive computations
    _curvature_cache: Dict[int, float] = {}

    def __init__(self, x_0: float = 1.0, chi_feg: float = 0.638, enable_volte: bool = False):
        """
        Initialize the curvature clock walker.

        Args:
            x_0: Starting position (typically 1.0)
            chi_feg: FEG coupling constant (field equation gauge / Volte threshold)
            enable_volte: Whether to enable explicit Volte operator corrections
        """
        self.x_0 = x_0
        self.chi_feg = chi_feg
        self.enable_volte = enable_volte
        self.x_current = x_0
        self.phase_accumulated = 0.0
        self.history = []
        self.geometry_x = []
        self.geometry_y = []

        # Initialize Volte system if enabled
        if self.enable_volte:
            from .volte import AntClockVolteSystem
            self.volte_system = AntClockVolteSystem(chi_feg=chi_feg)
        else:
            self.volte_system = None

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

    def digit_mirror(self, d: int) -> Digit:
        """
        Digit mirror operator μ₇(d) = d^7 mod 10

        CE1 Galois involution on digit space.
        Involution on oscillating pairs 2↔8, 3↔7
        Fixed sector: {0,1,4,5,6,9}

        Args:
            d: Digit (0-9)

        Returns:
            Mirrored digit as validated Digit type
        """
        result = pow(d, 7, 10)
        return validate_digit(result)

    def count_digits(self, x: float, digit: Digit) -> int:
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

    def continued_fraction_expansion(self, x: float, max_terms: int = 20) -> List[int]:
        """
        Compute continued fraction expansion of x.

        CE1: Combinatorial skeleton - every number becomes a sequence/grammar
        Returns the sequence of partial quotients.

        Args:
            x: Real number to expand
            max_terms: Maximum number of terms to compute

        Returns:
            List of integer partial quotients [a0, a1, a2, ...]
        """
        if not math.isfinite(x) or math.isnan(x):
            return []

        terms = []
        current = x
        seen = set()  # Detect cycles for quadratic irrationals

        for _ in range(max_terms):
            if current in seen:
                break  # Cycle detected
            seen.add(current)

            integer_part = math.floor(current)
            terms.append(integer_part)

            fractional_part = current - integer_part
            if fractional_part == 0:
                break  # Exact rational

            current = 1.0 / fractional_part

        return terms

    def continued_fraction_convergents(self, terms: List[int]) -> List[Tuple[int, int]]:
        """
        Compute convergents of a continued fraction.

        CE3: Each convergent is a simplex (rational triangle) around the truth.
        Returns list of (numerator, denominator) pairs.

        Args:
            terms: Partial quotients from continued fraction expansion

        Returns:
            List of (p_n, q_n) convergents
        """
        if not terms:
            return []

        convergents = []

        # Initialize with first convergent
        if len(terms) >= 1:
            p_prev, q_prev = terms[0], 1
            convergents.append((p_prev, q_prev))

        if len(terms) >= 2:
            p_curr, q_curr = terms[1] * terms[0] + 1, terms[1]
            convergents.append((p_curr, q_curr))

            # Compute remaining convergents
            for i in range(2, len(terms)):
                p_next = terms[i] * p_curr + p_prev
                q_next = terms[i] * q_curr + q_prev
                convergents.append((p_next, q_next))
                p_prev, q_prev = p_curr, q_curr
                p_curr, q_curr = p_next, q_next

        return convergents

    def gauss_map(self, x: float) -> float:
        """
        Gauss map: x ↦ 1/x - ⌊1/x⌋

        CE2 Transport: CE1's discrete recursion becomes CE2's dynamical flow.
        This map has an invariant distribution - CE2's signature.

        Args:
            x: Input in (0, 1)

        Returns:
            Gauss map output in (0, 1)
        """
        if not (0 < x < 1):
            return 0.0

        return 1.0 / x - math.floor(1.0 / x)

    def khinchin_constant_sample(self, n_samples: int = 1000) -> float:
        """
        Estimate Khinchin's constant through Monte Carlo sampling.

        CE2 Invariant: Statistical regularity emerging from CE1 combinatorics.
        K ≈ 2.6854520010...

        Args:
            n_samples: Number of random samples

        Returns:
            Estimated Khinchin constant
        """
        np.random.seed(42)  # For reproducibility

        log_products = []
        for _ in range(n_samples):
            # Generate random continued fraction terms
            # Khinchin distribution favors small integers
            terms = []
            for _ in range(10):  # Reasonable length
                # Geometric distribution favoring 1,2,3,...
                term = int(np.random.geometric(0.5)) + 1
                terms.append(term)

            # Compute product of (1 + 1/(a_i(a_i+2)))
            product = 1.0
            for a in terms:
                product *= (1.0 + 1.0 / (a * (a + 2)))

            if product > 1:
                log_products.append(math.log(product))

        if not log_products:
            return 0.0

        return math.exp(np.mean(log_products))

    def digital_polynomial(self, n: int, base: int = 10) -> List[int]:
        """
        Convert number to digital polynomial coefficients.

        CE1: Maps digits → coefficients, carries structure → factorization.
        n = d_k * base^k + d_{k-1} * base^{k-1} + ... + d_0

        Args:
            n: Number to convert
            base: Base for polynomial representation

        Returns:
            List of coefficients [d_k, d_{k-1}, ..., d_0]
        """
        if n == 0:
            return [0]

        coefficients = []
        while n > 0:
            coefficients.append(n % base)
            n //= base

        return coefficients[::-1]  # Reverse to get highest degree first

    def polynomial_evaluation(self, coeffs: List[int], x: float, base: int = 10) -> float:
        """
        Evaluate digital polynomial at point x.

        CE2 Transport: Digital polynomial becomes spectral operator.

        Args:
            coeffs: Polynomial coefficients
            x: Evaluation point
            base: Base (affects scaling)

        Returns:
            Polynomial value at x
        """
        result = 0.0
        for coeff in coeffs:
            result = result * x + coeff
        return result

    def universal_clock_increment(self, event_type: str, layer: int) -> int:
        """
        Universal clock increment based on event type and layer.

        CE1: Discrete recursion ticks
        CE2: Flow parameter increments
        CE3: Event/action index increments

        Args:
            event_type: Type of event ('bracket', 'flow_step', 'simplex_flip', etc.)
            layer: CE layer (1, 2, or 3)

        Returns:
            Clock increment value
        """
        base_increment = {
            'bracket': 1,      # CE1 structure event
            'flow_step': 10,   # CE2 dynamics step
            'simplex_flip': 100  # CE3 emergence event
        }.get(event_type, 1)

        # Layer scaling
        layer_multiplier = 10 ** (layer - 1)

        return base_increment * layer_multiplier

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
        Evolve one step using curvature-driven dynamics with optional Volte corrections.

        The position evolves according to the curvature flow:
        dx/dt = κ(x) * (1 + Q_{9/11}(x)) * χ_FEG + V(x,u)  [if Volte enabled]

        dθ/dt = κ(x) * χ_FEG

        When Volte is enabled, this implements the full Volte equation:
        dx/dt = F(x,u) + V(x,u) where V activates when stress > χ_FEG.

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

        # Base curvature-driven velocity field
        velocity = kappa * (1 + tension) * self.chi_feg

        # Apply Volte correction if enabled
        if self.enable_volte and self.volte_system is not None:
            # Compute stress level
            stress_level = abs(kappa)  # Curvature magnitude as stress

            # Check if Volte should activate (stress > threshold)
            if stress_level > self.chi_feg:
                # Compute Volte correction
                volte_correction = self.volte_system._ce_aware_volte_correction(x, 0.0)
                # Apply correction to velocity
                velocity += volte_correction

        # Phase increment based on curvature magnitude
        phase_increment = kappa * self.chi_feg * dt

        # Euler integration: x_{n+1} = x_n + v(x_n) * dt
        new_x = x + velocity * dt

        return new_x, phase_increment

    def evolve(self, steps: int) -> Tuple[List[AntClockStep], AntClockSummary]:
        """
        Evolve the walker through digit shells for specified number of steps.

        CE2→CE3 aggregator: local dynamics → global emergent state.

        Args:
            steps: Number of evolution steps

        Returns:
            Tuple of (history: CE2 trajectory, summary: CE3 witness)
        """
        self.history = []
        self.geometry_x = []
        self.geometry_y = []

        x = self.x_0
        phase_total = 0.0

        for step in range(steps):
            # Record current state as CE2 AntClockStep
            theta = self.angular_coordinate(int(x))
            digit_shell = validate_shell_index(len(str(int(x))) if x > 0 else 1)
            mirror_cross = (theta == 3 * math.pi / 2)  # Mirror phase shell crossing

            state: AntClockStep = {
                'step': step,
                'x': x,
                'phase': phase_total,
                'digit_shell': digit_shell,
                'clock_rate': self.clock_rate(x),
                'mirror_cross': mirror_cross
            }

            self.history.append(state)

            # Update geometry for visualization
            self.geometry_x.append(math.cos(theta))
            self.geometry_y.append(math.sin(theta))

            # Evolve one step
            x, phase_inc = self.evolve_step(x)
            phase_total += phase_inc

        # Compute bifurcation index (simplified version)
        bifurcation_index = max([self.pascal_curvature(h['digit_shell']) for h in self.history[-10:]] or [0])

        # Create CE3 witness summary
        summary: AntClockSummary = {
            'total_steps': steps,
            'final_x': x,
            'total_phase': phase_total,
            'bifurcation_index': bifurcation_index,
            'max_digit_shell': max([h['digit_shell'] for h in self.history]),
            'mirror_phase_transitions': len([h for h in self.history if h['mirror_cross']])
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

        # Mark mirror-phase shells (mirror crossings)
        mirror_indices = [i for i, h in enumerate(self.history)
                         if h['mirror_cross']]
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
def create_walker(x_0: float = 1.0, chi_feg: float = 0.638, enable_volte: bool = False) -> CurvatureClockWalker:
    """
    Create a new CurvatureClockWalker instance.

    Args:
        x_0: Starting position
        chi_feg: FEG coupling constant (Volte threshold)
        enable_volte: Whether to enable Volte operator corrections

    Returns:
        Configured CurvatureClockWalker
    """
    return CurvatureClockWalker(x_0=x_0, chi_feg=chi_feg, enable_volte=enable_volte)


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

