"""
pascal_clock.py

AntClock: Pascal Curvature Clock Walker.

A minimal implementation of the curvature-clock walker - the smallest nontrivial
creature that actually moves using Pascal clock machinery.

Author: Joel
"""

from typing import Callable, Optional, Dict, Any, Tuple
import numpy as np
from scipy.special import comb

# ============================================================================
# Core Pascal Curvature Functions
# ============================================================================

def pascal_radius(n: int) -> float:
    """
    Compute the "radius" of row n of Pascal's triangle.

    r_n = log(C(n, floor(n/2)))

    This is the "bulk thickness" of the row - how wide the combinatorial middle is.

    Args:
        n: Row index (n >= 0)

    Returns:
        r_n = log of central binomial coefficient
    """
    if n < 0:
        return 0.0

    # Central coefficient: C(n, floor(n/2))
    k_center = n // 2
    central_coeff = comb(n, k_center, exact=False)

    # Avoid log(0)
    if central_coeff <= 0:
        return 0.0

    return np.log(central_coeff)

def pascal_curvature(n: int) -> float:
    """
    Compute discrete curvature of Pascal row n.

    κ_n = r_{n+1} - 2r_n + r_{n-1}

    Args:
        n: Row index (n >= 1)

    Returns:
        κ_n = discrete curvature
    """
    if n < 1:
        return 0.0

    r_n_plus_1 = pascal_radius(n + 1)
    r_n = pascal_radius(n)
    r_n_minus_1 = pascal_radius(n - 1) if n >= 1 else 0.0

    kappa = r_n_plus_1 - 2.0 * r_n + r_n_minus_1

    return float(kappa)

def digit_count(x: float) -> int:
    """
    Compute digit count of x.

    d(x) = floor(log10(x)) + 1

    Args:
        x: The number

    Returns:
        d(x) = number of digits
    """
    if x <= 0:
        return 1

    # Handle very small numbers
    if x < 1.0:
        # Count significant digits
        x_str = f"{x:.15f}"
        if '.' in x_str:
            digits = x_str.replace('.', '').replace('-', '').lstrip('0')
            return len(digits) if digits else 1
        return 1

    # Standard case: floor(log10(x)) + 1
    d = int(np.floor(np.log10(abs(x)))) + 1
    return max(1, d)

def digit_boundary_curvature(x: float) -> float:
    """
    Digit-boundary curvature operator K(x).

    K(x) = κ_{d(x)}

    Args:
        x: The number

    Returns:
        K(x) = curvature at digit-shell d(x)
    """
    d = digit_count(x)
    return pascal_curvature(d)

def count_digits(number: float) -> Dict[str, int]:
    """
    Count digits in a number.

    Args:
        number: The number to analyze

    Returns:
        Dictionary with digit counts
    """
    # Convert to string representation
    num_str = f"{number:.15f}".replace('.', '').replace('-', '')

    digit_counts = {}
    for digit in '0123456789':
        digit_counts[digit] = num_str.count(digit)

    return digit_counts

def extract_ballast_and_units(number: float) -> Dict[str, int]:
    """
    Extract ballast (0's) and units (9's) from a number.

    The 9/11 conjecture: 0's = ballast, 9's = tension units

    Args:
        number: The number to analyze

    Returns:
        Dictionary with ballast and unit analysis
    """
    digit_counts = count_digits(number)

    ballasts = digit_counts.get('0', 0)
    units = digit_counts.get('9', 0)

    total_digits = sum(digit_counts.values())

    # Q₉₍₁₁₎ = tension / (ballast + 1)
    q_9_11 = units / (ballasts + 1.0) if (ballasts + 1.0) > 0 else float('inf')

    return {
        'ballasts': ballasts,
        'units': units,
        'total_digits': total_digits,
        'q_9_11': q_9_11,
        'digit_counts': digit_counts
    }

def compute_q_9_11(number: float) -> float:
    """
    Compute Q₉₍₁₁₎ = tension / (ballast + 1).

    Args:
        number: The number

    Returns:
        Q₉₍₁₁₎ value
    """
    analysis = extract_ballast_and_units(number)
    return analysis['q_9_11']

def local_curvature_charge(x: float) -> float:
    """
    Local digit-curvature charge K_loc(x) = K(x) · (1 + Q_9/11(x))

    Args:
        x: The number

    Returns:
        K_loc(x) = local curvature charge
    """
    K_x = digit_boundary_curvature(x)
    Q_9_11 = compute_q_9_11(x)

    K_loc = K_x * (1.0 + Q_9_11)

    return float(K_loc)

def clock_rate(x: float, chi_feg: float = 0.638) -> float:
    """
    Clock rate R(x) = χ_FEG · K_loc(x)

    Args:
        x: Current state
        chi_feg: Feigenbaum scaling factor (default: 0.638)

    Returns:
        R(x) = clock rate
    """
    K_loc = local_curvature_charge(x)
    R = chi_feg * K_loc

    return float(R)

# ============================================================================
# The Tiny Machine: Curvature-Clock Walker
# ============================================================================

def jump_factor_curvature(d: int) -> int:
    """
    Compute jump factor based on curvature: J_d = floor(|κ_d| · 10^d)

    Args:
        d: Digit-shell index

    Returns:
        J_d = jump factor
    """
    kappa_d = pascal_curvature(d)
    J_d = int(np.floor(abs(kappa_d) * (10.0 ** d)))
    return max(1, J_d)

def jump_factor_constant(d: int) -> int:
    """
    Constant jump factor: J_d = 1

    Args:
        d: Digit-shell index (unused, but kept for interface consistency)

    Returns:
        J_d = 1
    """
    return 1

class CurvatureClockWalker:
    """
    The Tiny Machine: Curvature-Clock Walker.

    A complete dynamic system that moves using the Pascal curvature clock.

    State at time step t:
    - x_t ∈ ℕ: the "body" (integer)
    - τ_t ∈ ℝ: the internal clock phase
    - φ_t ∈ [0, 2π): an angle on the unit circle (for frequency & geometry)

    Update rules:
    1. Clock phase: τ_{t+1} = τ_t + R(x_t)
    2. Angle: φ_{t+1} = (φ_t + R(x_t)) mod 2π
    3. Body: x_{t+1} = x_t + 1 (same shell) or x_t + J_d (boundary)
    """

    def __init__(self,
                 x_0: int = 1,
                 tau_0: float = 0.0,
                 phi_0: float = 0.0,
                 chi_feg: float = 0.638,
                 jump_factor: Optional[Callable[[int], int]] = None):
        """
        Initialize Curvature-Clock Walker.

        Args:
            x_0: Initial body state (integer)
            tau_0: Initial clock phase
            phi_0: Initial angle (in radians)
            chi_feg: Feigenbaum scaling factor
            jump_factor: Function J_d(n) for jump at digit-boundary d
                         (default: constant 1)
        """
        self.x = x_0
        self.tau = tau_0
        self.phi = phi_0 % (2.0 * np.pi)  # Wrap to [0, 2π)

        self.chi_feg = chi_feg

        # Jump factor function
        if jump_factor is None:
            # Default: constant jump
            self.jump_factor = jump_factor_constant
        else:
            self.jump_factor = jump_factor

        # History
        self.history: list[Dict[str, Any]] = []
        self.boundary_events: list[Dict[str, Any]] = []

    def R(self, x: float) -> float:
        """
        Compute clock rate R(x).

        Args:
            x: Current state

        Returns:
            R(x) = clock rate
        """
        return clock_rate(x, chi_feg=self.chi_feg)

    def step(self) -> Dict[str, Any]:
        """
        Single step of the walker.

        Returns:
            Dictionary with state and metadata
        """
        # Current state
        x_t = self.x
        tau_t = self.tau
        phi_t = self.phi
        d_t = digit_count(x_t)

        # Compute clock rate
        R_val = self.R(x_t)

        # Update clock phase
        tau_next = tau_t + R_val

        # Update angle (same rate spins the pointer)
        phi_next = (phi_t + R_val) % (2.0 * np.pi)

        # Update body
        d_next_check = digit_count(x_t + 1)

        if d_next_check == d_t:
            # Same shell: smooth step
            x_next = x_t + 1
            boundary_crossed = False
            jump_used = 1
        else:
            # Digit boundary: jump
            J_d = self.jump_factor(d_t)
            x_next = x_t + J_d
            boundary_crossed = True
            jump_used = J_d

            # Record boundary event
            d_next = digit_count(x_next)
            K_old = digit_boundary_curvature(x_t)
            K_new = digit_boundary_curvature(x_next)

            self.boundary_events.append({
                't': len(self.history),
                'x_old': x_t,
                'x_new': x_next,
                'd_old': d_t,
                'd_new': d_next,
                'K_old': K_old,
                'K_new': K_new,
                'tau': tau_next,
                'phi': phi_next,
                'jump': J_d
            })

        # Update state
        self.x = x_next
        self.tau = tau_next
        self.phi = phi_next

        # Record history
        metadata = {
            't': len(self.history),
            'x': x_t,
            'x_next': x_next,
            'tau': tau_t,
            'tau_next': tau_next,
            'phi': phi_t,
            'phi_next': phi_next,
            'R': R_val,
            'd': d_t,
            'd_next': digit_count(x_next),
            'K': digit_boundary_curvature(x_t),
            'Q_9_11': compute_q_9_11(x_t),
            'boundary_crossed': boundary_crossed,
            'jump_used': jump_used
        }

        self.history.append(metadata)

        return metadata

    def evolve(self, n_steps: int) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
        """
        Evolve walker for n_steps.

        Args:
            n_steps: Number of steps to evolve

        Returns:
            (history, summary) tuple
        """
        # Reset history if needed
        if len(self.history) == 0:
            # Record initial state
            self.history.append({
                't': 0,
                'x': self.x,
                'x_next': self.x,
                'tau': self.tau,
                'tau_next': self.tau,
                'phi': self.phi,
                'phi_next': self.phi,
                'R': self.R(self.x),
                'd': digit_count(self.x),
                'd_next': digit_count(self.x),
                'K': digit_boundary_curvature(self.x),
                'Q_9_11': compute_q_9_11(self.x),
                'boundary_crossed': False,
                'jump_used': 0
            })

        # Evolve
        for _ in range(n_steps):
            self.step()

        # Compute summary statistics
        R_seq = [h['R'] for h in self.history]
        phi_seq = [h['phi'] for h in self.history]
        tau_seq = [h['tau'] for h in self.history]
        x_seq = [h['x'] for h in self.history]

        # Frequency: average angular speed
        if len(R_seq) > 0:
            frequency = np.mean(R_seq)
        else:
            frequency = 0.0

        summary = {
            'n_steps': len(self.history) - 1,
            'x_final': self.x,
            'tau_final': self.tau,
            'phi_final': self.phi,
            'frequency': frequency,
            'R_mean': float(np.mean(R_seq)),
            'R_std': float(np.std(R_seq)),
            'R_min': float(np.min(R_seq)),
            'R_max': float(np.max(R_seq)),
            'boundary_count': len(self.boundary_events),
            'x_range': (int(min(x_seq)), int(max(x_seq))),
            'tau_range': (float(min(tau_seq)), float(max(tau_seq)))
        }

        return self.history, summary

    def get_geometry(self) -> Tuple[list[float], list[float]]:
        """
        Get geometry: (cos(φ_t), sin(φ_t)) trajectory.

        Returns:
            (x_coords, y_coords) tuple for plotting on unit circle
        """
        x_coords = [np.cos(h['phi']) for h in self.history]
        y_coords = [np.sin(h['phi']) for h in self.history]

        return x_coords, y_coords

    def reset(self, x_0: Optional[int] = None,
              tau_0: Optional[float] = None,
              phi_0: Optional[float] = None):
        """
        Reset walker state.

        Args:
            x_0: New initial body state
            tau_0: New initial clock phase
            phi_0: New initial angle
        """
        if x_0 is not None:
            self.x = x_0
        if tau_0 is not None:
            self.tau = tau_0
        if phi_0 is not None:
            self.phi = phi_0 % (2.0 * np.pi)

        self.history = []
        self.boundary_events = []
