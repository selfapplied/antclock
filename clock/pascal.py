"""
clock.pascal_core - Core mathematical primitives for Pascal curvature clocks.

This module contains the fundamental mathematical operations that form the
foundation of the curvature-clock system. These primitives are designed to be
pure functions that can be composed to build more complex behaviors.

Author: Joel
"""

from typing import Dict
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
    if n == 0:
        return 0.0

    # For large n, use Stirling approximation: log(C(n,k)) ≈ n*log(n) - n + 0.5*log(2πn) + ...
    # For central binomial coefficient C(n, n//2), we can use a simplified approximation
    try:
        # Central coefficient: C(n, floor(n/2))
        k_center = n // 2
        central_coeff = comb(n, k_center, exact=False)

        # Check for NaN or infinity
        if not np.isfinite(central_coeff) or central_coeff <= 0:
            # Use Stirling approximation for large n
            # log(C(n, n/2)) ≈ n*log(2) - 0.5*log(π*n/2)
            return n * np.log(2) - 0.5 * np.log(np.pi * n / 2)

        return np.log(central_coeff)
    except (ValueError, OverflowError):
        # Fallback for very large n
        return n * np.log(2) - 0.5 * np.log(np.pi * n / 2)


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
    if not np.isfinite(x) or x <= 0:
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
    try:
        d = int(np.floor(np.log10(float(abs(x))))) + 1
        return max(1, d)
    except (ValueError, OverflowError):
        # For very large numbers, estimate based on string representation
        x_str = str(int(abs(x)))
        return len(x_str)


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


def bifurcation_index(x: float, c_star: float = 0.0, delta: float = 0.638) -> int:
    """
    Compute bifurcation index B_t = floor( -log|c_t - c_*| / log δ )

    This counts how many renormalization layers deep we've fallen into
    the universality cascade. In the curvature-clock system, the "critical
    value" c_t is the current curvature κ_{d(x)}, and c_* = 0 represents
    the trivial fixed point.

    Args:
        x: Current state value
        c_star: Fixed critical value (default: 0.0)
        delta: Scaling factor (default: Feigenbaum constant 0.638)

    Returns:
        B_t: Bifurcation depth (renormalization layers)
    """
    # Current curvature as critical value
    c_t = digit_boundary_curvature(x)

    # Avoid log(0)
    diff = abs(c_t - c_star)
    if diff <= 1e-12:
        return 0

    # Compute bifurcation index
    if delta <= 0 or delta >= 1:
        # Fallback for invalid delta
        return digit_count(x)

    # B_t = floor( -log|κ_d - 0| / log δ )
    # Since κ_d can be positive or negative, we take absolute value
    # The negative sign makes deeper renormalization (smaller |κ_d|) give larger B_t
    try:
        log_ratio = -np.log(diff) / np.log(delta)
        B_t = int(np.floor(log_ratio))
        return max(0, B_t)  # Ensure non-negative
    except (ValueError, ZeroDivisionError):
        # Fallback to digit count if numerical issues
        return digit_count(x)
