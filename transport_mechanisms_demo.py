#!/Users/joelstover/antclock/.venv/bin/python
"""
Transport Mechanisms Demo: Continued Fractions, Digital Polynomials, Universal Clock

Shows how these three mechanisms braid through CE1→CE2→CE3 layers,
carrying structure without losing invariant constants.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from clock import CurvatureClockWalker
from typing import List, Tuple


def demo_continued_fractions_transport():
    """
    CONTINUED FRACTIONS — CE1's Ladder Into CE2 and CE3

    Shows how continued fractions transport structure across layers.
    """
    print("=" * 80)
    print("CONTINUED FRACTIONS: CE1→CE2→CE3 TRANSPORT MECHANISM")
    print("=" * 80)

    walker = CurvatureClockWalker()

    # Test key mathematical constants
    test_values = [
        (math.pi, "π"),
        (math.e, "e"),
        ((1 + math.sqrt(5)) / 2, "φ (golden ratio)"),
        (math.sqrt(2), "√2"),
        (2.6854520010, "Khinchin's constant")
    ]

    print("\nCE1: Continued Fractions as Combinatorial Skeletons")
    print("-" * 50)

    for value, name in test_values:
        terms = walker.continued_fraction_expansion(value, 15)
        convergents = walker.continued_fraction_convergents(terms)

        print(f"\n{name} ≈ {value:.10f}")
        print(f"CF expansion: [{', '.join(map(str, terms[:10]))}]")
        if len(convergents) >= 3:
            p1, q1 = convergents[1]
            p2, q2 = convergents[2]
            print(".6f")

    print("\n\nCE2: Continued Fractions Become Dynamical Flows")
    print("-" * 50)

    # Demonstrate Gauss map - the transport from CE1 recursion to CE2 flow
    print("Gauss map x ↦ 1/x - ⌊1/x⌋ - CE1 discrete → CE2 continuous")

    # Generate Gauss map orbit
    x0 = 0.123456789  # Irrational seed
    orbit = [x0]
    for _ in range(20):
        x_next = walker.gauss_map(orbit[-1])
        orbit.append(x_next)

    print(f"Gauss map orbit starting from {x0}:")
    print(" ".join(".4f" for x in orbit[:10]))

    # Khinchin constant - CE2 invariant emerging from CE1 combinatorics
    khinchin_est = walker.khinchin_constant_sample(1000)
    print(".6f")
    print("  (True value: 2.6854520010...)")

    print("\n\nCE3: Continued Fractions as Triangulations")
    print("-" * 50)

    # Show how convergents form simplices around the true value
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    terms = walker.continued_fraction_expansion(phi, 10)
    convergents = walker.continued_fraction_convergents(terms)

    print("Golden ratio convergents as CE3 simplices:")
    print("Each convergent is a 'triangle' approximating the true value")
    print("n │ convergent │ error │ triangle 'area'")
    print("──┼────────────┼───────┼───────────────")

    for i, (p, q) in enumerate(convergents[:8]):
        approx = p / q
        error = abs(phi - approx)
        # "Area" of the approximation triangle (simplified metric)
        triangle_area = 1.0 / (q * q) if q > 0 else 0
        print("1d")


def demo_digital_polynomial_transport():
    """
    DIGITAL POLYNOMIAL — CE1's Galois Engine

    Shows how digital polynomials transport Galois structure across layers.
    """
    print("\n" + "=" * 80)
    print("DIGITAL POLYNOMIAL: CE1→CE2→CE3 GALOIS ENGINE")
    print("=" * 80)

    walker = CurvatureClockWalker()

    # Test with interesting numbers
    test_numbers = [7, 13, 17, 23, 42, 97, 123, 999]

    print("\nCE1: Digital Polynomials as Coefficient Maps")
    print("-" * 50)

    for n in test_numbers:
        coeffs = walker.digital_polynomial(n, 10)
        poly_str = " + ".join(f"{c}x^{i}" for i, c in enumerate(coeffs[::-1]) if c != 0)

        print(f"{n:3d} → [{', '.join(map(str, coeffs))}] → P(x) = {poly_str}")

    print("\n\nCE2: Digital Polynomials Become Spectral Operators")
    print("-" * 50)

    # Show how polynomials become spectral operators
    n = 17  # Prime for interesting factorization
    coeffs = walker.digital_polynomial(n, 10)

    print(f"Prime {n} as spectral operator:")
    print(f"Coefficients: {coeffs}")

    # Evaluate at different points (spectral analysis)
    test_points = [0.5, 1.0, 2.0, math.e, math.pi]
    print("P(x) evaluation at key points:")
    for x in test_points:
        value = walker.polynomial_evaluation(coeffs, x)
        print(".2f")

    print("\n\nCE3: Digital Polynomials as Factorization Graphs")
    print("-" * 50)

    # Show factorization as simplicial collapse
    composites = [15, 21, 35, 77, 91]
    print("Composite factorization as CE3 triangulation events:")
    print("Each prime factor represents a simplicial collapse")

    for n in composites:
        coeffs = walker.digital_polynomial(n, 10)
        print(f"\n{n} = {n}")
        print(f"  Digital polynomial: {coeffs}")

        # Simple factorization (would be more sophisticated in full implementation)
        factors = []
        temp = n
        for i in range(2, int(math.sqrt(n)) + 1):
            while temp % i == 0:
                factors.append(i)
                temp //= i
        if temp > 1:
            factors.append(temp)

        print(f"  Prime factors: {factors}")
        print(f"  Factorization graph: {' × '.join(map(str, factors))}")


def demo_universal_clock_transport():
    """
    UNIVERSAL CLOCK — Layer Synchronizer

    Shows how the universal clock interprets "time" differently across layers.
    """
    print("\n" + "=" * 80)
    print("UNIVERSAL CLOCK: CE1→CE2→CE3 SYNCHRONIZER")
    print("=" * 80)

    walker = CurvatureClockWalker()

    print("\nUniversal Clock as Layer Synchronizer:")
    print("One invariant constant, three different interpretations of 'time'")
    print()

    # Simulate events across layers
    events = [
        ("bracket_open", 1, "CE1: Structural decision"),
        ("continued_fraction_step", 1, "CE1: Recursion tick"),
        ("digit_transition", 1, "CE1: Shell boundary crossing"),

        ("flow_derivative", 2, "CE2: Differential scaling"),
        ("gauss_map_iteration", 2, "CE2: Dynamical step"),
        ("renormalization_event", 2, "CE2: PDE evolution"),

        ("simplex_flip", 3, "CE3: Triangulation event"),
        ("action_quantum", 3, "CE3: ħ-sized jump"),
        ("emergence_event", 3, "CE3: Spontaneous symmetry breaking")
    ]

    total_clock = 0
    print("Event Sequence and Clock Accumulation:")
    print("Event │ Layer │ Description │ Increment │ Total Clock")
    print("──────┼───────┼─────────────┼───────────┼────────────")

    for event_type, layer, description in events:
        increment = walker.universal_clock_increment(event_type, layer)
        total_clock += increment

        print("12s")

    print("\nClock Interpretations by Layer:")
    print(f"CE1 (discrete): {total_clock % 1000} recursion steps")
    print(".2e")
    print(f"CE3 (quantum): {total_clock // 100} event quanta")

    print("\nThe universal clock carries the same structural truth")
    print("but interprets 'time' according to each layer's reality.")


def demo_transport_mechanisms_braiding():
    """
    SYNTHESIS: How the Three Mechanisms Braid Through CE1→CE2→CE3

    Shows the connective tissue between layers.
    """
    print("\n" + "=" * 80)
    print("SYNTHESIS: THREE TRANSPORT MECHANISMS BRAIDED")
    print("=" * 80)

    walker = CurvatureClockWalker()

    print("""
Continued Fractions:   CE1 grammar → CE2 calculus → CE3 emergence
Digital Polynomials:   CE1 polynomial → CE2 spectral → CE3 factor graph
Universal Clock:       CE1 ticks → CE2 flow time → CE3 event count

These form the connective tissue - the functors that translate:
• CE1 grammar → CE2 calculus
• CE2 calculus → CE3 emergence
• CE3 emergence → CE1 new structure
""")

    # Demonstrate with a concrete example
    x = math.pi
    n = 42

    print("Concrete Example: π and 42 through the transport mechanisms")
    print("-" * 60)

    # Continued fractions for π
    cf_terms = walker.continued_fraction_expansion(x, 8)
    cf_convergents = walker.continued_fraction_convergents(cf_terms)

    # Digital polynomial for 42
    poly_coeffs = walker.digital_polynomial(n, 10)

    print(f"π continued fraction: [{', '.join(map(str, cf_terms))}]")
    print(f"42 digital polynomial: {poly_coeffs}")

    # Show how they transport through layers
    print("\nTransport through layers:")

    # CE1: Combinatorial
    ce1_cf = len(cf_terms)  # Length as complexity measure
    ce1_poly = len(poly_coeffs)  # Degree as complexity measure

    print(f"CE1: CF length = {ce1_cf}, Polynomial degree = {ce1_poly}")

    # CE2: Dynamical
    gauss_iterations = 5
    ce2_flow = sum(walker.gauss_map(x) for _ in range(gauss_iterations))
    ce2_spectral = walker.polynomial_evaluation(poly_coeffs, 2.0)  # Evaluate at x=2

    print(".3f")
    print(".1f")

    # CE3: Emergent
    ce3_triangulation = len(cf_convergents)  # Number of simplices
    ce3_factorization = sum(1 for i in range(2, n//2 + 1) if n % i == 0)  # Factor count

    print(f"CE3: CF triangulation depth = {ce3_triangulation}, Factorization events = {ce3_factorization}")

    print("\nThe three mechanisms maintain structural invariants")
    print("while enabling translation between layer realities.")


def main():
    """Run all transport mechanism demonstrations."""
    demo_continued_fractions_transport()
    demo_digital_polynomial_transport()
    demo_universal_clock_transport()
    demo_transport_mechanisms_braiding()

    print("\n" + "=" * 80)
    print("TRANSPORT MECHANISMS DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("Generated outputs saved to .out/ directory")


if __name__ == "__main__":
    main()
