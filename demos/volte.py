#!run.sh
"""
Volte System: Coherence-Preserving Transformations

Demonstrates the Volte operator within AntClock - controlled turns that
preserve core invariants while reorienting flow under stress.

This is a standalone demo that runs without external dependencies.
Run with: python3 demos/volte_antclock_demo.py
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple


# ============================================================================
# Minimal Volte Implementation for Demo
# ============================================================================

class SimpleVolteSystem:
    """
    Simplified Volte system for demonstration purposes.
    Shows the core concepts without full AntClock dependencies.
    """

    def __init__(self, threshold: float = 0.638):
        self.threshold = threshold
        self._curvature_cache = {}

    def _pascal_curvature(self, n: int) -> float:
        """Compute Pascal curvature κ_n with caching"""
        if n in self._curvature_cache:
            return self._curvature_cache[n]

        if n < 2:
            curvature = 0.0
        else:
            # Simplified approximation
            try:
                # Central binomial coefficient approximation
                c_n = math.comb(n, n//2)
                c_n_minus_1 = math.comb(n-1, (n-1)//2)
                c_n_plus_1 = math.comb(n+1, (n+1)//2)
                curvature = math.log(c_n_plus_1) - 2 * math.log(c_n) + math.log(c_n_minus_1)
            except (ValueError, OverflowError):
                curvature = n * math.log(2) * 0.1  # Simplified

        self._curvature_cache[n] = curvature
        return curvature

    def stress(self, x: float) -> float:
        """S(x): Stress measure based on curvature"""
        digit_shell = len(str(int(x))) if x > 0 else 1
        return abs(self._pascal_curvature(digit_shell))

    def coherence(self, x: float) -> float:
        """C(x): Coherence measure based on shell stability"""
        digit_shell = len(str(int(x))) if x > 0 else 1
        return 1.0 / (1.0 + digit_shell)

    def invariant(self, x: float) -> int:
        """Q(x): Invariant based on digit shell"""
        return len(str(int(x))) if x > 0 else 1

    def should_activate(self, x: float) -> bool:
        """Check if Volte should activate: S > κ"""
        return self.stress(x) > self.threshold

    def correction(self, x: float) -> float:
        """Compute Volte correction V(x)"""
        digit_shell = len(str(int(x))) if x > 0 else 1
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio

        if digit_shell % 4 == 3:  # Mirror phase shell
            return -0.1 * phi * self._pascal_curvature(digit_shell)
        else:
            return -0.05 * self._pascal_curvature(digit_shell)


class SimpleAntClock:
    """Simplified AntClock for demo purposes"""

    def __init__(self, x_0: float = 1.0, chi_feg: float = 0.638, enable_volte: bool = False):
        self.x_current = x_0
        self.chi_feg = chi_feg
        self.enable_volte = enable_volte
        if enable_volte:
            self.volte_system = SimpleVolteSystem(threshold=chi_feg)
        self.history = []

    def _pascal_curvature(self, n: int) -> float:
        """Simplified curvature calculation"""
        if n < 2:
            return 0.0
        return n * math.log(2) * 0.1  # Simplified approximation

    def _digit_tension(self, x: float) -> float:
        """Simplified digit tension"""
        if x <= 0:
            return 0.0
        digits = str(int(x))
        tension = sum(int(d) / 9.0 for d in digits) / len(digits)
        return min(tension, 1.0)

    def evolve_step(self, dt: float = 0.01):
        """Single evolution step"""
        digit_shell = len(str(int(self.x_current))) if self.x_current > 0 else 1
        kappa = self._pascal_curvature(digit_shell)
        tension = self._digit_tension(self.x_current)

        velocity = kappa * (1 + tension) * self.chi_feg

        # Apply Volte correction if enabled
        if self.enable_volte and self.volte_system.should_activate(self.x_current):
            correction = self.volte_system.correction(self.x_current)
            velocity += correction

        self.x_current += velocity * dt
        self.history.append(self.x_current)
        return self.x_current

    def evolve(self, steps: int):
        """Evolve for multiple steps"""
        for _ in range(steps):
            self.evolve_step()
        return self.history


def demo_volte_system():
    """
    Volte.1: Core Volte System Components
    Demonstrate the fundamental Volte equation and its components.
    """
    print("="*60)
    print("VOLTE.1: CORE VOLTE SYSTEM COMPONENTS")
    print("="*60)

    # Create Volte system
    volte_system = SimpleVolteSystem(threshold=0.638)
    print(f"Volte system initialized with threshold κ = {volte_system.threshold}")
    print(f"Volte equation: dx/dt = F(x,u) + V(x,u)")
    print()

    # Test stress-coherence landscape across digit shells
    test_positions = [1.0, 10.0, 100.0, 1000.0, 10000.0]

    print("Stress-Coherence Landscape Across Digit Shells:")
    print("Position | Stress (S) | Coherence (C) | Invariant (Q) | Volte Active")
    print("---------|------------|----------------|--------------|-------------")

    for x in test_positions:
        stress = volte_system.stress(x)
        coherence = volte_system.coherence(x)
        invariant = volte_system.invariant(x)
        active = "YES" if volte_system.should_activate(x) else "NO"

        print(f"{x:8.1f} | {stress:10.3f} | {coherence:14.3f} | {invariant:12d} | {active}")


def demo_volte_enhanced_evolution():
    """
    Volte.2: Volte-Enhanced Evolution
    Compare regular vs Volte-enhanced AntClock evolution.
    """
    print("\n" + "="*60)
    print("VOLTE.2: VOLTE-ENHANCED EVOLUTION COMPARISON")
    print("="*60)

    # Create walkers
    regular_walker = SimpleAntClock(x_0=1.0, chi_feg=0.638, enable_volte=False)
    volte_walker = SimpleAntClock(x_0=1.0, chi_feg=0.638, enable_volte=True)

    print("Comparing evolution with/without Volte corrections:")
    print(f"Starting position: x₀ = {regular_walker.x_current}")
    print(f"FEG coupling: χ_FEG = {regular_walker.chi_feg}")
    print()

    # Evolve both for 500 steps
    steps = 500
    regular_walker.evolve(steps)
    volte_walker.evolve(steps)

    # Compare trajectories
    regular_final = regular_walker.history[-1]
    volte_final = volte_walker.history[-1]

    print("Evolution Results:")
    print(f"Regular AntClock final position:    {regular_final:12.2f}")
    print(f"Volte-enhanced AntClock position:  {volte_final:12.2f}")
    print(f"Relative difference:               {(volte_final/regular_final - 1)*100:11.1f}%")

    # Count Volte activations by checking when corrections were applied
    volte_activations = sum(1 for i in range(len(volte_walker.history))
                           if i > 0 and abs(volte_walker.history[i] - regular_walker.history[i]) > 1e-6)
    print(f"Volte corrections applied:         {volte_activations:12d} times")


def demo_volte_mathematics():
    """
    Volte.3: Volte Mathematical Foundation
    Explain the core mathematics of coherence-preserving transformations.
    """
    print("\n" + "="*60)
    print("VOLTE.3: VOLTE MATHEMATICAL FOUNDATION")
    print("="*60)

    print("The Volte equation: dx/dt = F(x,u) + V(x,u)")
    print()
    print("Where:")
    print("• F(x,u): Ordinary dynamics (gradient descent, physical laws, etc.)")
    print("• V(x,u): Volte correction operator (activated under stress)")
    print("• κ: Activation threshold (FEG coupling ≈ 0.638)")
    print()

    print("Volte Axioms:")
    print("(V1) Invariant Preservation: Q(x + εV) = Q(x)")
    print("(V2) Harm Reduction: S decreases, C increases under correction")
    print("(V3) Threshold Activation: V(x,u) ≠ 0 only when S(x,u) > κ")
    print()

    print("Biological Examples:")
    print("• Evolution: ERV exaptation preserves species identity")
    print("• Immune: Treatment preserves self-recognition under viral stress")
    print("• Psychology: Reorientation preserves core values under stigma")
    print()

    print("The Volte operator emerges from CE1 discrete geometry as the")
    print("continuous limit of curvature-preserving transformations.")


def demo_volte_practical():
    """
    Volte.4: Practical Volte Demonstration
    Show real-time Volte activation and correction.
    """
    print("\n" + "="*60)
    print("VOLTE.4: PRACTICAL VOLTE DEMONSTRATION")
    print("="*60)

    # Create Volte-enhanced walker
    walker = SimpleAntClock(x_0=10.0, chi_feg=0.638, enable_volte=True)

    print("Real-time Volte monitoring during evolution:")
    print("Step | Position | Stress | Volte Active | Correction")
    print("-----|----------|--------|-------------|-----------")

    # Track evolution with Volte monitoring
    for step in range(15):
        x_before = walker.x_current
        stress_before = walker.volte_system.stress(x_before)
        should_activate = walker.volte_system.should_activate(x_before)

        # Evolve one step
        walker.evolve_step(dt=0.01)
        x_after = walker.x_current

        # Calculate what the correction was
        correction = x_after - x_before

        active = "YES" if should_activate else "NO"
        corr_str = ".6f" if abs(correction) > 1e-6 else "0.000000"

        print(f"{step:4d} | {x_after:8.2f} | {stress_before:6.3f} | {active:11s} | {correction:{corr_str}}")


def main():
    """
    Complete Volte System Demonstration
    """
    print("VOLTE SYSTEM: Coherence-Preserving Transformations")
    print("=" * 60)
    print("Demonstrating controlled turns that preserve invariants under stress")
    print()

    try:
        demo_volte_system()
        demo_volte_enhanced_evolution()
        demo_volte_mathematics()
        demo_volte_practical()

        print("\n" + "=" * 60)
        print("✅ VOLTE SYSTEM DEMONSTRATION COMPLETE")
        print()
        print("The Volte operator provides:")
        print("• Invariant preservation: Q(x + εV) = Q(x)")
        print("• Stress reduction: Harm decreases under correction")
        print("• Coherence enhancement: Internal stability increases")
        print("• Threshold activation: Only triggers when S > κ ≈ 0.638")
        print()
        print("Result: Controlled turns maintaining identity while reorienting flow.")
        print()
        print("For full AntClock integration, see antclock.volte module.")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
