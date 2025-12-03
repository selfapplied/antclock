"""
clock.antclock - AntClock: Pascal Curvature Clock System - Main Integration Module

This is the main AntClock system that orchestrates the specialized modules:
- pascal_core: Core mathematical primitives
- homology_engine: Topological and homology computations
- clock_mechanics: Walker implementations
- analysis_framework: Chaos analysis tools

Author: Joel
"""

from typing import Dict, Any, List
import numpy as np
from scipy.special import comb

# ============================================================================
# Import Specialized Modules
# ============================================================================

# Core mathematical primitives
from .pascal import (
    pascal_radius, pascal_curvature, digit_count, digit_boundary_curvature,
    count_digits, extract_ballast_and_units, compute_q_9_11,
    local_curvature_charge, clock_rate, bifurcation_index
)

# Homology and topological framework
from .homology import (
    pascal_simplicial_complex, betti_numbers_digit_shell, enhanced_betti_numbers,
    digit_boundary_filtration, bifurcation_from_persistence,
    Shell_n, Pascal_n, DigitHomologyComplex,
    WeightedBettiSum, CouplingLaw, RG_PersistentHomology
)

# Clock mechanics and walkers
from .mechanics import (
    jump_factor_curvature, jump_factor_constant,
    CE1_BifurcationHomology, create_ce1_operator,
    Row7DigitMirror, MirrorPhaseResonance, CurvatureCriticalRows,
    CurvatureClockWalker, Row7ActivatedWalker
)

# Analysis and measurement framework
from .analysis import (
    MemoryHistory, WitnessMeasurement, ChaosAnalysisFramework
)


# ============================================================================
# Main AntClock Interface
# ============================================================================

class AntClock:
    """
    AntClock: Complete Pascal Curvature Clock System

    This is the main interface to the AntClock system, providing access to:
    - Clock walkers and mechanics
    - Mathematical primitives
    - Homology computations
    - Chaos analysis tools
    """

    def __init__(self):
        """Initialize the complete AntClock system."""
        # Core components
        self.homology_engine = DigitHomologyComplex()
        self.analysis_framework = ChaosAnalysisFramework()

        # Default walker
        self.walker = CurvatureClockWalker()

    # ============================================================================
    # Core Clock Operations
    # ============================================================================

    def create_walker(self, x_0: int = 1, tau_0: float = 0.0, phi_0: float = 0.0,
                      chi_feg: float = 0.638, row7_activated: bool = False) -> CurvatureClockWalker:
        """
        Create a new clock walker.

        Args:
            x_0: Initial position
            tau_0: Initial clock phase
            phi_0: Initial angle
            chi_feg: Feigenbaum constant
            row7_activated: Whether to use Row7 symmetry breaking

        Returns:
            Configured CurvatureClockWalker instance
        """
        if row7_activated:
            return Row7ActivatedWalker(x_0, tau_0, phi_0, chi_feg)
        else:
            return CurvatureClockWalker(x_0, tau_0, phi_0, chi_feg)

    def run_trajectory(self, steps: int = 100, **walker_kwargs) -> dict:
        """
        Run a complete trajectory.

        Args:
            steps: Number of steps to run
            **walker_kwargs: Arguments for create_walker()

        Returns:
            Dictionary with trajectory data and analysis
        """
        # Create walker
        walker = self.create_walker(**walker_kwargs)

        # Run trajectory
        history, summary = walker.evolve(steps)

        # Analyze trajectory
        analysis = self.analysis_framework.analyze_single_trajectory({
            'id': 0,
            'states': history,
            'statistics': summary,
            'initial_conditions': walker_kwargs
        })

        return {
            'walker': walker,
            'history': history,
            'summary': summary,
            'analysis': analysis,
            'geometry': walker.get_geometry()
        }

    # ============================================================================
    # Mathematical Primitives
    # ============================================================================

    def pascal_analysis(self, n: int) -> dict:
        """
        Complete Pascal analysis for row n.

        Args:
            n: Pascal row index

        Returns:
            Dictionary with radius, curvature, and shell information
        """
        radius = pascal_radius(n)
        curvature = pascal_curvature(n)
        shell_digit = digit_count(10**n if n > 0 else 1)

        return {
            'row': n,
            'radius': radius,
            'curvature': curvature,
            'shell_digit': shell_digit,
            'binomial_center': int(comb(n, n//2, exact=False))
        }

    def digit_shell_analysis(self, x: float) -> dict:
        """
        Complete analysis of digit shell properties.

        Args:
            x: Number to analyze

        Returns:
            Dictionary with digit analysis
        """
        d = digit_count(x)
        K = digit_boundary_curvature(x)
        ballast_units = extract_ballast_and_units(x)
        B_t = bifurcation_index(x)

        return {
            'x': x,
            'digit_count': d,
            'boundary_curvature': K,
            'bifurcation_index': B_t,
            'ballast_units': ballast_units,
            'local_curvature_charge': local_curvature_charge(x),
            'clock_rate': clock_rate(x)
        }

    # ============================================================================
    # Homology and Topology
    # ============================================================================

    def homology_analysis(self, digit_shell: int) -> dict:
        """
        Complete homology analysis for a digit shell.

        Args:
            digit_shell: Shell to analyze

        Returns:
            Dictionary with homology information
        """
        betti = self.homology_engine.betti_numbers_X_r(digit_shell)
        filtration = self.homology_engine.filtration_X_r(digit_shell)

        return {
            'shell': digit_shell,
            'betti_numbers': betti,
            'filtration': filtration,
            'complex_size': len(filtration['vertices'])
        }

    def coupling_analysis(self, trajectories: list) -> dict:
        """
        Analyze coupling laws across trajectories.

        Args:
            trajectories: List of trajectory data

        Returns:
            Coupling law analysis
        """
        coupling = CouplingLaw()
        return coupling.verify_conservation(trajectories)

    # ============================================================================
    # Chaos Analysis
    # ============================================================================

    def analyze_trajectories(self, trajectories: list,
                           parameter_name: str = 'control_parameter') -> dict:
        """
        Complete chaos analysis of trajectory family.

        Args:
            trajectories: List of trajectory dictionaries
            parameter_name: Name of parameter varying across family

        Returns:
            Complete four-quadrant analysis
        """
        return self.analysis_framework.run_complete_analysis(trajectories, parameter_name)

    def bifurcation_study(self, parameter_range: tuple, steps_per_value: int = 50) -> dict:
        """
        Study bifurcation behavior across parameter range.

        Args:
            parameter_range: (min_param, max_param, num_values)
            steps_per_value: Steps per parameter value

        Returns:
            Bifurcation analysis results
        """
        min_param, max_param, num_values = parameter_range
        parameters = [min_param + i * (max_param - min_param) / (num_values - 1)
                     for i in range(num_values)]

        trajectories = []
        for param in parameters:
            # Run trajectory with this parameter (using it as chi_feg)
            result = self.run_trajectory(steps=steps_per_value, chi_feg=param)
            trajectory_data = {
                'id': len(trajectories),
                'states': result['history'],
                'statistics': result['summary'],
                'initial_conditions': {'control_parameter': param, 'chi_feg': param}
            }
            trajectories.append(trajectory_data)

        return self.analyze_trajectories(trajectories, 'control_parameter')

    # ============================================================================
    # Symmetry Breaking
    # ============================================================================

    def row7_analysis(self, x: int) -> dict:
        """
        Analyze Row7 digit morphism on a number.

        Args:
            x: Number to analyze

        Returns:
            Row7 morphism analysis
        """
        morphed = Row7DigitMirror.apply_to_number(x)
        analysis = Row7DigitMirror.analyze_digit_partition(x)

        return {
            'original': x,
            'morphed': morphed,
            'changed': x != morphed,
            'partition_analysis': analysis,
            'energy': Row7DigitMirror.morphism_energy(x)
        }

    def critical_curvature_analysis(self, n: int) -> dict:
        """
        Analyze curvature-critical properties of row n.

        Args:
            n: Row to analyze

        Returns:
            Critical curvature analysis
        """
        return CurvatureCriticalRows.get_row_properties(n)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_trajectory(steps: int = 50, **kwargs) -> dict:
    """
    Quick trajectory generation for testing.

    Args:
        steps: Number of steps
        **kwargs: Walker parameters

    Returns:
        Trajectory data
    """
    clock = AntClock()
    return clock.run_trajectory(steps, **kwargs)


def demo_pascal_curvature():
    """Demonstrate Pascal curvature analysis."""
    clock = AntClock()

    print("Pascal Curvature Analysis:")
    print("Row  Radius      Curvature   Shell")
    print("---  ----------  ----------  -----")

    for n in range(1, 8):
        analysis = clock.pascal_analysis(n)
        print("3d")

    print()


def demo_digit_shells():
    """Demonstrate digit shell analysis."""
    clock = AntClock()

    test_values = [1.0, 10.0, 100.0, 999.0]

    print("Digit Shell Analysis:")
    print("Value    Digits  Curvature  B_t  Clock_Rate")
    print("--------  ------  ---------  ---  ----------")

    for x in test_values:
        analysis = clock.digit_shell_analysis(x)
        print(".1f")

    print()


if __name__ == "__main__":
    print("AntClock System Demonstration")
    print("=" * 40)

    # Run demonstrations
    demo_pascal_curvature()
    demo_digit_shells()

    # Quick trajectory
    print("Running quick trajectory...")
    trajectory = quick_trajectory(steps=20)
    print(f"Trajectory completed: {len(trajectory['history'])} steps")
    print(".3f")
    print(f"Analysis: Lyapunov = {trajectory['analysis']['lyapunov_exponent']:.3f}")

    print("\nAntClock system ready! ✨")
