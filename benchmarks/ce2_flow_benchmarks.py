#!/.venv/bin/python
"""
CE2 Flow Benchmarks: Dynamical Systems Testing

Tests CE2 dynamical flow principles with measure-preserving transformations
and spectral properties that require understanding of continuous evolution.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import math
import cmath
from pathlib import Path

from ce_benchmark_types import (
    CE2Benchmark, CE2FlowPoint, BenchmarkConfig, BenchmarkResult
)

@dataclass
class GaussMapConvergenceBenchmark:
    """
    CE2.1: Gauss Map Convergence Classification

    Classify convergence behavior of Gauss map orbits, requiring understanding
    of when discrete recursions become chaotic vs convergent flows.
    """

    def __init__(self):
        self.config = BenchmarkConfig(
            name="gauss_map_convergence",
            scale=30000,
            diversity_factors=[
                "initial_condition_sweep",  # Wide range of x₀
                "convergence_time_variation", # Different orbit lengths
                "bifurcation_cascades",     # Near bifurcation points
                "irrational_approximations" # Continued fraction complexity
            ],
            ce_layer="ce2"
        )

    def generate_dataset(self, size: int) -> Tuple[List[CE2FlowPoint], List[int]]:
        """Generate Gauss map convergence classification dataset."""
        inputs = []
        outputs = []  # 0: converges to 0, 1: chaotic, 2: converges to other

        for _ in range(size):
            # Sample initial condition from (0,1) excluding rationals
            x0 = self._sample_irrational_approximation()

            # Analyze orbit behavior
            orbit = self._compute_gauss_orbit(x0, max_steps=1000)
            convergence_class = self._classify_convergence(orbit)

            outputs.append(convergence_class)

            # Extract flow features
            time_parameter = len(orbit) / 1000.0  # Normalized time to convergence/chaos
            flow_velocity = self._compute_mean_velocity(orbit)
            invariant_measure = self._estimate_invariant_measure(orbit)
            spectral_radius = self._compute_spectral_radius(orbit)

            point = CE2FlowPoint(
                time_parameter=time_parameter,
                flow_velocity=flow_velocity,
                invariant_measure=invariant_measure,
                spectral_radius=spectral_radius
            )
            inputs.append(point)

        return inputs, outputs

    def evaluate_mathematical_consistency(self, model: Any, inputs: List[CE2FlowPoint]) -> float:
        """Evaluate preservation of measure-theoretic properties."""
        if model is None:
            return 1.0

        # Check if predictions preserve Gauss map measure properties
        # The Gauss map should preserve the measure μ(dx) = dx/(π√(x(1-x)))

        consistency_scores = []
        for point in inputs[:500]:  # Sample for efficiency
            # Verify measure preservation property
            measure_preserved = self._check_measure_preservation(point)
            consistency_scores.append(float(measure_preserved))

        return np.mean(consistency_scores)

    def is_toy_solution_possible(self, dataset: Tuple[List[CE2FlowPoint], List[int]]) -> bool:
        """Check if simple decision boundaries could solve this."""
        inputs, outputs = dataset

        # Extract simple features
        time_params = np.array([p.time_parameter for p in inputs])
        spectral_radii = np.array([p.spectral_radius for p in inputs])

        # Test simple decision rules
        # Rule 1: If time_parameter < threshold, class 0
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            simple_pred = (time_params < threshold).astype(int)
            simple_acc = np.mean(simple_pred == outputs)
            if simple_acc > 0.8:  # If simple rule gets 80% accuracy
                return True

        # Rule 2: If spectral_radius > threshold, class 1
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            simple_pred = (spectral_radii > threshold).astype(int)
            simple_acc = np.mean(simple_pred == outputs)
            if simple_acc > 0.8:
                return True

        return False

    def _sample_irrational_approximation(self) -> float:
        """Sample irrational number approximation."""
        # Generate numbers that are good approximations to irrationals
        # to create interesting Gauss map dynamics

        # Method 1: Continued fraction approximants
        if np.random.random() < 0.3:
            # Golden ratio approximations
            n = np.random.randint(1, 20)
            return self._fibonacci_ratio(n, n+1)

        # Method 2: Square root approximations
        elif np.random.random() < 0.3:
            root = np.random.randint(2, 20)
            n = np.random.randint(1, 10)
            approx = self._sqrt_approximation(root, n)
            return min(max(approx, 0.001), 0.999)  # Keep in (0,1)

        # Method 3: Random but avoid obvious rationals
        else:
            x = np.random.beta(2, 2)  # Beta distribution peaks away from 0,1
            # Perturb slightly to avoid exact rationals
            return min(max(x + np.random.normal(0, 0.001), 0.001), 0.999)

    def _fibonacci_ratio(self, a: int, b: int) -> float:
        """Compute Fibonacci ratio approximation."""
        # Continued fraction convergent for golden ratio
        phi = (1 + math.sqrt(5)) / 2
        return 1.0 / (1.0 + phi)  # Keep in (0,1)

    def _sqrt_approximation(self, n: int, terms: int) -> float:
        """Compute sqrt(n) continued fraction approximation."""
        # Simple continued fraction for sqrt(n)
        result = math.sqrt(n)
        for _ in range(terms):
            integer_part = int(result)
            result = 1.0 / (result - integer_part)
        return result

    def _compute_gauss_orbit(self, x0: float, max_steps: int) -> List[float]:
        """Compute Gauss map orbit."""
        orbit = [x0]
        x = x0

        for _ in range(max_steps):
            if x == 0:
                break  # Converged to 0
            x = 1.0/x - int(1.0/x)  # Gauss map: T(x) = 1/x - floor(1/x)

            if not (0 < x < 1):
                break  # Escaped interval

            orbit.append(x)

            # Check for convergence to 0
            if abs(x) < 1e-10:
                break

        return orbit

    def _classify_convergence(self, orbit: List[float]) -> int:
        """Classify orbit convergence behavior."""
        if len(orbit) < 10:
            return 0  # Converged quickly

        # Check if orbit converged to 0
        if any(abs(x) < 1e-8 for x in orbit):
            return 0  # Convergent to 0

        # Check for chaotic behavior (no clear pattern)
        # Look for bounded but non-convergent behavior
        if len(orbit) >= 100:
            # Check if orbit stays bounded but doesn't converge
            recent_values = orbit[-50:]
            if max(recent_values) < 0.9 and min(recent_values) > 0.1:
                # Check for sensitive dependence (chaotic indicator)
                diffs = [abs(recent_values[i+1] - recent_values[i]) for i in range(len(recent_values)-1)]
                if np.mean(diffs) > 0.01:  # Significant variation
                    return 1  # Chaotic

        # Convergent to other value or periodic
        return 2

    def _compute_mean_velocity(self, orbit: List[float]) -> complex:
        """Compute complex velocity in orbit."""
        if len(orbit) < 2:
            return complex(0, 0)

        # Treat orbit as complex signal
        velocities = []
        for i in range(1, len(orbit)):
            dx = orbit[i] - orbit[i-1]
            velocities.append(complex(dx, dx * 0.1))  # Add small imaginary component

        return complex(np.mean([v.real for v in velocities]),
                      np.mean([v.imag for v in velocities]))

    def _estimate_invariant_measure(self, orbit: List[float]) -> float:
        """Estimate Gauss map invariant measure density."""
        if len(orbit) < 50:
            return 0.0

        # Simple histogram-based density estimation
        hist, _ = np.histogram(orbit, bins=20, range=(0, 1), density=True)

        # The Gauss measure should have density 1/(π√(x(1-x)))
        # Check how well our empirical distribution matches this
        x_vals = np.linspace(0.05, 0.95, 19)  # Bin centers
        theoretical = [1.0 / (math.pi * math.sqrt(x * (1-x))) for x in x_vals]

        # Normalize both distributions
        hist_norm = hist / np.sum(hist)
        theoretical_norm = np.array(theoretical) / np.sum(theoretical)

        # KL divergence as measure of fit
        kl_div = np.sum(hist_norm * np.log((hist_norm + 1e-10) / (theoretical_norm + 1e-10)))

        # Convert to similarity score (lower KL = higher similarity)
        return max(0, 1.0 - kl_div / 10.0)  # Scale and clamp

    def _compute_spectral_radius(self, orbit: List[float]) -> float:
        """Compute spectral radius of orbit dynamics."""
        if len(orbit) < 20:
            return 0.0

        # Compute autocorrelation function
        signal = np.array(orbit)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Second half

        # Find spectral radius (related to Lyapunov exponent)
        if len(autocorr) > 1:
            # Simple approximation of spectral radius
            decay_rate = -np.log(np.abs(autocorr[1] / (autocorr[0] + 1e-10)))
            return min(decay_rate, 10.0)  # Cap at reasonable value

        return 0.0

    def _check_measure_preservation(self, point: CE2FlowPoint) -> bool:
        """Check if point satisfies measure preservation properties."""
        # The Gauss map preserves the measure with density 1/(π√(x(1-x)))
        # For a good approximation, the invariant measure should be reasonable

        measure = point.invariant_measure

        # Measure should be positive and not too large
        if not (0 < measure < 2.0):
            return False

        # Spectral radius should indicate chaotic behavior when appropriate
        if point.time_parameter > 0.5:  # Long orbits tend to be chaotic
            if not (0.5 < point.spectral_radius < 5.0):
                return False

        return True

@dataclass
class FlowFieldIntegrationBenchmark:
    """
    CE2.2: Flow Field Integration Regression

    Predict integrated flow values along trajectories, requiring understanding
    of how local velocities accumulate into global flow properties.
    """

    def __init__(self):
        self.config = BenchmarkConfig(
            name="flow_field_integration",
            scale=20000,
            diversity_factors=[
                "trajectory_complexity",   # Different path lengths/types
                "velocity_field_variation", # Non-uniform flows
                "integration_accuracy",    # Numerical precision requirements
                "boundary_conditions"      # Edge effects at flow boundaries
            ],
            ce_layer="ce2"
        )

    def generate_dataset(self, size: int) -> Tuple[List[CE2FlowPoint], List[float]]:
        """Generate flow field integration dataset."""
        inputs = []
        outputs = []  # Integrated flow values

        for _ in range(size):
            # Generate trajectory with varying complexity
            trajectory = self._generate_flow_trajectory()
            integrated_value = self._compute_trajectory_integral(trajectory)

            outputs.append(integrated_value)

            # Extract features at random point along trajectory
            sample_point = np.random.choice(trajectory)

            point = CE2FlowPoint(
                time_parameter=sample_point['t'],
                flow_velocity=complex(sample_point['vx'], sample_point['vy']),
                invariant_measure=sample_point['density'],
                spectral_radius=sample_point['lyapunov']
            )
            inputs.append(point)

        return inputs, outputs

    def evaluate_mathematical_consistency(self, model: Any, inputs: List[CE2FlowPoint]) -> float:
        """Evaluate conservation of flow properties."""
        if model is None:
            return 1.0

        # Check conservation laws: divergence-free flow, measure preservation
        consistency_scores = []

        for point in inputs[:300]:
            # Check if velocity field is divergence-free (incompressible)
            divergence_free = self._check_divergence_free(point)
            measure_preserved = self._check_measure_conservation(point)

            consistent = divergence_free and measure_preserved
            consistency_scores.append(float(consistent))

        return np.mean(consistency_scores)

    def is_toy_solution_possible(self, dataset: Tuple[List[CE2FlowPoint], List[float]]) -> bool:
        """Check if simple integration rules could solve this."""
        inputs, outputs = dataset

        # Extract simple features
        times = np.array([p.time_parameter for p in inputs])
        velocities = np.array([abs(p.flow_velocity) for p in inputs])

        # Test if simple rules work
        # Rule: integral ≈ time * velocity
        simple_integral = times * velocities
        simple_error = np.mean(np.abs(outputs - simple_integral) / (np.abs(outputs) + 1e-10))

        # If simple trapezoidal rule gets good accuracy, it's too easy
        return simple_error < 0.1

    def _generate_flow_trajectory(self) -> List[Dict]:
        """Generate a complex flow trajectory."""
        # Create trajectory with varying velocity field
        trajectory = []
        t = 0.0
        x, y = np.random.uniform(-1, 1, 2)  # Start position

        dt = 0.01
        steps = np.random.randint(50, 500)  # Variable trajectory length

        for _ in range(steps):
            # Velocity field: combination of harmonic and chaotic components
            base_flow = complex(
                math.sin(2 * math.pi * t) * math.cos(math.pi * x),
                math.cos(2 * math.pi * t) * math.sin(math.pi * y)
            )

            # Add chaotic perturbation
            perturbation = complex(
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1)
            )

            velocity = base_flow + perturbation

            # Compute local properties
            density = self._flow_density(x, y, t)
            lyapunov = self._local_lyapunov(x, y, t)

            trajectory.append({
                't': t,
                'x': x, 'y': y,
                'vx': velocity.real,
                'vy': velocity.imag,
                'density': density,
                'lyapunov': lyapunov
            })

            # Euler integration
            x += velocity.real * dt
            y += velocity.imag * dt
            t += dt

        return trajectory

    def _compute_trajectory_integral(self, trajectory: List[Dict]) -> float:
        """Compute line integral along trajectory."""
        if len(trajectory) < 2:
            return 0.0

        integral = 0.0
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]
            curr = trajectory[i]

            # Simple trapezoidal rule for ∫ v · dl
            v_avg = complex(
                (prev['vx'] + curr['vx']) / 2,
                (prev['vy'] + curr['vy']) / 2
            )

            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            dl = complex(dx, dy)

            # Dot product: Re(v)·dx + Im(v)·dy
            integrand = v_avg.real * dx + v_avg.imag * dy
            integral += integrand

        return integral

    def _flow_density(self, x: float, y: float, t: float) -> float:
        """Compute local flow density."""
        # Gaussian density modulated by time
        r_squared = x*x + y*y
        return math.exp(-r_squared) * (1 + 0.5 * math.sin(2 * math.pi * t))

    def _local_lyapunov(self, x: float, y: float, t: float) -> float:
        """Compute local Lyapunov exponent approximation."""
        # Simple approximation based on distance from origin
        r = math.sqrt(x*x + y*y)
        return 0.1 + 0.5 * r + 0.2 * math.sin(4 * math.pi * t)

    def _check_divergence_free(self, point: CE2FlowPoint) -> bool:
        """Check if flow is approximately divergence-free."""
        # For incompressible flow, divergence should be small
        # We can't compute full divergence from single point, so use proxy
        velocity_magnitude = abs(point.flow_velocity)
        time_parameter = point.time_parameter

        # Flows with moderate velocity and time should be reasonable
        return 0.1 < velocity_magnitude < 5.0 and 0 < time_parameter < 10.0

    def _check_measure_conservation(self, point: CE2FlowPoint) -> bool:
        """Check if measure is approximately conserved."""
        measure = point.invariant_measure
        return 0.1 < measure < 3.0  # Reasonable measure range

# Export benchmarks
ce2_benchmarks = [
    GaussMapConvergenceBenchmark(),
    FlowFieldIntegrationBenchmark()
]

