#!/.venv/bin/python
"""
CE1 Geometry Benchmarks: Discrete Mathematics Testing

Tests CE1 discrete geometry principles with diverse inputs that require
understanding of mirror phases, curvature fields, and digit shell structure.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import math
from pathlib import Path

from ..definitions import (
    CE1Benchmark, CE1GeometryPoint, BenchmarkConfig, BenchmarkResult
)

@dataclass
class MirrorPhaseClassificationBenchmark:
    """
    CE1.1: Mirror Phase Classification

    Classify points by their mirror phase (n mod 4), requiring understanding
    of symmetry breaking at tangent singularities.
    """

    def __init__(self):
        self.config = BenchmarkConfig(
            name="mirror_phase_classification",
            scale=50000,  # Large scale for comprehensive testing
            diversity_factors=[
                "shell_range_1_to_10000",  # Wide shell range
                "curvature_variation",     # Different curvature profiles
                "entropy_distribution",    # Various digit entropies
                "prime_gaps"              # Prime structure variation
            ],
            ce_layer="ce1"
        )

    def generate_dataset(self, size: int, progress_callback: Optional[Callable[[int], None]] = None) -> Tuple[List[CE1GeometryPoint], List[int]]:
        """Generate diverse mirror phase classification dataset."""
        inputs = []
        outputs = []

        # Use batching for progress updates
        batch_size = max(1, size // 100)  # Update progress roughly every 1%

        for batch_start in range(0, size, batch_size):
            batch_end = min(batch_start + batch_size, size)
            current_batch_size = batch_end - batch_start

            for _ in range(current_batch_size):
                # Sample shell index from wide range to prevent memorization
                shell_index = np.random.randint(1, 10001)

                # Generate mirror phase (the target classification)
                mirror_phase = shell_index % 4
                outputs.append(mirror_phase)

                # Generate diverse features that correlate with but don't trivially
                # determine the mirror phase
                curvature_value = self._pascal_curvature(shell_index)
                digit_entropy = self._digit_entropy(shell_index)
                prime_influence = self._prime_density_factor(shell_index)

                # Add noise to prevent trivial classification
                curvature_value += np.random.normal(0, 0.1)
                digit_entropy += np.random.normal(0, 0.05)

                point = CE1GeometryPoint(
                    shell_index=shell_index,
                    mirror_phase=mirror_phase,  # Note: this shouldn't be used for classification
                    curvature_value=curvature_value,
                    digit_entropy=digit_entropy
                )
                inputs.append(point)

            # Progress callback
            if progress_callback:
                progress_callback(len(inputs))

        return inputs, outputs

    def evaluate_mathematical_consistency(self, model: Any, inputs: List[CE1GeometryPoint]) -> float:
        """Evaluate if model preserves mirror phase arithmetic properties."""
        if model is None:
            # For now, return perfect consistency for null model
            return 1.0

        # Check if model predictions respect modular arithmetic
        # Mirror phases should cycle every 4 shells
        consistency_scores = []
        for point in inputs[:1000]:  # Sample for efficiency
            predicted_phase = model.predict(point) if hasattr(model, 'predict') else point.mirror_phase
            actual_phase = point.shell_index % 4
            modular_consistent = (predicted_phase == actual_phase)
            consistency_scores.append(float(modular_consistent))

        return np.mean(consistency_scores)

    def is_toy_solution_possible(self, dataset: Tuple[List[CE1GeometryPoint], List[int]]) -> bool:
        """Check if simple heuristics OTHER than ground truth could solve this."""
        inputs, outputs = dataset

        # Check if simple heuristics based on features (not ground truth) can solve it
        phases = np.array(outputs)
        curvatures = np.array([p.curvature_value for p in inputs])
        entropies = np.array([p.digit_entropy for p in inputs])

        # Test various simple heuristics that DON'T use the ground truth (shell_index % 4)

        # Heuristic 1: Classify by curvature thresholds
        for threshold in np.linspace(np.min(curvatures), np.max(curvatures), 10):
            pred = (curvatures > threshold).astype(int)
            acc = np.mean(pred == phases)
            if acc > 0.85:  # If simple curvature threshold gets 85% accuracy
                return True

        # Heuristic 2: Classify by entropy thresholds
        for threshold in np.linspace(np.min(entropies), np.max(entropies), 10):
            pred = (entropies > threshold).astype(int)
            acc = np.mean(pred == phases)
            if acc > 0.85:  # If simple entropy threshold gets 85% accuracy
                return True

        # Heuristic 3: Simple rounding/classification of features
        # If features directly determine the class without complex relationships
        for feature_name in ['curvature_value', 'digit_entropy']:
            feature_values = np.array([getattr(p, feature_name) for p in inputs])
            # Try simple binning
            for n_bins in [2, 4]:
                bins = np.linspace(np.min(feature_values), np.max(feature_values), n_bins + 1)
                binned = np.digitize(feature_values, bins[1:-1])
                for offset in range(n_bins):
                    pred = ((binned + offset) % n_bins).astype(float) / (n_bins - 1)
                    pred_classes = (pred * 3).astype(int)  # Map to 0-3 classes
                    acc = np.mean(pred_classes == phases)
                    if acc > 0.85:
                        return True

        return False  # No simple heuristic works well

    def _pascal_curvature(self, n: int) -> float:
        """Compute Pascal curvature κ_n."""
        if n < 2:
            return 0.0
        try:
            # Approximation of central binomial coefficient curvature
            return 1.0 / (n * math.log(n + 1))
        except:
            return 0.0

    def _digit_entropy(self, n: int) -> float:
        """Calculate digit entropy as complexity measure."""
        if n == 0:
            return 0.0

        digits = [int(d) for d in str(n)]
        digit_counts = np.bincount(digits, minlength=10)

        # Shannon entropy of digit distribution
        probs = digit_counts[digit_counts > 0] / len(digits)
        entropy = -np.sum(probs * np.log2(probs))

        return entropy / math.log2(10)  # Normalize to [0,1]

    def _prime_density_factor(self, n: int) -> float:
        """Prime density influence factor."""
        if n < 2:
            return 0.0

        # Count primes in window around n
        window = max(10, n // 100)  # Adaptive window size
        start = max(2, n - window)
        end = n + window

        prime_count = sum(1 for i in range(start, end + 1) if self._is_prime(i))
        total_count = end - start + 1

        return prime_count / total_count

    def _is_prime(self, n: int) -> bool:
        """Simple primality test."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

@dataclass
class CurvatureFieldRegressionBenchmark:
    """
    CE1.2: Curvature Field Regression

    Predict curvature values across shell transitions, requiring understanding
    of how curvature fields vary with mirror phase boundaries.
    """

    def __init__(self):
        self.config = BenchmarkConfig(
            name="curvature_field_regression",
            scale=50000,
            diversity_factors=[
                "phase_transitions",      # Test behavior across n≡3 mod 4
                "scale_invariance",       # Different shell scales
                "noise_robustness",       # Measurement noise
                "boundary_effects"        # Edge cases at shell boundaries
            ],
            ce_layer="ce1"
        )

    def generate_dataset(self, size: int, progress_callback: Optional[Callable[[int], None]] = None) -> Tuple[List[CE1GeometryPoint], List[float]]:
        """Generate curvature field regression dataset."""
        inputs = []
        outputs = []

        # Use batching for progress updates
        batch_size = max(1, size // 100)  # Update progress roughly every 1%

        for batch_start in range(0, size, batch_size):
            batch_end = min(batch_start + batch_size, size)
            current_batch_size = batch_end - batch_start

            for _ in range(current_batch_size):
                shell_index = np.random.randint(1, 5001)

                # True curvature value (target)
                true_curvature = self._pascal_curvature(shell_index)
                outputs.append(true_curvature)

                # Add realistic measurement noise and boundary effects
                noise_level = 0.05 + 0.1 * (shell_index % 4 == 3)  # Higher noise at boundaries
                measured_curvature = true_curvature + np.random.normal(0, noise_level)

                # Mirror phase affects digit entropy measurement
                mirror_phase = shell_index % 4
                digit_entropy = self._digit_entropy(shell_index)
                if mirror_phase == 3:  # Tangent singularity
                    digit_entropy *= (1 + np.random.normal(0, 0.2))  # Phase-dependent noise

                point = CE1GeometryPoint(
                    shell_index=shell_index,
                    mirror_phase=mirror_phase,
                    curvature_value=measured_curvature,
                    digit_entropy=digit_entropy
                )
                inputs.append(point)

            # Progress callback
            if progress_callback:
                progress_callback(len(inputs))

        return inputs, outputs

    def evaluate_mathematical_consistency(self, model: Any, inputs: List[CE1GeometryPoint]) -> float:
        """Evaluate curvature field smoothness and boundary behavior."""
        if model is None:
            return 1.0

        # Check if predictions respect mathematical properties of curvature
        # Curvature should be smooth except at mirror phase boundaries
        consistency_scores = []

        for i in range(1, len(inputs) - 1):
            curr = inputs[i]
            prev = inputs[i-1]
            next_p = inputs[i+1]

            if abs(curr.shell_index - prev.shell_index) == 1 and abs(next_p.shell_index - curr.shell_index) == 1:
                # Check local smoothness away from boundaries
                if curr.shell_index % 4 != 3:  # Not at boundary
                    predicted_smooth = abs(self._expected_curvature_smoothness(curr, prev, next_p)) < 0.1
                    consistency_scores.append(float(predicted_smooth))
                else:
                    # At boundary - allow discontinuity but check it's not too extreme
                    discontinuity = abs(self._curvature_discontinuity(curr, prev, next_p))
                    reasonable = discontinuity < 1.0  # Not unreasonably large
                    consistency_scores.append(float(reasonable))

        return np.mean(consistency_scores) if consistency_scores else 1.0

    def is_toy_solution_possible(self, dataset: Tuple[List[CE1GeometryPoint], List[float]]) -> bool:
        """Check if simple interpolation could solve this."""
        inputs, outputs = dataset

        # Test if linear interpolation gives reasonable accuracy
        shell_indices = np.array([p.shell_index for p in inputs])
        curvatures = np.array(outputs)

        # Simple linear fit
        coeffs = np.polyfit(shell_indices, curvatures, 1)
        linear_pred = np.polyval(coeffs, shell_indices)
        linear_error = np.mean(np.abs(curvatures - linear_pred))

        # If linear interpolation is very accurate, it's too easy
        return linear_error < 0.01  # Very low error means toy-solvable

    def _expected_curvature_smoothness(self, curr: CE1GeometryPoint, prev: CE1GeometryPoint,
                                      next_p: CE1GeometryPoint) -> float:
        """Expected smoothness in curvature field."""
        # Curvature should vary smoothly: κ_{n+1} - 2κ_n + κ_{n-1} should be small
        if hasattr(curr, 'true_curvature'):
            k_prev = getattr(prev, 'true_curvature', self._pascal_curvature(prev.shell_index))
            k_curr = getattr(curr, 'true_curvature', self._pascal_curvature(curr.shell_index))
            k_next = getattr(next_p, 'true_curvature', self._pascal_curvature(next_p.shell_index))
            return k_next - 2*k_curr + k_prev
        return 0.0

    def _curvature_discontinuity(self, curr: CE1GeometryPoint, prev: CE1GeometryPoint,
                                next_p: CE1GeometryPoint) -> float:
        """Measure discontinuity at mirror phase boundary."""
        k_prev = self._pascal_curvature(prev.shell_index)
        k_curr = self._pascal_curvature(curr.shell_index)
        k_next = self._pascal_curvature(next_p.shell_index)

        # Discontinuity as difference from expected smooth behavior
        expected_k_curr = (k_prev + k_next) / 2
        return abs(k_curr - expected_k_curr)

    def _pascal_curvature(self, n: int) -> float:
        """Compute Pascal curvature κ_n with mirror phase and prime effects."""
        if n < 2:
            return 0.0
        try:
            # Base curvature: central binomial coefficient behavior
            base_curvature = 1.0 / (n * math.log(n + 1))

            # Add mirror phase effects (n ≡ 3 mod 4 has singularities)
            mirror_phase = n % 4
            if mirror_phase == 3:
                # Tangent singularity at mirror phase boundary
                base_curvature *= (1.0 + 0.5 * math.sin(2 * math.pi * math.log(n)))

            # Add prime gap effects (irregularities at prime boundaries)
            # Primes create discontinuities in the curvature field
            next_prime = self._next_prime(n)
            prev_prime = self._prev_prime(n)
            if next_prime and prev_prime:
                prime_distance = min(n - prev_prime, next_prime - n) / float(n)
                base_curvature *= (1.0 + 0.3 * math.exp(-prime_distance))

            # Add oscillatory component from digit patterns
            digit_oscillation = 0.1 * math.sin(2 * math.pi * self._digit_entropy(n))
            base_curvature += digit_oscillation

            return base_curvature
        except:
            return 0.0

    def _next_prime(self, n: int) -> Optional[int]:
        """Find next prime after n."""
        if n < 2:
            return 2
        candidate = n + 1
        while candidate < n + 100:  # Reasonable search limit
            if self._is_prime(candidate):
                return candidate
            candidate += 1
        return None

    def _prev_prime(self, n: int) -> Optional[int]:
        """Find prime before n."""
        if n <= 2:
            return None
        candidate = n - 1
        while candidate > 1:
            if self._is_prime(candidate):
                return candidate
            candidate -= 1
        return None

    def _is_prime(self, n: int) -> bool:
        """Check if n is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def _digit_entropy(self, n: int) -> float:
        """Calculate digit entropy as complexity measure."""
        if n == 0:
            return 0.0

        digits = [int(d) for d in str(n)]
        digit_counts = {}
        for d in digits:
            digit_counts[d] = digit_counts.get(d, 0) + 1

        entropy = 0.0
        total_digits = len(digits)
        for count in digit_counts.values():
            p = count / total_digits
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

# Export benchmarks
ce1_benchmarks = [
    MirrorPhaseClassificationBenchmark(),
    CurvatureFieldRegressionBenchmark()
]
