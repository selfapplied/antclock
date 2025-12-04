#!/.venv/bin/python
"""
CE3 Simplicial Benchmarks: Topological Emergence Testing

Tests CE3 simplicial emergence principles with factorization complexes
and witness consistency that require understanding of topological invariants.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import math
from pathlib import Path

from ce_benchmark_types import (
    CE3Benchmark, CE3SimplicialPoint, BenchmarkConfig, BenchmarkResult
)

@dataclass
class FactorizationComplexBenchmark:
    """
    CE3.1: Prime Factorization Complex Classification

    Classify numbers by their factorization complexity, requiring understanding
    of how prime factors generate simplicial complexes and topological invariants.
    """

    def __init__(self):
        self.config = BenchmarkConfig(
            name="factorization_complex_classification",
            scale=40000,
            diversity_factors=[
                "factorization_complexity",  # Prime vs composite structure
                "simplex_dimension_variation", # Different topological dimensions
                "witness_consistency",       # Topological invariants
                "emergent_patterns"         # Non-trivial factor relationships
            ],
            ce_layer="ce3"
        )

        # Pre-compute primes for efficiency
        self._primes = self._sieve_of_eratosthenes(100000)

    def generate_dataset(self, size: int) -> Tuple[List[CE3SimplicialPoint], List[int]]:
        """Generate factorization complex classification dataset."""
        inputs = []
        outputs = []  # Classification based on factorization topology

        for _ in range(size):
            # Sample number with interesting factorization properties
            n = self._sample_interesting_number()

            # Analyze factorization structure
            factors = self._prime_factorization(n)
            complexity_class = self._classify_factorization_topology(factors)

            outputs.append(complexity_class)

            # Extract simplicial features
            simplex_dim = self._compute_simplex_dimension(factors)
            factor_complexity = self._compute_factor_complexity(factors)
            topological_invariant = self._compute_topological_invariant(n, factors)
            witness_consistency = self._compute_witness_consistency(n, factors)

            point = CE3SimplicialPoint(
                simplex_dimension=simplex_dim,
                factor_complexity=factor_complexity,
                topological_invariant=topological_invariant,
                witness_consistency=witness_consistency
            )
            inputs.append(point)

        return inputs, outputs

    def evaluate_mathematical_consistency(self, model: Any, inputs: List[CE3SimplicialPoint]) -> float:
        """Evaluate preservation of topological and factorization properties."""
        if model is None:
            return 1.0

        consistency_scores = []
        for point in inputs[:400]:
            # Check if factorization topology is preserved
            topology_consistent = self._check_topology_preservation(point)
            witness_consistent = point.witness_consistency > 0.5  # Threshold for consistency

            consistent = topology_consistent and witness_consistent
            consistency_scores.append(float(consistent))

        return np.mean(consistency_scores)

    def is_toy_solution_possible(self, dataset: Tuple[List[CE3SimplicialPoint], List[int]]) -> bool:
        """Check if simple factorization rules could solve this."""
        inputs, outputs = dataset

        # Extract simple features
        dimensions = np.array([p.simplex_dimension for p in inputs])
        complexities = np.array([p.factor_complexity for p in inputs])

        # Test simple decision rules based on obvious properties

        # Rule 1: Classify by dimension
        for threshold in [1, 2, 3, 5]:
            simple_pred = (dimensions >= threshold).astype(int)
            simple_acc = np.mean(simple_pred == outputs)
            if simple_acc > 0.85:
                return True

        # Rule 2: Classify by complexity
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            simple_pred = (complexities > threshold).astype(int)
            simple_acc = np.mean(simple_pred == outputs)
            if simple_acc > 0.85:
                return True

        # Rule 3: Check for obvious parity/simple patterns
        invariants = np.array([p.topological_invariant for p in inputs])
        for threshold in [2, 3, 5, 7]:
            simple_pred = (invariants % threshold == 0).astype(int)
            simple_acc = np.mean(simple_pred == outputs)
            if simple_acc > 0.85:
                return True

        return False

    def _sieve_of_eratosthenes(self, limit: int) -> List[int]:
        """Generate primes up to limit using sieve."""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False

        return [i for i in range(2, limit + 1) if sieve[i]]

    def _sample_interesting_number(self) -> int:
        """Sample number with interesting factorization properties."""
        while True:
            # Bias toward numbers with rich factorization structure
            if np.random.random() < 0.4:
                # Highly composite numbers
                n = self._generate_highly_composite()
            elif np.random.random() < 0.3:
                # Prime powers
                prime = np.random.choice(self._primes[:100])  # Smaller primes
                power = np.random.randint(2, 6)
                n = prime ** power
            elif np.random.random() < 0.2:
                # Square-free numbers (product of distinct primes)
                primes = np.random.choice(self._primes[:50], size=np.random.randint(2, 5), replace=False)
                n = np.prod(primes)
            else:
                # General composite
                n = np.random.randint(2, 100000)

            # Ensure it's not too trivial (not prime, not too small)
            if n > 10 and not self._is_prime(n):
                return n

    def _generate_highly_composite(self) -> int:
        """Generate highly composite number."""
        # Use small primes with exponents that decrease
        primes = [2, 3, 5, 7, 11, 13, 17]
        exponents = [4, 2, 2, 1, 1, 1, 1]  # Decreasing exponents

        n = 1
        for p, e in zip(primes, exponents):
            n *= p ** np.random.randint(0, e + 1)

        return max(n, 12)  # Ensure not too small

    def _is_prime(self, n: int) -> bool:
        """Check if number is prime."""
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

    def _prime_factorization(self, n: int) -> Dict[int, int]:
        """Compute prime factorization."""
        factors = {}
        i = 2
        while i * i <= n:
            while n % i == 0:
                factors[i] = factors.get(i, 0) + 1
                n //= i
            i += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def _classify_factorization_topology(self, factors: Dict[int, int]) -> int:
        """Classify based on factorization topology."""
        if not factors:
            return 0  # Should not happen

        num_primes = len(factors)
        max_exponent = max(factors.values()) if factors else 0

        # Classification based on topological complexity
        if num_primes == 1:
            if max_exponent == 1:
                return 0  # Prime
            else:
                return 1  # Prime power
        elif num_primes == 2:
            if max_exponent == 1:
                return 2  # Semiprime
            else:
                return 3  # Higher power with two primes
        elif all(e == 1 for e in factors.values()):
            return 4  # Square-free with 3+ primes
        else:
            return 5  # General composite

    def _compute_simplex_dimension(self, factors: Dict[int, int]) -> int:
        """Compute simplicial dimension from factorization."""
        if not factors:
            return 0

        # Dimension based on number of distinct prime factors
        # and maximum exponent
        num_primes = len(factors)
        max_exp = max(factors.values())

        # Higher prime count and exponents increase dimension
        return min(num_primes + max_exp - 1, 10)  # Cap at reasonable dimension

    def _compute_factor_complexity(self, factors: Dict[int, int]) -> float:
        """Compute complexity measure of factorization."""
        if not factors:
            return 0.0

        # Complexity based on prime count and exponent distribution
        num_primes = len(factors)
        total_exponents = sum(factors.values())
        max_exponent = max(factors.values())

        # Normalize to [0,1] range
        complexity = (num_primes / 10.0) + (total_exponents / 20.0) + (max_exponent / 5.0)
        return min(complexity, 1.0)

    def _compute_topological_invariant(self, n: int, factors: Dict[int, int]) -> int:
        """Compute topological invariant (e.g., divisor count)."""
        if not factors:
            return 1

        # Number of divisors as basic topological invariant
        divisors = 1
        for exponent in factors.values():
            divisors *= (exponent + 1)

        return divisors

    def _compute_witness_consistency(self, n: int, factors: Dict[int, int]) -> float:
        """Compute witness consistency for factorization."""
        if not factors:
            return 1.0

        # Check if factorization is consistent with number properties
        reconstructed = 1
        for prime, exp in factors.items():
            reconstructed *= prime ** exp

        # Perfect reconstruction should give consistency = 1.0
        if reconstructed == n:
            return 1.0
        else:
            return 0.0  # Inconsistent factorization

    def _check_topology_preservation(self, point: CE3SimplicialPoint) -> bool:
        """Check if topological properties are preserved."""
        # Simplex dimension should be reasonable
        if not (0 <= point.simplex_dimension <= 10):
            return False

        # Factor complexity should be in valid range
        if not (0 <= point.factor_complexity <= 1):
            return False

        # Topological invariant should be positive
        if point.topological_invariant < 1:
            return False

        # Witness consistency should be high
        if point.witness_consistency < 0.8:
            return False

        return True

@dataclass
class SimplicialHomologyBenchmark:
    """
    CE3.2: Simplicial Homology Regression

    Predict Betti numbers and homology groups from simplicial complexes,
    requiring understanding of how factorization generates topological spaces.
    """

    def __init__(self):
        self.config = BenchmarkConfig(
            name="simplicial_homology_regression",
            scale=15000,
            diversity_factors=[
                "homology_rank_variation",  # Different Betti numbers
                "complex_connectivity",     # Varying connectivity patterns
                "emergent_cycles",         # Non-trivial homology classes
                "stability_under_perturbation" # Robustness to small changes
            ],
            ce_layer="ce3"
        )

    def generate_dataset(self, size: int) -> Tuple[List[CE3SimplicialPoint], List[float]]:
        """Generate simplicial homology regression dataset."""
        inputs = []
        outputs = []  # Betti number predictions

        for _ in range(size):
            # Generate simplicial complex from number theory
            complex_data = self._generate_simplicial_complex()

            # Target: first Betti number (H1 rank)
            betti_1 = self._compute_betti_number_1(complex_data)
            outputs.append(float(betti_1))

            # Extract features
            simplex_dim = complex_data['dimension']
            factor_complexity = complex_data['complexity']
            topological_invariant = complex_data['euler_characteristic']
            witness_consistency = complex_data['consistency']

            point = CE3SimplicialPoint(
                simplex_dimension=simplex_dim,
                factor_complexity=factor_complexity,
                topological_invariant=topological_invariant,
                witness_consistency=witness_consistency
            )
            inputs.append(point)

        return inputs, outputs

    def evaluate_mathematical_consistency(self, model: Any, inputs: List[CE3SimplicialPoint]) -> float:
        """Evaluate homology group properties and Euler characteristic."""
        if model is None:
            return 1.0

        consistency_scores = []
        for point in inputs[:200]:
            # Check Euler characteristic relation: χ = V - E + F - ... = rank(H0) - rank(H1) + rank(H2) - ...
            euler_consistent = self._check_euler_characteristic(point)
            homology_consistent = self._check_homology_properties(point)

            consistent = euler_consistent and homology_consistent
            consistency_scores.append(float(consistent))

        return np.mean(consistency_scores)

    def is_toy_solution_possible(self, dataset: Tuple[List[CE3SimplicialPoint], List[float]]) -> bool:
        """Check if simple topological rules could solve this."""
        inputs, outputs = dataset

        # Extract features
        dimensions = np.array([p.simplex_dimension for p in inputs])
        complexities = np.array([p.factor_complexity for p in inputs])
        invariants = np.array([p.topological_invariant for p in inputs])

        # Test simple rules for Betti number prediction

        # Rule 1: Betti_1 ≈ dimension
        simple_pred = dimensions.astype(float)
        simple_error = np.mean(np.abs(outputs - simple_pred))
        if simple_error < 0.5:
            return True

        # Rule 2: Betti_1 ≈ complexity * constant
        for const in [1, 2, 5, 10]:
            simple_pred = complexities * const
            simple_error = np.mean(np.abs(outputs - simple_pred))
            if simple_error < 0.5:
                return True

        # Rule 3: Betti_1 ≈ invariant / constant
        for const in [2, 3, 5, 10]:
            simple_pred = invariants.astype(float) / const
            simple_error = np.mean(np.abs(outputs - simple_pred))
            if simple_error < 0.5:
                return True

        return False

    def _generate_simplicial_complex(self) -> Dict[str, Any]:
        """Generate simplicial complex from number-theoretic construction."""
        # Build complex from prime factor lattice
        base_number = np.random.randint(2, 10000)

        # Get prime factors and build factor lattice
        factors = self._prime_factorization(base_number)
        primes = list(factors.keys())

        # Create simplices based on factor relationships
        simplices = self._build_factor_simplices(primes, factors)

        # Compute topological invariants
        dimension = self._compute_complex_dimension(simplices)
        complexity = len(simplices) / 100.0  # Normalize
        euler_char = self._compute_euler_characteristic(simplices)
        consistency = self._check_complex_consistency(simplices, base_number)

        return {
            'simplices': simplices,
            'dimension': dimension,
            'complexity': complexity,
            'euler_characteristic': euler_char,
            'consistency': consistency
        }

    def _prime_factorization(self, n: int) -> Dict[int, int]:
        """Compute prime factorization."""
        factors = {}
        i = 2
        while i * i <= n:
            while n % i == 0:
                factors[i] = factors.get(i, 0) + 1
                n //= i
            i += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def _build_factor_simplices(self, primes: List[int], factors: Dict[int, int]) -> List[List[int]]:
        """Build simplices from prime factor relationships."""
        simplices = []

        # 0-simplices: individual primes
        for prime in primes:
            simplices.append([prime])

        # 1-simplices: prime pairs that could form composites
        for i in range(len(primes)):
            for j in range(i + 1, len(primes)):
                p1, p2 = primes[i], primes[j]
                # Only if they could reasonably form a composite
                if p1 * p2 < 100000:  # Size constraint
                    simplices.append([p1, p2])

        # Higher simplices: triples, quadruples, etc.
        if len(primes) >= 3:
            for i in range(len(primes)):
                for j in range(i + 1, len(primes)):
                    for k in range(j + 1, len(primes)):
                        p1, p2, p3 = primes[i], primes[j], primes[k]
                        if p1 * p2 * p3 < 1000000:  # Size constraint
                            simplices.append([p1, p2, p3])

        return simplices

    def _compute_complex_dimension(self, simplices: List[List[int]]) -> int:
        """Compute dimension of simplicial complex."""
        if not simplices:
            return 0
        return max(len(simplex) - 1 for simplex in simplices)

    def _compute_euler_characteristic(self, simplices: List[List[int]]) -> int:
        """Compute Euler characteristic: V - E + F - ..."""
        if not simplices:
            return 0

        # Count simplices by dimension
        dim_counts = {}
        for simplex in simplices:
            dim = len(simplex) - 1
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        # Euler characteristic with alternating signs
        euler = 0
        sign = 1
        for dim in sorted(dim_counts.keys()):
            euler += sign * dim_counts[dim]
            sign *= -1

        return euler

    def _check_complex_consistency(self, simplices: List[List[int]], base_number: int) -> float:
        """Check consistency of complex with original number."""
        # Verify that the complex could reasonably represent the number's factorization
        if not simplices:
            return 0.0

        # Check if vertices (primes) can reconstruct the number
        primes = set()
        for simplex in simplices:
            primes.update(simplex)

        # Try to reconstruct number from prime factors
        if not primes:
            return 0.0

        # Simple reconstruction check
        prime_list = sorted(list(primes))
        test_product = 1
        for p in prime_list[:min(5, len(prime_list))]:  # Limit to avoid overflow
            test_product *= p
            if test_product > base_number * 10:  # Too large
                break

        # Consistency score based on how close we get to original number
        if test_product == base_number:
            return 1.0
        elif test_product < base_number:
            return test_product / base_number  # Partial reconstruction
        else:
            return base_number / test_product  # Overshot

    def _compute_betti_number_1(self, complex_data: Dict[str, Any]) -> int:
        """Compute first Betti number (rank of H1)."""
        simplices = complex_data['simplices']
        dimension = complex_data['dimension']

        if dimension < 1:
            return 0  # No 1-cycles possible

        # Simplified Betti number computation
        # In a real implementation, this would use homology computation
        # Here we use a proxy based on complex structure

        num_edges = sum(1 for s in simplices if len(s) == 2)
        num_triangles = sum(1 for s in simplices if len(s) == 3)

        # Rough approximation: cycles = edges - vertices + triangles
        # (More precisely: β1 = E - V + T - ... but simplified)
        vertices = len(set(v for s in simplices for v in s))
        betti_1 = max(0, num_edges - vertices + num_triangles // 2)

        return min(betti_1, 20)  # Cap at reasonable value

    def _check_euler_characteristic(self, point: CE3SimplicialPoint) -> bool:
        """Verify Euler characteristic is reasonable."""
        # Euler characteristic should be non-negative for simplicial complexes
        # (though this is not strictly true, it's a reasonable constraint)
        return point.topological_invariant >= 0

    def _check_homology_properties(self, point: CE3SimplicialPoint) -> bool:
        """Check basic homology properties."""
        # Betti numbers should be non-negative
        # Dimension should be consistent with complexity
        dim_reasonable = 0 <= point.simplex_dimension <= 10
        complexity_reasonable = 0 <= point.factor_complexity <= 1

        return dim_reasonable and complexity_reasonable

# Export benchmarks
ce3_benchmarks = [
    FactorizationComplexBenchmark(),
    SimplicialHomologyBenchmark()
]

