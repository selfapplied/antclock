#!/.venv/bin/python
"""
CE Benchmark Types: Type-Safe Benchmark Framework

Implements benchmarks that directly test CE1/CE2/CE3 mathematical principles
with diverse inputs that make toy solutions impossible.
"""

from typing import Protocol, TypeVar, Generic, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import time
from pathlib import Path

# Type variables for generic benchmarks
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')
TModel = TypeVar('TModel')

class ProgressTracker:
    """Tracks and displays benchmark progress with detailed metrics."""

    def __init__(self, total_benchmarks: int, total_samples: int):
        self.total_benchmarks = total_benchmarks
        self.total_samples = total_samples
        self.completed_benchmarks = 0
        self.completed_samples = 0
        self.start_time = time.time()
        self.benchmark_start_time = None
        self.last_update_time = time.time()
        self.update_interval = 0.1  # Update every 100ms

    def start_benchmark(self, benchmark_name: str, benchmark_samples: int):
        """Start tracking a new benchmark."""
        self.benchmark_start_time = time.time()
        self.current_benchmark_name = benchmark_name
        self.current_benchmark_samples = benchmark_samples
        self.current_samples_generated = 0

        print(f"\n🏃 Starting {benchmark_name}")
        print(f"   Target samples: {benchmark_samples:,}")
        print(f"   Progress: [{self.completed_benchmarks}/{self.total_benchmarks}] benchmarks")
        self._print_progress_header()

    def update_generation_progress(self, samples_generated: int):
        """Update progress during dataset generation."""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.current_samples_generated = samples_generated
        elapsed = current_time - self.benchmark_start_time
        rate = samples_generated / elapsed if elapsed > 0 else 0

        # Progress bar
        progress_pct = samples_generated / self.current_benchmark_samples
        bar_width = 40
        filled = int(bar_width * progress_pct)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Time estimates
        eta = (self.current_benchmark_samples - samples_generated) / rate if rate > 0 else 0

        print(f"\r   📊 [{bar}] {progress_pct:.1%} | {samples_generated:,}/{self.current_benchmark_samples:,} samples | "
              f"{rate:.0f} samples/sec | ETA: {eta:.1f}s", end="", flush=True)

        self.last_update_time = current_time

    def complete_benchmark_generation(self):
        """Mark dataset generation as complete."""
        elapsed = time.time() - self.benchmark_start_time
        rate = self.current_benchmark_samples / elapsed if elapsed > 0 else 0
        print(f"\r   ✅ Dataset generated: {self.current_benchmark_samples:,} samples in {elapsed:.2f}s ({rate:.0f} samples/sec)")
        print("   🔍 Evaluating mathematical consistency...")
        self.completed_samples += self.current_benchmark_samples

    def complete_benchmark(self):
        """Mark entire benchmark as complete."""
        self.completed_benchmarks += 1
        total_elapsed = time.time() - self.start_time

        # Overall progress
        overall_progress = self.completed_benchmarks / self.total_benchmarks
        eta_total = (total_elapsed / self.completed_benchmarks) * (self.total_benchmarks - self.completed_benchmarks) if self.completed_benchmarks > 0 else 0

        print("   🎯 Benchmark completed successfully!")
        print(f"   📊 Progress: {overall_progress:.1f} complete | ETA: {eta_total:.1f}s")
    def _print_progress_header(self):
        """Print progress header with system info."""
        print("   💻 Starting benchmark execution...")

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    name: str
    scale: int  # Minimum training examples required
    diversity_factors: List[str]  # What makes inputs diverse
    ce_layer: str  # 'ce1', 'ce2', or 'ce3'
    output_dir: Path = Path(".out")

@dataclass
class BenchmarkResult:
    """Results from benchmark execution."""
    accuracy: float
    convergence_speed: float  # Epochs to convergence
    mathematical_consistency: float  # How well it preserves CE properties
    generalization_gap: float  # Train-test performance difference
    metadata: Dict[str, Any]

class CEBenchmark(Protocol[TInput, TOutput, TModel]):
    """Protocol for CE-aware benchmarks."""

    @property
    def config(self) -> BenchmarkConfig: ...

    def generate_dataset(self, size: int) -> Tuple[List[TInput], List[TOutput]]:
        """Generate diverse dataset that can't be solved by heuristics."""
        ...

    def evaluate_mathematical_consistency(self, model: TModel, inputs: List[TInput]) -> float:
        """Measure how well the model preserves CE mathematical properties."""
        ...

    def is_toy_solution_possible(self, dataset: Tuple[List[TInput], List[TOutput]]) -> bool:
        """Check if simple heuristics could solve this dataset."""
        ...

@dataclass
class CE1GeometryPoint:
    """CE1: Point in discrete geometry space."""
    shell_index: int
    mirror_phase: int  # 0,1,2,3 mod 4
    curvature_value: float
    digit_entropy: float

@dataclass
class CE2FlowPoint:
    """CE2: Point in dynamical flow space."""
    time_parameter: float
    flow_velocity: complex
    invariant_measure: float
    spectral_radius: float

@dataclass
class CE3SimplicialPoint:
    """CE3: Point in simplicial emergence space."""
    simplex_dimension: int
    factor_complexity: float
    topological_invariant: int
    witness_consistency: float

# Specific benchmark types
CE1Benchmark = CEBenchmark[CE1GeometryPoint, int, Any]
CE2Benchmark = CEBenchmark[CE2FlowPoint, complex, Any]
CE3Benchmark = CEBenchmark[CE3SimplicialPoint, float, Any]

@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks."""
    name: str
    ce1_benchmarks: List[CE1Benchmark]
    ce2_benchmarks: List[CE2Benchmark]
    ce3_benchmarks: List[CE3Benchmark]

    def run_all(self, output_dir: Path = Path(".out")) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in suite with detailed progress tracking."""
        results = {}
        output_dir.mkdir(exist_ok=True)

        # Initialize progress tracker
        all_benchmarks = self.ce1_benchmarks + self.ce2_benchmarks + self.ce3_benchmarks
        total_samples = sum(b.config.scale for b in all_benchmarks)
        progress = ProgressTracker(len(all_benchmarks), total_samples)

        print(f"🚀 Starting {self.name}")
        print(f"   Total benchmarks: {len(all_benchmarks)}")
        print(f"   Total samples: {total_samples:,}")
        print("=" * 60)

        for i, benchmark in enumerate(all_benchmarks, 1):
            # Start benchmark tracking
            progress.start_benchmark(benchmark.config.name, benchmark.config.scale)

            # Run benchmark with progress updates
            result = self._run_single_benchmark_with_progress(benchmark, progress)
            results[benchmark.config.name] = result

            # Complete benchmark
            progress.complete_benchmark()

            # Save individual results immediately
            result_file = output_dir / f"{benchmark.config.name}_result.json"
            self._save_result(result, result_file)

            # Force garbage collection to prevent memory accumulation
            import gc
            gc.collect()

        # Save suite summary
        summary_file = output_dir / f"{self.name}_summary.json"
        self._save_suite_summary(results, summary_file)

        # Final summary
        total_time = time.time() - progress.start_time
        print(f"\n🎉 Suite completed in {total_time:.2f}s!")
        print(f"   Results saved to {output_dir}/")

        return results

    def _run_single_benchmark(self, benchmark: CEBenchmark) -> BenchmarkResult:
        """Run a single benchmark with full evaluation."""
        # Generate diverse dataset
        train_inputs, train_outputs = benchmark.generate_dataset(benchmark.config.scale)
        test_inputs, test_outputs = benchmark.generate_dataset(benchmark.config.scale // 10)

        # Verify it's not solvable by toy methods
        if benchmark.is_toy_solution_possible((train_inputs, train_outputs)):
            raise ValueError(f"Benchmark {benchmark.config.name} can be solved by toy methods!")

        # Placeholder for actual model training/evaluation
        # This would be implemented by specific CE models
        accuracy = 0.0  # Would be computed from actual model
        convergence_speed = 0.0
        mathematical_consistency = benchmark.evaluate_mathematical_consistency(None, train_inputs)
        generalization_gap = 0.0

        return BenchmarkResult(
            accuracy=accuracy,
            convergence_speed=convergence_speed,
            mathematical_consistency=mathematical_consistency,
            generalization_gap=generalization_gap,
            metadata={
                'dataset_size': len(train_inputs),
                'diversity_factors': benchmark.config.diversity_factors,
                'ce_layer': benchmark.config.ce_layer
            }
        )

    def _run_single_benchmark_with_progress(self, benchmark: CEBenchmark, progress: ProgressTracker) -> BenchmarkResult:
        """Run a single benchmark with detailed progress tracking."""
        # Generate training dataset with progress updates
        print("   🔧 Generating training dataset...")
        train_inputs, train_outputs = self._generate_dataset_with_progress(
            benchmark, benchmark.config.scale, "train", progress)

        # Generate test dataset (smaller, no progress needed)
        print("   🧪 Generating test dataset...")
        test_inputs, test_outputs = benchmark.generate_dataset(benchmark.config.scale // 10)

        # Mark generation complete
        progress.complete_benchmark_generation()

        # Verify it's not solvable by toy methods
        if benchmark.is_toy_solution_possible((train_inputs, train_outputs)):
            raise ValueError(f"Benchmark {benchmark.config.name} can be solved by toy methods!")

        # Evaluate mathematical consistency
        start_consistency = time.time()
        mathematical_consistency = benchmark.evaluate_mathematical_consistency(None, train_inputs)
        consistency_time = time.time() - start_consistency

        print(f"   📐 Mathematical consistency: {mathematical_consistency:.3f} (computed in {consistency_time:.2f}s)")

        # Placeholder for actual model training/evaluation
        # This would be implemented by specific CE models
        accuracy = 0.0  # Would be computed from actual model
        convergence_speed = 0.0
        generalization_gap = 0.0

        return BenchmarkResult(
            accuracy=accuracy,
            convergence_speed=convergence_speed,
            mathematical_consistency=mathematical_consistency,
            generalization_gap=generalization_gap,
            metadata={
                'dataset_size': len(train_inputs),
                'diversity_factors': benchmark.config.diversity_factors,
                'ce_layer': benchmark.config.ce_layer,
                'generation_time': progress.benchmark_start_time and (time.time() - progress.benchmark_start_time),
                'consistency_time': consistency_time
            }
        )

    def _generate_dataset_with_progress(self, benchmark: CEBenchmark, size: int,
                                       dataset_type: str, progress: ProgressTracker) -> Tuple[List, List]:
        """Generate dataset with progress tracking."""
        # Use the benchmark's generate_dataset method with progress callback
        def progress_callback(current_size: int):
            progress.update_generation_progress(current_size)

        # Check if benchmark supports progress callback
        if hasattr(benchmark.generate_dataset, '__code__') and 'progress_callback' in benchmark.generate_dataset.__code__.co_varnames:
            inputs, outputs = benchmark.generate_dataset(size, progress_callback)
        else:
            # Fallback for benchmarks without progress callback support
            inputs, outputs = benchmark.generate_dataset(size)
            # Simulate progress updates
            batch_size = max(1, size // 100)
            for i in range(0, size, batch_size):
                progress.update_generation_progress(min(i + batch_size, size))

        return inputs, outputs

    def _save_result(self, result: BenchmarkResult, filepath: Path):
        """Save individual benchmark result."""
        import json
        data = {
            'accuracy': result.accuracy,
            'convergence_speed': result.convergence_speed,
            'mathematical_consistency': result.mathematical_consistency,
            'generalization_gap': result.generalization_gap,
            'metadata': result.metadata
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_suite_summary(self, results: Dict[str, BenchmarkResult], filepath: Path):
        """Save suite-level summary."""
        import json
        summary = {
            'suite_name': self.name,
            'total_benchmarks': len(results),
            'average_accuracy': np.mean([r.accuracy for r in results.values()]),
            'average_consistency': np.mean([r.mathematical_consistency for r in results.values()]),
            'results': {name: {
                'accuracy': r.accuracy,
                'consistency': r.mathematical_consistency,
                'ce_layer': r.metadata['ce_layer']
            } for name, r in results.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

