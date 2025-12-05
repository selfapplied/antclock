# CE Benchmarks — Sentinel Node Architecture

## Overview

The `benchmark.py` file serves as the **sentinel node** — the central nervous system orchestrating AntClock's complete benchmarking ecosystem. Think thymus, not linter; conductor, not inspector.

This unified sentinel manages the entire benchmarking lifecycle: synthetic generation → structural verification → standard evaluation → metabolic timing analysis.

## Sentinel Node Philosophy

**"Verify before validate"** — Like an immune system screening antigens before integration, the sentinel node ensures the synthetic biome is structurally stable before allowing any standard benchmarking tasks.

```python
# Sentinel Node: Comprehensive ecosystem screening
def sentinel_workflow():
    # Phase 1: Immune screening (synthetic biome health)
    verify_synthetic_biome()      # CE1/CE2/CE3 structural integrity
    ensure_shell_diversity()       # Curvature spectrum coverage
    validate_topology()           # Simplicial consistency

    # Phase 2: Metabolic assessment (timing & convergence)
    assess_convergence()          # κ-guardian behavior
    profile_phase_transitions()   # δ-bifurcation captures
    measure_adaptation()          # χ-FEG modulation

    # Phase 3: Phenotype evaluation (standard tasks)
    evaluate_standard_tasks()     # SCAN, COGS, PCFG, CFQ
    aggregate_results()          # Unified reporting
    generate_paper_ready_data()  # Publication artifacts
```

## Architecture: Three Biome Layers

### 1. Synthetic Biome (Immune Screening Layer)

**CE-Core Benchmarks**: CE1/CE2/CE3 synthetic datasets that probe the framework's internal structure.

```python
from .ce.ce1 import generate_mirror_phase_benchmarks
from .ce.ce2 import generate_flow_field_benchmarks
from .ce.ce3 import generate_simplicial_benchmarks

# Sentinel verifies biome health before allowing training
biome_health = sentinel.verify_synthetic_biome()
if not biome_health.stable:
    raise EcosystemInstability("Synthetic biome requires stabilization")
```

#### CE1: Discrete Geometry
- Mirror-phase shell classification
- Curvature field regression
- Digit symmetry breaking patterns

#### CE2: Dynamical Flow
- Gauss map convergence analysis
- Flow field integration
- Period-doubling bifurcation tracking

#### CE3: Simplicial Topology
- Factorization complex classification
- Simplicial homology regression
- Vertex/edge/face coherence inference

### 2. Metabolic Layer (Timing Integration)

**Convergence & Adaptation Analysis**: Measures how CE's geometric structure accelerates learning.

```python
from .timing import ce_timing_engine

# Metabolic profiling through sentinel
metabolic_profile = sentinel.assess_metabolism(
    convergence_curves=True,
    phase_transitions=True,
    learning_adaptation=True
)
```

#### κ-Guardian Events
- Early stopping based on curvature stabilization
- Phase transition detection
- Geometric convergence acceleration

#### χ-FEG Modulation
- Learning rate adaptation based on field geometry
- Bifurcation-aware training dynamics
- Entropy-guided optimization

### 3. Phenotype Layer (Standard Tasks)

**Real-World Validation**: Standard ML benchmarks to demonstrate practical benefits.

```python
from .standard.scan import run_scan_benchmark
from .standard.cogs import run_cogs_benchmark
from .standard.cfq import run_cfq_benchmark

# Phenotype evaluation through sentinel
phenotype_results = sentinel.evaluate_phenotypes(
    tasks=['scan', 'cogs', 'cfq', 'pcfg', 'rpm', 'math'],
    ce_overlay=True  # Include CE timing traces
)
```

## Sentinel Node Operations

### Unified Command Interface

```bash
# Complete ecosystem screening
./run.sh benchmark.py

# Specific biome assessment
./run.sh benchmark.py --biome=synthetic --verify-only
./run.sh benchmark.py --biome=metabolic --profile
./run.sh benchmark.py --biome=phenotype --tasks=scan,cogs

# Individual component testing
./run.sh benchmark.py --ce1 --scale=50000
./run.sh benchmark.py --ce2 --flow-analysis
./run.sh benchmark.py --ce3 --topology-check
```

### Immune Screening Pipeline

```python
def immune_screening():
    """Complete synthetic biome health assessment"""

    # Structural integrity checks
    shell_diversity_ok = verify_shell_coverage()
    curvature_spectrum_ok = verify_curvature_distribution()
    entropy_balance_ok = verify_entropy_gradients()
    topology_consistent = verify_simplicial_complexes()

    # Pathogen detection (toy solutions)
    no_memorization = detect_toy_solution_resistance()
    no_shortcuts = detect_structural_shortcuts()

    # Biome stability
    stable_under_scale = verify_scaling_properties()
    stable_under_noise = verify_robustness()

    return EcosystemHealth(
        structural= all([shell_diversity_ok, curvature_spectrum_ok,
                        entropy_balance_ok, topology_consistent]),
        pathogen_free= all([no_memorization, no_shortcuts]),
        stable= all([stable_under_scale, stable_under_noise])
    )
```

### Metabolic Profiling

```python
def metabolic_profiling():
    """Assess CE's learning dynamics"""

    # Convergence acceleration
    baseline_time = measure_baseline_convergence()
    ce_time = measure_ce_convergence()
    speedup_factor = baseline_time / ce_time

    # Phase transition detection
    transitions = detect_bifurcations()
    stability_points = find_equilibria()

    # Adaptation quality
    learning_curves = profile_learning_dynamics()
    generalization_gaps = measure_generalization()

    return MetabolicProfile(
        speedup=speedup_factor,
        transitions=transitions,
        stability=stability_points,
        adaptation_quality=learning_curves,
        generalization=generalization_gaps
    )
```

## Directory Structure

```
benchmarks/
├── benchmark.py              # 🛡️ Sentinel Node - unified orchestrator
├── definitions.py            # 📋 Type definitions & protocols
├── modules.py                # 🔧 Shared infrastructure
├── timing.py                 # ⏱️ Metabolic timing engine
├── final_ce_timing_results.py # 📊 Timing result aggregator
│
├── ce/                       # 🧬 Synthetic Biome (Immune Screening)
│   ├── __init__.py
│   ├── ce1.py                # Discrete geometry benchmarks
│   ├── ce2.py                # Dynamical flow benchmarks
│   ├── ce3.py                # Simplicial topology benchmarks
│   └── synthetic.py          # CE benchmark runner
│
└── standard/                 # 🎯 Phenotype Layer (Standard Tasks)
    ├── __init__.py
    ├── scan.py               # Sequence-to-sequence parsing
    ├── cogs.py               # Semantic parsing
    ├── cfq.py                # Compositional questions
    ├── pcfg.py               # Syntactic parsing
    ├── rpm.py                # Visual reasoning
    ├── math.py               # Mathematical reasoning
    └── standard.py           # Standard benchmark orchestrator
```

## Sentinel Node Benefits

### 1. **Unified Ecosystem Management**
- Single entry point for all benchmarking activities
- Consistent environment across synthetic → metabolic → phenotype layers
- Coordinated reporting and result aggregation

### 2. **Immune System Protection**
- Verifies synthetic biome health before expensive standard tasks
- Prevents wasted computation on unstable environments
- Detects and rejects pathological configurations

### 3. **Metabolic Intelligence**
- Tracks convergence acceleration from geometric structure
- Profiles learning dynamics and phase transitions
- Provides timing intelligence for paper-ready results

### 4. **Phenotype Validation**
- Comprehensive standard task evaluation
- CE timing overlay for comparative analysis
- Publication-ready result formatting

### 5. **Architectural Harmony**
- Clean separation: synthetic/metabolic/phenotype
- Biological metaphor guides system design
- Living system that adapts and self-regulates

## Usage Examples

### Complete Ecosystem Screening
```bash
# Full sentinel node activation
./run.sh benchmark.py

# Generates: verification → synthetic → metabolic → phenotype results
# Outputs: .out/benchmark_results/ with complete ecosystem report
```

### Synthetic Biome Health Check
```bash
# Immune screening only
./run.sh benchmark.py --verify-synthetic

# Checks: shell diversity, curvature coverage, entropy balance
# Outputs: biome health report and structural validation
```

### Metabolic Profiling
```bash
# Timing and convergence analysis
./run.sh benchmark.py --profile-metabolic --epochs=100

# Measures: κ-guardian events, χ-FEG modulation, speedup factors
# Outputs: timing curves and phase transition analysis
```

### Phenotype Evaluation
```bash
# Standard task benchmarking
./run.sh benchmark.py --evaluate-phenotype --tasks=scan,cogs,cfq

# Runs: SCAN, COGS, CFQ with CE timing overlays
# Outputs: comparative performance analysis
```

## Integration with Run.sh Immune System

The sentinel node works in perfect harmony with `run.sh`'s viral immunity injection model:

```bash
# run.sh: Script-level immune screening
./run.sh some_script.py        # Tests script → grants permissions if healthy

# benchmark.py: Ecosystem-level immune screening
./run.sh benchmark.py          # Tests environment → validates biome if stable
```

**run.sh** ensures individual scripts are viable before granting execution rights.

**benchmark.py** ensures the entire benchmarking ecosystem is healthy before allowing evaluation.

Together they create a complete immune system for AntClock's mathematical exploration ecosystem.

---

**The sentinel node architecture transforms benchmarking from passive validation into active ecosystem stewardship — a living system that protects, nurtures, and validates AntClock's mathematical discoveries.**
