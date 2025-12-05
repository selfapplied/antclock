# AntClock Documentation Navigation

## Overview

This directory contains detailed documentation for AntClock's architecture and systems.

## Available Documentation

### [spec.md](spec.md) - Complete Mathematical Specification
Complete canonical specification of the CE1→CE2→CE3 framework:

- **CE Framework Overview**: Three-layer architecture connecting discrete→dynamical→simplicial
- **Detailed Mathematical Framework**: CE1 components, key operators, clock dynamics, theorems
- **Mathematical Foundations**: Curvature flows, digit symmetries, Galois covering spaces
- **Transport Mechanisms**: Continued fractions, digital polynomials, universal clock
- **Categorical Structure**: Shadow towers, functor categories, coherence laws
- **Riemann Hypothesis Connection**: Discrete analogues of critical line, zeros, L-functions
- **Single Source of Truth**: Complete mathematical specification for all implementations

### [run.md](run.md) - Execution Architecture
Comprehensive guide to AntClock's **viral immunity injection model**:

- **`run.sh`**: Universal immune system entry point and environment manager
- **Viral Immunity Model**: Biological analogy for script validation and integration
- **Antigen Testing**: Scripts prove "non-pathogenic" before gaining execution rights
- **Immune Surveillance**: Failed scripts rejected, successful ones create "antibodies" (hashbangs)
- **Hashbang System**: Unified `#!/run.sh` execution across all components
- **Environment Abstraction**: Handles virtual environments, caches, sandboxes
- **Agent Unification**: Consistent execution for demos, benchmarks, tools, ζ-cards

### [benchmarks.md](benchmarks.md) - Sentinel Node Architecture
Complete guide to the benchmarking ecosystem:

- **`benchmark.py`**: Sentinel node orchestrating synthetic → metabolic → phenotype layers
- **Three Biome Architecture**: Immune screening, metabolic profiling, phenotype evaluation
- **Synthetic Biome**: CE1/CE2/CE3 structural verification before training
- **Metabolic Layer**: κ-guardian events, χ-FEG modulation, convergence acceleration
- **Phenotype Layer**: Standard ML benchmarks with CE timing overlays
- **Immune Integration**: Harmony with run.sh's viral immunity injection model

### [applications.md](applications.md) - Practical Applications
Complete guide to AntClock's coherence engine applications:

- **Coherence Engine**: Structural break detection, stability maintenance, information compression
- **8 Application Domains**: Fault-tolerant hashing, AI error correction, predictive signals, neural architectures, control systems, data classification, generative compression
- **Research Context**: Connections between combinatorial curvature and real-world applications
- **Implementation Status**: Current capabilities and future extension opportunities

### Key Concepts
- **Unified Agent Interface**: All components use the same execution model
- **Conservative Permissions**: Only tested, working scripts become executable
- **Environment Portability**: Works across development, sandboxed, and production environments
- **Clean Source Management**: Centralized caching, no `__pycache__` in source directories
- **Creative Commons License**: CC BY-SA 4.0 requiring attribution and share-alike

## Quick Reference

```bash
# Execute any component directly
./demos/antclock.py         # Mathematical demonstrations
./benchmarks/benchmark.py   # CE framework validation
./tools/test_types.py       # Type system tools

# Full ecosystem via Makefile
make                        # Complete pipeline execution
make test                   # Test suite
make benchmarks             # Benchmark suite

# All route through run.sh for consistent environment setup
```

---

**See [../README.md](../README.md) for the main project overview and quick start guide.**
