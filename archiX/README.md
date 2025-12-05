# archiX: White Paper Content

This directory contains all materials for the AntClock white paper/archiiiv publication.

**Note**: Core Volte system implementations have been integrated into the main `antclock/` package for active development. The files here serve as archival documentation and paper-ready content. Use `antclock.volte` and `antclock.volte_bridge` for programmatic access to Volte functionality.

## Directory Structure

### Volte System Definition
- **`volte_system_definition.py`** - Complete mathematical implementation
- **`volte_definition_archiiiv.md`** - Markdown version for direct paper inclusion (Definition 1)
- **`volte_definition_latex.tex`** - LaTeX version for academic publication
- **`volte_ce1_connection.py`** - CE1 framework bridge

### Benchmark Data & Results
- **`data/`** - Publication-ready benchmark results and dataset documentation
  - `benchmark_results/` - **Validated** CE framework performance data for publication
    - `ce_timing_results.json` - CE timing speedup (2.3x) and accuracy (4.6x) results
    - `mirror_phase_classification_result.json` - Mirror phase classification results
    - `real_benchmark_results.json` - Real-world benchmark performance
    - `section6_empirical_results.json` - Section 6 empirical validation
  - `datasets/` - Benchmark dataset references and methodologies
- **`sync.py`** - Sync validated data from `.data/` to `data/` for publication

### Development Output Convention
- **`.out/`** - Development outputs, intermediate results, and generated artifacts
  - `ce_benchmark_comprehensive_report.json` - Comprehensive benchmark suite (development)
  - `ce_benchmark_verification_summary.json` - Verification summary (development)
  - Various timing and classification results accumulate here during development

## Key White Paper Components

### Definition 1: Volte System
The general Volte equation provides a unifying mathematical framework for coherence-preserving transformations across biological, immunological, and psychological domains.

**Core Equation:**
```
dx/dt = F(x,u) + 𝓥(x,u)
```

**Axioms (V1-V3):**
- (V1) Invariant preservation: Q(x + ε 𝓥) = Q(x)
- (V2) Harm reduction, coherence enhancement
- (V3) Threshold-triggered activation

### Specializations
- **Evolution/ERVs**: x = genomic architecture, Q = species identity, S = maladaptive load
- **Immune Fields**: x = immune cell population, Q = self-recognition, S = viral load
- **Psychological**: x = narrative state, Q = core values, S = stigma pressure

### Benchmark Results
- CE timing evaluation: 2.3x faster convergence, 4.6x accuracy improvement
- Verified CE1/CE2/CE3 mathematical consistency
- Dataset diversity confirmed across SCAN, COGS, CFQ, PCFG, RPM benchmarks

## Usage

### For Paper Writing
```bash
# View the main definition
cat volte_definition_archiiiv.md

# Check benchmark results
cat data/benchmark_results/ce_timing_results.json
```

### For Development & Research
```bash
# Run main AntClock demo (comprehensive CE1/CE2/CE3 showcase)
cd ../demos && python antclock.py

# Run Volte system demo (coherence-preserving transformations)
cd ../demos && python volte.py

# Use integrated Volte system in AntClock
cd ..
python -c "from antclock import create_walker; walker = create_walker(enable_volte=True); print('Volte-enabled AntClock created')"

# Run CE benchmark verification (Phase 1 of complete pipeline)
./run.sh benchmarks/benchmark.py
```

### For Validation
```bash
# Test integrated Volte implementation
python -c "from antclock.volte import AntClockVolteSystem; print('Core Volte system loaded')"

# Test CE1 bridge
python -c "from antclock.volte_bridge import VolteCE1Bridge; bridge = VolteCE1Bridge(); print('CE1-Volte bridge ready')"
```

## File Formats

- **`.py`** - Executable implementations with CE1-aligned mathematics
- **`.md`** - Markdown for direct paper inclusion
- **`.tex`** - LaTeX for academic publication
- **`data/`** - JSON benchmark results and documentation

## Syncing Validated Data

After running benchmarks and manually validating results:

```bash
# Dry run to see what would be synced
python archiX/sync.py --dry-run

# Sync all validated datasets
python archiX/sync.py

# Sync only specific dataset
python archiX/sync.py --dataset scan

# Force sync without prompts
python archiX/sync.py --force
```

This moves validated data from `.data/` to `archiX/data/` for publication.

## Integration with Main Repository

This directory contains white paper content only. The main AntClock repository provides:
- Core CE framework implementation
- Benchmark infrastructure
- Biological instantiations
- Development tools

## Citation

When using archiX content in publications:

```
The Volte system is defined in Definition 1 of the AntClock framework,
providing a mathematical formalism for coherence-preserving transformations
across biological, immunological, and psychological domains.
```

---

*"I went through hell and came out more myself, not less."*

— The Volte equation formalizes this universal pattern of guardian turns.
