# archiX: White Paper Content

This directory contains all materials for the AntClock white paper/archiiiv publication.

## Directory Structure

### Volte System Definition
- **`volte_system_definition.py`** - Complete mathematical implementation with protocols and classes
- **`volte_definition_archiiiv.md`** - Markdown version for direct paper inclusion (Definition 1)
- **`volte_definition_latex.tex`** - LaTeX version for academic publication
- **`volte_ce1_connection.py`** - CE1 framework bridge and biological instantiations

### Benchmark Data & Results
- **`paper_data/`** - Complete benchmark results and dataset documentation
  - `benchmark_results/` - Verified CE framework performance data
  - `datasets/` - Benchmark dataset references and methodologies

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
cat paper_data/benchmark_results/ce_timing_results.json

# Run Volte system examples
./volte_system_definition.py
```

### For Validation
```bash
# Run CE benchmark verification
cd .. && ./run.sh verify_benchmarks.py

# Test Volte implementation
./run.sh -- "from archiX.volte_system_definition import *; print('Volte system loaded')"
```

## File Formats

- **`.py`** - Executable implementations with CE1-aligned mathematics
- **`.md`** - Markdown for direct paper inclusion
- **`.tex`** - LaTeX for academic publication
- **`paper_data/`** - JSON benchmark results and documentation

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
