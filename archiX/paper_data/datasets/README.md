# Benchmark Datasets

This directory contains references to the benchmark datasets used for evaluating the CE framework.

## Systematic Generalization Benchmarks

### SCAN (Sequence-to-Sequence with Actions and Navigation)
- **Location**: `../real_data/scan/`
- **Scale**: 16,728 train examples, 4,182 test examples
- **Task**: Systematic compositional generalization in command sequences
- **CE Results**: 2.3x faster convergence, 4.6x accuracy improvement

### COGS (Compositional Generalization Semantic Parsing)
- **Location**: `../real_data/COGS-main/`
- **Scale**: 24,155 train examples, 3,000 test examples
- **Task**: Semantic parsing with systematic compositionality
- **CE Results**: Evaluated for tensor dimension compatibility

### CFQ (Compositional Freebase Questions)
- **Location**: `../real_data/cfq/`
- **Scale**: ~100K train examples, ~10K test examples
- **Task**: Complex semantic parsing from natural language to SPARQL
- **CE Results**: Evaluated for compositional generalization

### PCFG (Probabilistic Context-Free Grammar)
- **Location**: Generated synthetically
- **Scale**: Configurable synthetic dataset
- **Task**: Grammar parsing with probabilistic structure
- **CE Results**: Evaluated for tensor dimension issues

### RPM/RAVEN (Raven's Progressive Matrices)
- **Location**: `../real_data/raven.zip`
- **Scale**: 10,000 train examples, 1,000 test examples
- **Task**: Visual pattern completion and analogy reasoning
- **CE Results**: Baseline performance established

### Mathematical Reasoning
- **Location**: `../real_data/math/`
- **Scale**: 1,000 train examples, 4 test examples
- **Task**: Mathematical pattern recognition and completion
- **CE Results**: Baseline accuracy measured

## CE Framework Evaluation

The CE framework was evaluated on these datasets to demonstrate:

1. **Systematic Generalization**: Ability to generalize beyond training distribution
2. **Compositional Understanding**: Handling of complex nested structures
3. **Timing Advantages**: Convergence speed improvements via CE awareness
4. **Mathematical Consistency**: Preservation of CE1/CE2/CE3 properties

## CE Timing Results Summary

```python
# From final_ce_timing_results.py
SCAN Results:
- Baseline: 41.62s (2 epochs), 2.7% accuracy
- CE Timing: 19.29s (1 epoch, early stopped), 12.4% accuracy
- Speedup: 2.3x faster
- Accuracy: 4.6x better

CE Features Demonstrated:
- Kappa Guardian Early Stopping
- Chi-FEG Learning Rate Scheduling
- Awareness Loop Optimization
- Phase-Locked Training
```

## Dataset Access

All datasets are included in the repository at `benchmarks/real_data/` for reproducibility.

## References

- Lake et al. "Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks" (SCAN)
- Kim & Linzen "COGS: A Compositional Generalization Challenge Based on Semantic Interpretation" (COGS)
- Keysers et al. "Measuring Compositional Generalization: A Comprehensive Method on Realistic Data" (CFQ)
- Carpenter et al. "Raven's Progressive Matrices" (RPM)
