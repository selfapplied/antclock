# AntClock Paper Data

This directory contains benchmark results and dataset references for the AntClock CE framework paper.

## Contents

### benchmark_results/
- `mirror_phase_classification_result.json` - Results from CE1 geometry benchmark demonstrating mirror phase classification accuracy

### datasets/
- References to benchmark datasets used in evaluation
- CE timing results summary

## Benchmark Results Interpretation

### Mirror Phase Classification
- **Dataset**: 50,000 integers from shells 1-10,000
- **Diversity Factors**: Shell range, curvature variation, entropy distribution, prime gaps
- **CE Layer**: CE1 (discrete grammar)
- **Accuracy**: Baseline performance metrics for mirror-phase shell classification

## Key Paper Results

### CE Timing Evaluation
From `final_ce_timing_results.py`:
- SCAN benchmark: 2.3x faster convergence, 4.6x better accuracy
- CE features: Kappa Guardian early stopping, Chi-FEG learning rate scheduling
- Demonstrated across systematic generalization benchmarks

### Datasets Used
- **SCAN**: Systematic compositional generalization (16,728 train, 4,182 test)
- **COGS**: Compositional generalization semantic parsing (24,155 train, 3,000 test)
- **CFQ**: Compositional Freebase Questions (~100K train, ~10K test)
- **PCFG**: Probabilistic context-free grammar parsing
- **RPM/RAVEN**: Raven's Progressive Matrices (10K train, 1K test)
- **Math**: Mathematical reasoning pattern completion (1K train, 4 test)

## Usage

```bash
# Run CE timing results summary
python final_ce_timing_results.py

# View benchmark result details
cat benchmark_results/mirror_phase_classification_result.json

# Access original datasets
ls ../benchmarks/real_data/
```

## References

- [SCAN Dataset](https://github.com/brendenlake/SCAN)
- [COGS Dataset](https://github.com/nlp-research/COGS)
- [CFQ Dataset](https://github.com/google-research/google-research/tree/master/cfq)
- [Raven's Progressive Matrices](https://www.ravenprogressivematrices.com/)

---

*Data supporting the CE1→CE2→CE3 framework for discrete Riemann geometry.*
