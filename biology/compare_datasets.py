#!/usr/bin/env python3
"""
Compare ERV Datasets: Analyze patterns across different ERV types.

Compares:
- General ERV (1000 sequences)
- HERV (200 sequences)
- Primate ERV (200 sequences)
- Combined (1400 sequences)
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_analysis(file_path: Path) -> dict:
    """Load analysis JSON file."""
    if not file_path.exists():
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_statistics(analysis: dict) -> dict:
    """Extract key statistics from analysis."""
    if not analysis or 'analyses' not in analysis:
        return None
    
    analyses = analysis['analyses']
    if not analyses:
        return None
    
    stresses = []
    coherences = []
    hurst_values = []
    g_values = []
    g_crit_values = []
    should_protect = []
    
    for a in analyses:
        transform = a.get('transform', {})
        stresses.append(transform.get('stress_S', 0))
        coherences.append(transform.get('coherence_C', 0))
        
        nash = transform.get('nash_equilibrium', {})
        if nash:
            hurst_values.append(nash.get('hurst', 0.5))
            g_values.append(nash.get('G', 0))
            g_crit_values.append(nash.get('g_crit', 0))
            should_protect.append(nash.get('should_protect', False))
    
    return {
        'num_sequences': len(analyses),
        'avg_stress': float(np.mean(stresses)) if stresses else 0,
        'std_stress': float(np.std(stresses)) if stresses else 0,
        'avg_coherence': float(np.mean(coherences)) if coherences else 0,
        'std_coherence': float(np.std(coherences)) if coherences else 0,
        'avg_hurst': float(np.mean(hurst_values)) if hurst_values else 0,
        'std_hurst': float(np.std(hurst_values)) if hurst_values else 0,
        'avg_g': float(np.mean(g_values)) if g_values else 0,
        'avg_g_crit': float(np.mean(g_crit_values)) if g_crit_values else 0,
        'protect_rate': float(np.mean(should_protect)) if should_protect else 0,
        'protect_count': sum(should_protect) if should_protect else 0
    }


def compare_datasets():
    """Compare all ERV datasets."""
    data_dir = Path(__file__).parent / "data" / "genbank"
    
    datasets = {
        'General ERV (1000)': data_dir / "genbank_analysis_500.json",  # We have 500 analyzed
        'HERV (200)': data_dir / "herv_analysis.json",
        'Primate ERV (200)': None,  # Not analyzed yet
    }
    
    print("=" * 70)
    print("ERV DATASET COMPARISON")
    print("=" * 70)
    
    results = {}
    
    for name, file_path in datasets.items():
        print(f"\n📊 {name}:")
        
        if file_path is None:
            print("   ⏳ Not analyzed yet")
            continue
        
        analysis = load_analysis(file_path)
        stats = extract_statistics(analysis)
        
        if stats:
            results[name] = stats
            print(f"   Sequences: {stats['num_sequences']}")
            print(f"   Avg Stress: {stats['avg_stress']:.3f} ± {stats['std_stress']:.3f}")
            print(f"   Avg Coherence: {stats['avg_coherence']:.3f} ± {stats['std_coherence']:.3f}")
            print(f"   Avg Hurst: {stats['avg_hurst']:.3f} ± {stats['std_hurst']:.3f}")
            print(f"   Protection Rate: {stats['protect_rate']:.1%} ({stats['protect_count']}/{stats['num_sequences']})")
        else:
            print("   ⏳ Analysis file not found or incomplete")
    
    # Comparison
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        names = list(results.keys())
        for i in range(len(names) - 1):
            name1, name2 = names[i], names[i+1]
            stats1, stats2 = results[name1], results[name2]
            
            print(f"\n{name1} vs {name2}:")
            
            # Hurst comparison
            h_diff = stats2['avg_hurst'] - stats1['avg_hurst']
            print(f"   Hurst: {stats1['avg_hurst']:.3f} → {stats2['avg_hurst']:.3f} ({h_diff:+.3f})")
            
            # Coherence comparison
            c_diff = stats2['avg_coherence'] - stats1['avg_coherence']
            print(f"   Coherence: {stats1['avg_coherence']:.3f} → {stats2['avg_coherence']:.3f} ({c_diff:+.3f})")
            
            # Protection comparison
            p_diff = stats2['protect_rate'] - stats1['protect_rate']
            print(f"   Protection: {stats1['protect_rate']:.1%} → {stats2['protect_rate']:.1%} ({p_diff:+.1%})")
    
    return results


def main():
    """Main function."""
    results = compare_datasets()
    
    # Save comparison
    output_file = Path(__file__).parent / "data" / "genbank" / "dataset_comparison.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Comparison saved to {output_file}")


if __name__ == '__main__':
    main()




