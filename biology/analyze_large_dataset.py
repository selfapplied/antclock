#!/usr/bin/env python3
"""
Analyze large ERV dataset (500 sequences) and extract key insights.

This script processes the full analysis results and provides
statistical summaries and pattern detection.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_large_dataset(analysis_file: Path):
    """Analyze large dataset results."""
    print("=" * 70)
    print("LARGE DATASET ANALYSIS (500 Sequences)")
    print("=" * 70)
    
    if not analysis_file.exists():
        print(f"❌ Analysis file not found: {analysis_file}")
        print("   Run: python biology/erv/analyze_erv.py biology/data/genbank/erv_sequences.fasta --output {analysis_file}")
        return
    
    with open(analysis_file, 'r') as f:
        data = json.load(f)
    
    num_sequences = data.get('num_sequences', len(data.get('analyses', [])))
    analyses = data.get('analyses', [])
    
    print(f"\n📊 Dataset Size: {num_sequences} sequences")
    print(f"   Analyses completed: {len(analyses)}")
    
    if not analyses:
        print("⚠️ No analyses found in file")
        return
    
    # Extract metrics
    stresses = []
    coherences = []
    volte_activations = []
    nash_data = []
    exaptation_potentials = []
    
    for analysis in analyses:
        transform = analysis.get('transform', {})
        stresses.append(transform.get('stress_S', 0))
        coherences.append(transform.get('coherence_C', 0))
        volte_activations.append(transform.get('volte_activated', False))
        
        if 'nash_equilibrium' in transform:
            nash_data.append(transform['nash_equilibrium'])
        
        witness = analysis.get('witness', {})
        exaptation_potentials.append(witness.get('exaptation_potential', 0))
    
    # Statistical summary
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    
    print(f"\n1. Stress (S):")
    print(f"   Mean: {np.mean(stresses):.3f}")
    print(f"   Std:  {np.std(stresses):.3f}")
    print(f"   Min:  {np.min(stresses):.3f}")
    print(f"   Max:  {np.max(stresses):.3f}")
    print(f"   Median: {np.median(stresses):.3f}")
    
    print(f"\n2. Coherence (C):")
    print(f"   Mean: {np.mean(coherences):.3f}")
    print(f"   Std:  {np.std(coherences):.3f}")
    print(f"   Min:  {np.min(coherences):.3f}")
    print(f"   Max:  {np.max(coherences):.3f}")
    print(f"   Median: {np.median(coherences):.3f}")
    
    print(f"\n3. Volte Activations:")
    volte_count = sum(volte_activations)
    print(f"   Activated: {volte_count}/{len(volte_activations)} ({100*volte_count/len(volte_activations):.1f}%)")
    print(f"   Threshold: 0.638")
    stress_above_threshold = sum(1 for s in stresses if s >= 0.638)
    print(f"   Stress ≥ threshold: {stress_above_threshold}/{len(stresses)} ({100*stress_above_threshold/len(stresses):.1f}%)")
    
    print(f"\n4. Exaptation Potential:")
    print(f"   Mean: {np.mean(exaptation_potentials):.3f}")
    print(f"   Std:  {np.std(exaptation_potentials):.3f}")
    print(f"   High (≥0.7): {sum(1 for e in exaptation_potentials if e >= 0.7)}/{len(exaptation_potentials)}")
    print(f"   Low (≤0.3):  {sum(1 for e in exaptation_potentials if e <= 0.3)}/{len(exaptation_potentials)}")
    
    # Nash equilibrium analysis
    if nash_data:
        print(f"\n5. Nash Equilibrium (n={len(nash_data)}):")
        g_values = [n['G'] for n in nash_data]
        g_crit_values = [n['g_crit'] for n in nash_data]
        hurst_values = [n['hurst'] for n in nash_data]
        should_exapt = [n['should_exapt'] for n in nash_data]
        should_protect = [n['should_protect'] for n in nash_data]
        
        print(f"   Composition Gain (G):")
        print(f"      Mean: {np.mean(g_values):.3f}")
        print(f"      Std:  {np.std(g_values):.3f}")
        print(f"      Range: [{np.min(g_values):.3f}, {np.max(g_values):.3f}]")
        
        print(f"   Critical Threshold (G_crit):")
        print(f"      Mean: {np.mean(g_crit_values):.3f}")
        print(f"      Std:  {np.std(g_crit_values):.3f}")
        
        print(f"   Hurst Exponent:")
        print(f"      Mean: {np.mean(hurst_values):.3f}")
        print(f"      Std:  {np.std(hurst_values):.3f}")
        print(f"      Interpretation: ", end="")
        if np.mean(hurst_values) > 0.6:
            print("High memory (conservative coupling needed)")
        elif np.mean(hurst_values) < 0.4:
            print("Low memory (flexible coupling)")
        else:
            print("Moderate memory (balanced)")
        
        exapt_count = sum(should_exapt)
        protect_count = sum(should_protect)
        print(f"\n   Recommendations:")
        print(f"      Exapt: {exapt_count}/{len(nash_data)} ({100*exapt_count/len(nash_data):.1f}%)")
        print(f"      Protect: {protect_count}/{len(nash_data)} ({100*protect_count/len(nash_data):.1f}%)")
        
        # G vs G_crit comparison
        g_above_crit = sum(1 for g, g_crit in zip(g_values, g_crit_values) if g >= g_crit)
        print(f"\n   G ≥ G_crit: {g_above_crit}/{len(nash_data)} ({100*g_above_crit/len(nash_data):.1f}%)")
        if g_above_crit > len(nash_data) * 0.8:
            print(f"      💡 Strong pattern: Most sequences recommend protection")
        elif g_above_crit < len(nash_data) * 0.2:
            print(f"      💡 Strong pattern: Most sequences allow exaptation")
        else:
            print(f"      💡 Mixed pattern: Balanced protection/exaptation")
    
    # Pattern detection
    print("\n" + "=" * 70)
    print("PATTERN DETECTION")
    print("=" * 70)
    
    # Stress-Coherence correlation
    if len(stresses) > 1:
        correlation = np.corrcoef(stresses, coherences)[0, 1]
        print(f"\n1. Stress-Coherence Correlation: {correlation:.3f}")
        if abs(correlation) < 0.1:
            print("   💡 Very low correlation → Independent dimensions")
            print("      High coherence possible even with high stress")
        elif correlation < 0:
            print("   💡 Negative correlation → Stress reduces coherence")
        else:
            print("   💡 Positive correlation → Stress and coherence linked")
    
    # Coherence distribution
    high_coherence = sum(1 for c in coherences if c >= 0.7)
    print(f"\n2. Coherence Distribution:")
    print(f"   High (≥0.7): {high_coherence}/{len(coherences)} ({100*high_coherence/len(coherences):.1f}%)")
    print(f"   Medium (0.4-0.7): {sum(1 for c in coherences if 0.4 <= c < 0.7)}/{len(coherences)}")
    print(f"   Low (<0.4): {sum(1 for c in coherences if c < 0.4)}/{len(coherences)}")
    
    # Stress distribution
    high_stress = sum(1 for s in stresses if s >= 0.638)
    print(f"\n3. Stress Distribution:")
    print(f"   High (≥0.638): {high_stress}/{len(stresses)} ({100*high_stress/len(stresses):.1f}%)")
    print(f"   Medium (0.3-0.638): {sum(1 for s in stresses if 0.3 <= s < 0.638)}/{len(stresses)}")
    print(f"   Low (<0.3): {sum(1 for s in stresses if s < 0.3)}/{len(stresses)}")
    
    # Summary insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    insights = []
    
    if np.mean(coherences) > 0.6:
        insights.append("✅ High average coherence suggests stable ERV structures")
    
    if np.mean(stresses) < 0.5:
        insights.append("✅ Moderate stress levels suggest system stability")
    
    if volte_count < len(volte_activations) * 0.1:
        insights.append("✅ Low Volte activation suggests system not in crisis")
    
    if nash_data and sum(should_protect) > len(nash_data) * 0.7:
        insights.append("✅ Nash equilibrium favors protection → conservative regime")
    
    if nash_data and np.mean([n['hurst'] for n in nash_data]) > 0.5:
        insights.append("✅ Moderate-high Hurst suggests evolutionary memory")
    
    for insight in insights:
        print(f"\n{insight}")
    
    print("\n" + "=" * 70)
    print("COMPARISON: 10 vs 500 Sequences")
    print("=" * 70)
    
    # Load old 10-sequence analysis if available
    old_file = Path(__file__).parent / "data" / "genbank" / "genbank_analysis_with_nash.json"
    if old_file.exists():
        with open(old_file, 'r') as f:
            old_data = json.load(f)
        
        old_analyses = old_data.get('analyses', [])
        if old_analyses:
            old_stresses = [a['transform'].get('stress_S', 0) for a in old_analyses]
            old_coherences = [a['transform'].get('coherence_C', 0) for a in old_analyses]
            
            print(f"\n10 sequences:")
            print(f"   Avg Stress: {np.mean(old_stresses):.3f}")
            print(f"   Avg Coherence: {np.mean(old_coherences):.3f}")
            
            print(f"\n500 sequences:")
            print(f"   Avg Stress: {np.mean(stresses):.3f}")
            print(f"   Avg Coherence: {np.mean(coherences):.3f}")
            
            stress_diff = np.mean(stresses) - np.mean(old_stresses)
            coherence_diff = np.mean(coherences) - np.mean(old_coherences)
            
            print(f"\nDifference:")
            print(f"   Stress: {stress_diff:+.3f}")
            print(f"   Coherence: {coherence_diff:+.3f}")
            
            if abs(stress_diff) < 0.05 and abs(coherence_diff) < 0.05:
                print("\n💡 Patterns are consistent across sample sizes!")
            else:
                print("\n💡 Patterns shift with larger sample - more diversity revealed")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze large ERV dataset")
    parser.add_argument('analysis_file', type=Path, 
                       help='Path to analysis JSON file')
    
    args = parser.parse_args()
    analyze_large_dataset(args.analysis_file)




