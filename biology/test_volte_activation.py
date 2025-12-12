#!/usr/bin/env python3
"""
Test Volte activation with high-stress sequences.

Creates sequences specifically designed to trigger Volte threshold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from biology.erv.analyze_erv import ERVState, ERVAnalyzer, ERVVolteSystem

def create_high_stress_sequence():
    """Create a sequence that will trigger Volte activation."""
    # High repeat density and low complexity = high stress
    # Create sequence with many repeated patterns
    base_pattern = "ATGC" * 50  # 200 bases
    repeated = (base_pattern * 3) + "AAAA" * 50  # High repeat density
    sequence = repeated[:800]  # 800 base sequence
    
    state = ERVState(
        sequence_id="high_stress_erv",
        sequence=sequence,
        integration_site=(1000, 1100),
        conserved_regions=[],  # No conserved regions = higher stress
        functional_annotations={'exapted': False}
    )
    
    return state

def test_volte_activation():
    """Test that high-stress sequences trigger Volte."""
    print("=" * 60)
    print("Testing Volte Activation with High-Stress Sequences")
    print("=" * 60)
    
    # Create high-stress sequence
    high_stress = create_high_stress_sequence()
    
    # Analyze
    analyzer = ERVAnalyzer()
    control = {'gene_proximity': 0.8}  # High proximity = more stress
    
    analysis = analyzer.analyze_erv(high_stress, control)
    
    print(f"\n📊 High-Stress Sequence Analysis:")
    print(f"   Sequence ID: {analysis['memory']['sequence_id']}")
    print(f"   Length: {analysis['memory']['length']}")
    print(f"   Stress (S): {analysis['transform']['stress_S']:.3f}")
    print(f"   Threshold (κ): {analysis['transform']['threshold']:.3f}")
    print(f"   Volte Activated: {analysis['transform']['volte_activated']}")
    
    if analysis['transform']['volte_activated']:
        print("\n✅ Volte activation triggered!")
        if 'next_state' in analysis['transform']:
            next_state = analysis['transform']['next_state']
            print(f"   Next state stress: {next_state['stress_S']:.3f}")
            print(f"   Stress reduced: {analysis['witness']['stress_reduced']}")
            print(f"   Coherence increased: {analysis['witness']['coherence_increased']}")
    else:
        print(f"\n⚠️ Volte not activated (stress {analysis['transform']['stress_S']:.3f} < threshold {analysis['transform']['threshold']:.3f})")
        print("   Creating even higher stress sequence...")
        
        # Try with even more repeats
        very_high_stress_seq = "AAAA" * 100 + "TTTT" * 100 + "GGGG" * 100 + "CCCC" * 100
        very_high_stress = ERVState(
            sequence_id="very_high_stress_erv",
            sequence=very_high_stress_seq,
            integration_site=(1000, 1100),
            conserved_regions=[],
            functional_annotations={}
        )
        
        analysis2 = analyzer.analyze_erv(very_high_stress, {'gene_proximity': 1.0})
        print(f"\n   Very high stress sequence:")
        print(f"   Stress: {analysis2['transform']['stress_S']:.3f}")
        print(f"   Volte Activated: {analysis2['transform']['volte_activated']}")

if __name__ == '__main__':
    test_volte_activation()





