#!/usr/bin/env python3
"""
Example workflow for ERV analysis using CE-Volte framework.

Demonstrates the complete pipeline from sequence analysis
to evolutionary dynamics understanding.
"""

from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from biology.erv.analyze_erv import ERVAnalyzer, ERVState
from biology.blast.analyze import BLASTAnalyzer


def example_erv_analysis():
    """Example: Analyze ERV sequences with Volte framework."""
    print("=" * 60)
    print("Example 1: ERV Volte System Analysis")
    print("=" * 60)
    
    # Create example ERV state
    example_sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    state = ERVState(
        sequence_id="example_erv_1",
        sequence=example_sequence,
        integration_site=(1000, 1050),
        conserved_regions=[(0, 20), (30, 50)],
        functional_annotations={'exapted': False, 'expression': 'low'}
    )
    
    analyzer = ERVAnalyzer()
    analysis = analyzer.analyze_erv(state)
    
    print(f"\nSequence: {state.sequence_id}")
    print(f"Length: {state.length}")
    print(f"\nVolte Components:")
    print(f"  Q (Invariant): {analysis['domain']['invariant_Q']}")
    print(f"  S (Stress): {analysis['transform']['stress_S']:.3f}")
    print(f"  C (Coherence): {analysis['transform']['coherence_C']:.3f}")
    print(f"  Volte Activated: {analysis['transform']['volte_activated']}")
    print(f"  Exaptation Potential: {analysis['witness']['exaptation_potential']:.3f}")
    
    if analysis['transform']['volte_activated']:
        print(f"\n⚠️  Stress exceeds threshold - exaptation may occur")
        if 'next_state' in analysis['transform']:
            next_state = analysis['transform']['next_state']
            print(f"  Next state stress: {next_state['stress_S']:.3f}")
            print(f"  Stress reduced: {analysis['witness']['stress_reduced']}")
            print(f"  Coherence increased: {analysis['witness']['coherence_increased']}")


def example_blast_integration():
    """Example: Integrate BLAST results with ERV analysis."""
    print("\n" + "=" * 60)
    print("Example 2: BLAST-ERV Integration")
    print("=" * 60)
    
    print("\nThis example would integrate BLAST results with ERV analysis.")
    print("In practice, you would:")
    print("  1. Run BLAST to find sequence similarities")
    print("  2. Extract conserved regions from alignments")
    print("  3. Use BLAST mismatches to inform stress calculations")
    print("  4. Analyze through unified CE-Volte framework")
    
    print("\nSee: python biology/erv/integrate_blast.py --help")


def example_workflow_summary():
    """Summary of the complete workflow."""
    print("\n" + "=" * 60)
    print("Complete ERV Analysis Workflow")
    print("=" * 60)
    
    workflow = """
1. Download ERV Sequences
   python biology/scripts/download_datasets.py --dataset genbank

2. Create BLAST Database
   python biology/blast/analyze.py --create-db sequences.fasta --db-name erv_db

3. Run BLAST Analysis
   python biology/blast/analyze.py query.fasta --db data/blast/erv_db

4. Analyze ERV with Volte Framework
   python biology/erv/analyze_erv.py sequences.fasta

5. Integrate BLAST and ERV Analysis
   python biology/erv/integrate_blast.py blast_results.txt sequences.fasta
"""
    print(workflow)


if __name__ == '__main__':
    example_erv_analysis()
    example_blast_integration()
    example_workflow_summary()
    
    print("\n" + "=" * 60)
    print("For more information, see:")
    print("  - biology/README.md")
    print("  - biology/SETUP.md")
    print("  - arXiv/working.md (Section 5.3: ERV Volte Systems)")
    print("=" * 60)





