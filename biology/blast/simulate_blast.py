#!/usr/bin/env python3
"""
Simulate BLAST results for testing when BLAST+ is not installed.

Generates realistic BLAST output format for testing the integration pipeline.
"""

import random
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biology.erv.analyze_erv import ERVAnalyzer, ERVState


def simulate_blast_results(query_states: List[ERVState], 
                          num_hits_per_query: int = 3) -> List[Dict]:
    """
    Generate simulated BLAST results.
    
    Creates realistic BLAST output with:
    - High identity alignments (conserved regions)
    - Varied evalues
    - Mismatch patterns
    """
    results = []
    
    for state in query_states:
        for i in range(num_hits_per_query):
            # Simulate alignment
            identity = random.uniform(75, 95)  # 75-95% identity
            length = random.randint(int(state.length * 0.3), int(state.length * 0.8))
            mismatch = int(length * (1 - identity / 100))
            
            # Query alignment positions
            qstart = random.randint(1, max(1, state.length - length))
            qend = qstart + length
            
            # Subject (database) positions
            sstart = random.randint(1, 1000)
            send = sstart + length
            
            # E-value (lower = better match)
            evalue = 10 ** random.uniform(-10, -3)
            
            # Bitscore
            bitscore = identity * length / 100
            
            results.append({
                'query_id': state.sequence_id,
                'subject_id': f'simulated_hit_{i+1}',
                'identity': identity,
                'length': length,
                'mismatch': mismatch,
                'gapopen': random.randint(0, 2),
                'query_start': qstart,
                'query_end': qend,
                'subject_start': sstart,
                'subject_end': send,
                'evalue': evalue,
                'bitscore': bitscore
            })
    
    return results


def write_blast_output(results: List[Dict], output_file: Path):
    """Write BLAST results in tab-separated format (outfmt 6)."""
    with open(output_file, 'w') as f:
        for r in results:
            f.write(f"{r['query_id']}\t{r['subject_id']}\t{r['identity']:.2f}\t"
                   f"{r['length']}\t{r['mismatch']}\t{r['gapopen']}\t"
                   f"{r['query_start']}\t{r['query_end']}\t"
                   f"{r['subject_start']}\t{r['subject_end']}\t"
                   f"{r['evalue']:.2e}\t{r['bitscore']:.1f}\n")
    
    print(f"✅ Simulated BLAST results written to {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate BLAST results for testing")
    parser.add_argument('query_file', type=Path, help='Query FASTA file')
    parser.add_argument('--output', type=Path, help='Output BLAST results file')
    parser.add_argument('--hits-per-query', type=int, default=3, help='Number of hits per query')
    
    args = parser.parse_args()
    
    # Parse query sequences
    analyzer = ERVAnalyzer()
    states = analyzer.parse_fasta(args.query_file)
    
    # Generate simulated results
    results = simulate_blast_results(states, args.hits_per_query)
    
    # Write output
    output_file = args.output or Path(args.query_file).parent / f"{args.query_file.stem}_simulated_blast.txt"
    write_blast_output(results, output_file)
    
    print(f"\n📊 Generated {len(results)} simulated BLAST hits for {len(states)} queries")

