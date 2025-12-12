#!/usr/bin/env python3
"""
Integration between BLAST analysis and ERV Volte systems.

Connects sequence similarity (BLAST) to evolutionary dynamics (Volte)
through the CE framework.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from biology.blast.analyze import BLASTAnalyzer
from biology.erv.analyze_erv import ERVAnalyzer, ERVState


class BLASTERVIntegrator:
    """
    Integrates BLAST sequence similarity with ERV Volte analysis.
    
    Uses BLAST results to inform:
    - Conserved regions (Q invariant)
    - Stress from mismatches (S functional)
    - Evolution potential (V exaptation)
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        
        self.blast_analyzer = BLASTAnalyzer(self.data_dir)
        self.erv_analyzer = ERVAnalyzer(self.data_dir)
    
    def blast_to_conserved_regions(self, blast_results: List[Dict], 
                                   query_length: int) -> List[tuple]:
        """
        Extract conserved regions from BLAST alignments.
        
        High identity regions indicate conserved structure (Q invariant).
        """
        if not blast_results:
            return []
        
        # Group by query position
        coverage = [0] * query_length
        
        for result in blast_results:
            qstart = result['query_start'] - 1  # Convert to 0-indexed
            qend = result['query_end']
            identity = result['identity']
            
            # Only count high-identity regions as conserved
            if identity > 80:  # 80% identity threshold
                for i in range(max(0, qstart), min(query_length, qend)):
                    coverage[i] = max(coverage[i], identity / 100.0)
        
        # Find contiguous conserved regions
        conserved = []
        in_region = False
        region_start = 0
        
        for i, cov in enumerate(coverage):
            if cov > 0.8 and not in_region:
                region_start = i
                in_region = True
            elif cov <= 0.8 and in_region:
                conserved.append((region_start, i))
                in_region = False
        
        if in_region:
            conserved.append((region_start, query_length))
        
        return conserved
    
    def blast_to_stress_control(self, blast_results: List[Dict]) -> Dict:
        """
        Convert BLAST results to stress control parameters.
        
        High mismatch rates indicate instability (S stress).
        Low evalue indicates strong similarity (reduces stress).
        """
        if not blast_results:
            return {'gene_proximity': 0.5}  # Default moderate stress
        
        # Average mismatch rate
        avg_mismatch = sum(r['mismatch'] for r in blast_results) / len(blast_results)
        avg_length = sum(r['length'] for r in blast_results) / len(blast_results)
        mismatch_rate = avg_mismatch / avg_length if avg_length > 0 else 0
        
        # Evalue indicates similarity strength
        min_evalue = min(r['evalue'] for r in blast_results)
        
        # High mismatch + low similarity = high stress
        stress_factor = mismatch_rate * (1.0 / (1.0 + min_evalue * 1e5))
        
        return {
            'gene_proximity': stress_factor,
            'mismatch_rate': mismatch_rate,
            'min_evalue': min_evalue
        }
    
    def integrate_analysis(self, blast_file: Path, query_file: Path) -> Dict:
        """
        Integrate BLAST results with ERV Volte analysis.
        
        Creates unified CE framework analysis combining:
        - BLAST CE analysis (bracket depth, coherence, evolution potential)
        - ERV Volte analysis (Q, S, C, V)
        """
        # Parse BLAST results
        blast_results = self.blast_analyzer.parse_blast_results(blast_file)
        blast_ce = self.blast_analyzer.analyze_with_ce(blast_results)
        
        # Parse ERV sequences
        erv_states = self.erv_analyzer.parse_fasta(query_file)
        
        integrated_results = {
            'blast_analysis': blast_ce,
            'erv_analyses': [],
            'integration': {}
        }
        
        for state in erv_states:
            # Extract conserved regions from BLAST
            conserved = self.blast_to_conserved_regions(blast_results, state.length)
            state.conserved_regions = conserved
            
            # Extract stress control from BLAST
            control = self.blast_to_stress_control(blast_results)
            
            # Run ERV Volte analysis
            erv_analysis = self.erv_analyzer.analyze_erv(state, control)
            
            integrated_results['erv_analyses'].append(erv_analysis)
        
        # Cross-integration insights
        integrated_results['integration'] = {
            'blast_bracket_depth_to_erv_invariant': {
                'blast_ce1_depth': blast_ce.get('bracket_depth', 0),
                'erv_invariant_preserved': all(
                    a['witness']['identity_preserved'] 
                    for a in integrated_results['erv_analyses']
                )
            },
            'blast_coherence_to_erv_coherence': {
                'blast_ce2_coherence': blast_ce.get('coherence', 0),
                'erv_avg_coherence': sum(
                    a['transform']['coherence_C'] 
                    for a in integrated_results['erv_analyses']
                ) / len(integrated_results['erv_analyses']) if integrated_results['erv_analyses'] else 0
            },
            'blast_evolution_to_erv_exaptation': {
                'blast_ce3_evolution_potential': blast_ce.get('evolution_potential', 0),
                'erv_avg_exaptation_potential': sum(
                    a['witness']['exaptation_potential'] 
                    for a in integrated_results['erv_analyses']
                ) / len(integrated_results['erv_analyses']) if integrated_results['erv_analyses'] else 0
            }
        }
        
        return integrated_results


def main():
    parser = argparse.ArgumentParser(
        description="Integrate BLAST analysis with ERV Volte systems"
    )
    parser.add_argument('blast_results', type=Path, 
                       help='BLAST output file (tab-separated format)')
    parser.add_argument('query_file', type=Path,
                       help='Query FASTA file (ERV sequences)')
    parser.add_argument('--output', type=Path,
                       help='Output JSON file for integrated analysis')
    
    args = parser.parse_args()
    
    if not args.blast_results.exists():
        print(f"❌ BLAST results file not found: {args.blast_results}")
        return
    
    if not args.query_file.exists():
        print(f"❌ Query file not found: {args.query_file}")
        return
    
    integrator = BLASTERVIntegrator()
    
    output_file = args.output or integrator.data_dir / "integrated_analysis.json"
    
    print(f"🔗 Integrating BLAST analysis with ERV Volte systems...")
    results = integrator.integrate_analysis(args.blast_results, args.query_file)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Integrated analysis saved to {output_file}")
    
    print(f"\n📊 Integration Summary:")
    print(f"   BLAST CE1 Bracket Depth: {results['blast_analysis'].get('bracket_depth', 0)}")
    print(f"   BLAST CE2 Coherence: {results['blast_analysis'].get('coherence', 0):.3f}")
    print(f"   BLAST CE3 Evolution Potential: {results['blast_analysis'].get('evolution_potential', 0):.3f}")
    print(f"   ERV Analyses: {len(results['erv_analyses'])}")
    
    integration = results['integration']
    print(f"\n🔗 Cross-Framework Insights:")
    print(f"   Identity Preserved: {integration['blast_bracket_depth_to_erv_invariant']['erv_invariant_preserved']}")
    print(f"   ERV Avg Coherence: {integration['blast_coherence_to_erv_coherence']['erv_avg_coherence']:.3f}")
    print(f"   ERV Avg Exaptation: {integration['blast_evolution_to_erv_exaptation']['erv_avg_exaptation_potential']:.3f}")


if __name__ == '__main__':
    main()





