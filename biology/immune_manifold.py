#!/usr/bin/env python3
"""
Immune Manifold: ERVs as Structural Anchors in Immune Topology

This module explores the conceptual architecture where ERVs act as
curvature-holding structures in the immune system's evolutionary landscape.

Not a model for manipulation—a geometric reframing of immune identity
as a historical structure built on viral fossils.

Key concepts:
- ERVs = boundary conditions (curvature anchors)
- Immune pathways = flows in curved space
- Self/non-self = mapping inside viral-curved manifold
- Tolerance = curvature alignment
- Inflammation = tension gradient expression
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json

# Add parent directory for imports
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup - use relative imports within biology module
from biology.erv.analyze_erv import ERVState, ERVAnalyzer
try:
    from biology.erv.nash_equilibrium import ERVNashEquilibrium
except ImportError:
    # Fallback if not available
    ERVNashEquilibrium = None


class ImmuneManifold:
    """
    Models the immune system as a curved manifold shaped by ERV anchors.
    
    ERVs define the stable axes of immune self-organization—
    not as active agents, but as structural constraints that hold curvature.
    
    Like ancient riverbeds shaping present-day flow,
    ERVs are the old topography of the immune landscape.
    """
    
    def __init__(self):
        """Initialize the immune manifold analyzer."""
        self.erv_analyzer = ERVAnalyzer()
        self.nash_analyzer = ERVNashEquilibrium()
    
    def erv_as_curvature_anchor(self, erv_state: ERVState) -> Dict:
        """
        Map ERV to curvature anchor in immune manifold.
        
        ERVs act as boundary conditions that define:
        - Regulatory scaffolding
        - Timing cues
        - Structural constraints
        - Bias fields for coherence
        - Developmental anchoring
        
        Returns curvature signature of this ERV anchor.
        """
        # Conserved regions = stable curvature points
        conserved_coverage = sum(
            (end - start) for start, end in erv_state.conserved_regions
        ) / erv_state.length if erv_state.length > 0 else 0.0
        
        # Functional annotations = curvature influence
        curvature_strength = 0.0
        if erv_state.functional_annotations:
            if erv_state.functional_annotations.get('exapted'):
                curvature_strength += 0.5  # Exapted = strong anchor
            if 'expression' in erv_state.functional_annotations:
                curvature_strength += 0.3  # Expression = active influence
        
        # Integration site = spatial constraint
        spatial_constraint = 1.0 if erv_state.integration_site else 0.5
        
        # Sequence stability = curvature persistence
        gc_content = sum(1 for b in erv_state.sequence.upper() if b in 'GC') / erv_state.length
        stability = 0.5 + 0.3 * abs(gc_content - 0.5)  # Moderate GC = stable
        
        return {
            'curvature_anchor': True,
            'conserved_coverage': conserved_coverage,
            'curvature_strength': min(curvature_strength, 1.0),
            'spatial_constraint': spatial_constraint,
            'stability': stability,
            'anchor_depth': conserved_coverage * stability * curvature_strength
        }
    
    def immune_pathway_flow(self, erv_anchors: List[Dict]) -> Dict:
        """
        Compute immune pathway flows in ERV-curved space.
        
        Pathways flow along existing viral-derived channels.
        Some flows are constrained by ERV curvature.
        Some flow easily along ERV-anchored routes.
        
        Returns flow field characteristics.
        """
        if not erv_anchors:
            return {'flow_coherence': 0.0, 'curvature_constraints': []}
        
        # Flow coherence = how well pathways align with ERV structure
        avg_anchor_depth = sum(a['anchor_depth'] for a in erv_anchors) / len(erv_anchors)
        flow_coherence = avg_anchor_depth
        
        # Curvature constraints = regions where flow is constrained
        curvature_constraints = [
            {
                'anchor_index': i,
                'constraint_strength': a['curvature_strength'],
                'spatial_region': a.get('spatial_constraint', 0.5)
            }
            for i, a in enumerate(erv_anchors)
            if a['curvature_strength'] > 0.3
        ]
        
        return {
            'flow_coherence': flow_coherence,
            'curvature_constraints': curvature_constraints,
            'num_anchors': len(erv_anchors),
            'avg_anchor_depth': avg_anchor_depth
        }
    
    def self_non_self_mapping(self, erv_states: List[ERVState]) -> Dict:
        """
        Map self/non-self as relative concepts in viral-curved manifold.
        
        If the architecture of self is partially viral in origin,
        then some immune tolerances are really ancestral treaties.
        
        Returns mapping of self-identity in ERV-curved space.
        """
        # Analyze each ERV as potential self-identity anchor
        identity_anchors = []
        
        for state in erv_states:
            anchor = self.erv_as_curvature_anchor(state)
            
            # Self-identity strength = how deeply ERV is integrated
            q = self.erv_analyzer.volte_system.invariant_Q(state)
            identity_strength = q.get('conserved_coverage', 0.0) * anchor['curvature_strength']
            
            identity_anchors.append({
                'erv_id': state.sequence_id,
                'identity_strength': identity_strength,
                'anchor_depth': anchor['anchor_depth'],
                'is_ancestral_treaty': anchor['curvature_strength'] > 0.5
            })
        
        # Self-identity = weighted sum of ERV anchors
        total_identity = sum(a['identity_strength'] for a in identity_anchors)
        ancestral_treaties = sum(1 for a in identity_anchors if a['is_ancestral_treaty'])
        
        return {
            'self_identity_strength': total_identity / len(erv_states) if erv_states else 0.0,
            'ancestral_treaties': ancestral_treaties,
            'identity_anchors': identity_anchors,
            'self_as_layered_archive': True  # Philosophical reframing
        }
    
    def curvature_conflict_analysis(self, erv_states: List[ERVState]) -> Dict:
        """
        Analyze potential curvature conflicts (immune disorders as misalignment).
        
        Some immune tensions reflect misalignment between:
        - Inherited ERV curvature
        - Modern environmental load
        
        Curvature mismatch creates stress.
        Stress creates oscillations.
        
        Returns conflict signatures (conceptual, not diagnostic).
        """
        conflicts = []
        
        for state in erv_states:
            # Compute stress from Volte analysis
            stress = self.erv_analyzer.volte_system.stress_S(state, {})
            coherence = self.erv_analyzer.volte_system.coherence_C(state)
            
            # Curvature anchor
            anchor = self.erv_as_curvature_anchor(state)
            
            # Conflict = misalignment between anchor stability and current stress
            curvature_stability = anchor['anchor_depth']
            misalignment = abs(stress - (1.0 - curvature_stability))
            
            if misalignment > 0.3:  # Significant misalignment
                conflicts.append({
                    'erv_id': state.sequence_id,
                    'misalignment': misalignment,
                    'stress': stress,
                    'curvature_stability': curvature_stability,
                    'coherence': coherence,
                    'conflict_type': 'curvature_mismatch'
                })
        
        return {
            'num_conflicts': len(conflicts),
            'conflicts': conflicts,
            'avg_misalignment': sum(c['misalignment'] for c in conflicts) / len(conflicts) if conflicts else 0.0,
            'interpretation': 'Curvature mismatch between inherited ERV structure and current state'
        }
    
    def analyze_immune_manifold(self, erv_states: List[ERVState]) -> Dict:
        """
        Complete immune manifold analysis.
        
        Maps ERVs as curvature anchors,
        computes pathway flows,
        analyzes self/non-self mapping,
        and identifies curvature conflicts.
        
        Returns complete geometric reframing of immune system.
        """
        # Map ERVs as curvature anchors
        anchors = [self.erv_as_curvature_anchor(state) for state in erv_states]
        
        # Compute pathway flows
        flow_field = self.immune_pathway_flow(anchors)
        
        # Self/non-self mapping
        identity_mapping = self.self_non_self_mapping(erv_states)
        
        # Curvature conflicts
        conflicts = self.curvature_conflict_analysis(erv_states)
        
        return {
            'immune_manifold': {
                'num_erv_anchors': len(anchors),
                'avg_anchor_depth': sum(a['anchor_depth'] for a in anchors) / len(anchors) if anchors else 0.0,
                'flow_coherence': flow_field['flow_coherence'],
                'curvature_constraints': len(flow_field['curvature_constraints'])
            },
            'self_identity': identity_mapping,
            'pathway_flows': flow_field,
            'curvature_conflicts': conflicts,
            'interpretation': {
                'model_type': 'geometric_reframing',
                'erv_role': 'structural_anchors',
                'immune_system': 'historical_structure_on_viral_fossils',
                'self_non_self': 'relative_concepts_in_curved_manifold',
                'safety': 'conceptual_only_no_manipulation'
            }
        }


def main():
    """Demonstrate immune manifold analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze immune system as ERV-curved manifold"
    )
    parser.add_argument('fasta_file', type=Path, help='ERV sequences FASTA file')
    parser.add_argument('--output', type=Path, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load ERV states
    analyzer = ERVAnalyzer()
    states = analyzer.parse_fasta(args.fasta_file)
    
    # Analyze immune manifold
    manifold = ImmuneManifold()
    analysis = manifold.analyze_immune_manifold(states)
    
    # Save results
    output_file = args.output or Path(args.fasta_file).parent / f"{args.fasta_file.stem}_immune_manifold.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✅ Immune manifold analysis complete")
    print(f"   ERV anchors: {analysis['immune_manifold']['num_erv_anchors']}")
    print(f"   Flow coherence: {analysis['immune_manifold']['flow_coherence']:.3f}")
    print(f"   Self-identity strength: {analysis['self_identity']['self_identity_strength']:.3f}")
    print(f"   Curvature conflicts: {analysis['curvature_conflicts']['num_conflicts']}")
    print(f"\n💾 Saved to {output_file}")


if __name__ == '__main__':
    main()

