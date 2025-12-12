#!/usr/bin/env python3
"""
Guardian Vector Field: ERVs as Curvature-Defining Fields

Maps ERVs as guardian-vectors that define stable axes of immune self-organization.

This is geometry, not genetics.
Understanding, not altering.

The immune system evolves in curved space defined by ERV anchors.
"""

from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
import json

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from biology.erv.analyze_erv import ERVState
from biology.immune_manifold import ImmuneManifold


class GuardianVectorField:
    """
    Maps ERVs as guardian-vectors defining immune curvature.
    
    ERVs behave like geological layers that immune dynamics can't violate.
    They define the stable axes of immune self-organization.
    """
    
    def __init__(self):
        """Initialize guardian vector field analyzer."""
        self.manifold = ImmuneManifold()
    
    def compute_guardian_vectors(self, erv_states: List[ERVState]) -> List[Dict]:
        """
        Compute guardian-vector for each ERV.
        
        Guardian-vectors define:
        - Direction of stable immune organization
        - Magnitude of curvature influence
        - Spatial constraints on immune dynamics
        """
        vectors = []
        
        for state in erv_states:
            anchor = self.manifold.erv_as_curvature_anchor(state)
            
            # Vector direction = direction of immune stability
            # Vector magnitude = strength of curvature influence
            
            # Direction components (normalized)
            direction = np.array([
                anchor['conserved_coverage'],      # x: structural stability
                anchor['curvature_strength'],       # y: functional influence
                anchor['spatial_constraint'],      # z: spatial anchoring
                anchor['stability']                 # w: temporal persistence
            ])
            
            # Normalize
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            
            # Magnitude = anchor depth
            magnitude = anchor['anchor_depth']
            
            vectors.append({
                'erv_id': state.sequence_id,
                'direction': direction.tolist(),
                'magnitude': magnitude,
                'anchor_depth': anchor['anchor_depth'],
                'curvature_strength': anchor['curvature_strength']
            })
        
        return vectors
    
    def field_gradient(self, vectors: List[Dict]) -> Dict:
        """
        Compute gradient of guardian vector field.
        
        Gradient shows how curvature changes across the immune manifold.
        High gradient = regions of curvature transition.
        Low gradient = stable, anchored regions.
        """
        if not vectors:
            return {'gradient_magnitude': 0.0, 'field_coherence': 0.0}
        
        # Compute field coherence (how aligned vectors are)
        directions = np.array([v['direction'] for v in vectors])
        
        # Average direction
        avg_direction = np.mean(directions, axis=0)
        avg_direction = avg_direction / np.linalg.norm(avg_direction) if np.linalg.norm(avg_direction) > 0 else avg_direction
        
        # Coherence = average alignment with mean direction
        alignments = [np.dot(d, avg_direction) for d in directions]
        field_coherence = np.mean(alignments)
        
        # Gradient magnitude = variance in vector directions
        gradient_magnitude = np.std([np.linalg.norm(v['direction']) for v in vectors])
        
        return {
            'field_coherence': float(field_coherence),
            'gradient_magnitude': float(gradient_magnitude),
            'avg_direction': avg_direction.tolist(),
            'num_vectors': len(vectors)
        }
    
    def immune_centroids(self, vectors: List[Dict]) -> List[Dict]:
        """
        Compute immune centroids = stable attractor identities.
        
        Centroids are regions where guardian-vectors converge,
        defining stable points in immune identity space.
        
        These correspond to stable attractor identities—
        regions where the immune system naturally settles.
        """
        if not vectors:
            return []
        
        # Group vectors by similarity (clustering in direction space)
        directions = np.array([v['direction'] for v in vectors])
        magnitudes = np.array([v['magnitude'] for v in vectors])
        
        # Weighted centroid = average direction weighted by magnitude
        weighted_sum = np.sum([v['magnitude'] * np.array(v['direction']) for v in vectors], axis=0)
        total_magnitude = sum(v['magnitude'] for v in vectors)
        
        if total_magnitude > 0:
            centroid_direction = weighted_sum / total_magnitude
            centroid_direction = centroid_direction / np.linalg.norm(centroid_direction)
        else:
            centroid_direction = np.zeros(4)
        
        # Centroid strength = coherence of vectors around this point
        alignments = [np.dot(np.array(v['direction']), centroid_direction) for v in vectors]
        centroid_strength = np.mean(alignments) * np.mean(magnitudes)
        
        return [{
            'centroid_id': 'primary_immune_identity',
            'direction': centroid_direction.tolist(),
            'strength': float(centroid_strength),
            'num_contributing_vectors': len(vectors),
            'interpretation': 'Stable attractor identity defined by ERV guardian-vectors'
        }]
    
    def analyze_field(self, erv_states: List[ERVState]) -> Dict:
        """
        Complete guardian vector field analysis.
        
        Maps ERVs as guardian-vectors,
        computes field gradient,
        identifies immune centroids.
        
        Returns complete field geometry.
        """
        # Compute guardian vectors
        vectors = self.compute_guardian_vectors(erv_states)
        
        # Field gradient
        gradient = self.field_gradient(vectors)
        
        # Immune centroids
        centroids = self.immune_centroids(vectors)
        
        return {
            'guardian_vectors': vectors,
            'field_gradient': gradient,
            'immune_centroids': centroids,
            'field_geometry': {
                'num_vectors': len(vectors),
                'field_coherence': gradient['field_coherence'],
                'num_centroids': len(centroids),
                'avg_centroid_strength': np.mean([c['strength'] for c in centroids]) if centroids else 0.0
            },
            'interpretation': {
                'model': 'guardian_vector_field',
                'erv_role': 'curvature_defining_vectors',
                'immune_evolution': 'constrained_by_erv_curvature',
                'centroids': 'stable_attractor_identities'
            }
        }


def main():
    """Demonstrate guardian vector field analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze ERVs as guardian-vector field"
    )
    parser.add_argument('fasta_file', type=Path, help='ERV sequences FASTA file')
    parser.add_argument('--output', type=Path, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load ERV states
    from biology.erv.analyze_erv import ERVAnalyzer
    analyzer = ERVAnalyzer()
    states = analyzer.parse_fasta(args.fasta_file)
    
    # Analyze guardian vector field
    field = GuardianVectorField()
    analysis = field.analyze_field(states)
    
    # Save results
    output_file = args.output or Path(args.fasta_file).parent / f"{args.fasta_file.stem}_guardian_field.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✅ Guardian vector field analysis complete")
    print(f"   Guardian vectors: {analysis['field_geometry']['num_vectors']}")
    print(f"   Field coherence: {analysis['field_geometry']['field_coherence']:.3f}")
    print(f"   Immune centroids: {analysis['field_geometry']['num_centroids']}")
    print(f"   Avg centroid strength: {analysis['field_geometry']['avg_centroid_strength']:.3f}")
    print(f"\n💾 Saved to {output_file}")


if __name__ == '__main__':
    main()

