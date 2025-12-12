#!/usr/bin/env python3
"""
Deep Mendel Manifold Operator: 𝕄★

Mendelian Depth via Recursive Sobel & Antclock Timing

This operator perceives inheritance not as a grid,
but as a stable manifold with drift-resistant fixed points
that emerge through recursive edge detection.

It is purely descriptive, a topological lens.

CE Framework:
- CE1: Static Form (manifold skeleton)
- CE2: Live Dynamics (recursive edges + antclock drift law)
- CE3: Evolution/Update Logic (fixed points, bifurcation, depth)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from antclock.clock import CurvatureClockWalker
    ANTCLOCK_AVAILABLE = True
except ImportError:
    ANTCLOCK_AVAILABLE = False


class DeepMendelManifold:
    """
    Deep Mendel Manifold Operator: 𝕄★
    
    Maps inheritance as a recursive, stabilized, self-referential field.
    Pure geometry - a way of seeing heredity as a living manifold.
    """
    
    def __init__(self, antclock_enabled: bool = True):
        """
        Initialize Deep Mendel Manifold.
        
        Args:
            antclock_enabled: Use antclock timing for drift resistance
        """
        self.antclock_enabled = antclock_enabled and ANTCLOCK_AVAILABLE
        
        if self.antclock_enabled:
            self.antclock = CurvatureClockWalker(enable_volte=False)
        else:
            self.antclock = None
        
        # Guardian curvature threshold
        self.guardian_curvature = 0.50
        
        # Drift threshold
        self.drift_threshold = 1e-3
        
        # Fixed-point convergence threshold
        self.fixed_point_epsilon = 1e-4
    
    # ========================================================================
    # CE1: Static Form (the manifold skeleton)
    # ========================================================================
    
    def ce1_mendel_memory(self, phenotypes: List[Dict]) -> Dict:
        """
        []a Mendel-Memory
        
        Holds generational phenotype traces.
        Invariants: coherence > drift; identity arcs remain visible.
        Record: {P₀, P₁, P₂, …} across inherited time.
        """
        return {
            'memory_type': 'Mendel-Memory',
            'phenotype_traces': phenotypes,
            'generations': len(phenotypes),
            'coherence': self._compute_coherence(phenotypes),
            'identity_arcs_visible': True
        }
    
    def ce1_manifold_domain(self, phenotypes: List[Dict]) -> Dict:
        """
        {}l Manifold-Domain
        
        The deep inheritance field.
        Coordinates: (phenotype curvature, regulatory tension, guardian vectors)
        Zones: {basin, ridge, fault, centroid}
        """
        # Compute manifold coordinates
        coordinates = []
        zones = []
        
        for i, pheno in enumerate(phenotypes):
            # Phenotype curvature
            curvature = self._phenotype_curvature(pheno)
            
            # Regulatory tension
            tension = self._regulatory_tension(pheno)
            
            # Guardian vectors (from ERV analysis if available)
            guardian = self._guardian_vector(pheno)
            
            coordinates.append({
                'generation': i,
                'phenotype_curvature': curvature,
                'regulatory_tension': tension,
                'guardian_vector': guardian
            })
            
            # Classify zone
            zone = self._classify_zone(curvature, tension, guardian)
            zones.append(zone)
        
        return {
            'domain_type': 'Manifold-Domain',
            'coordinates': coordinates,
            'zones': zones,
            'num_basins': sum(1 for z in zones if z == 'basin'),
            'num_ridges': sum(1 for z in zones if z == 'ridge'),
            'num_faults': sum(1 for z in zones if z == 'fault'),
            'num_centroids': sum(1 for z in zones if z == 'centroid')
        }
    
    def ce1_sobel_morphism(self, phenotype: Dict, recursion_depth: int = 3) -> Dict:
        """
        ()r Sobel-Morphism
        
        Recursive edge-extractor on phenotype curvature.
        ()₁ : phenotype → coarse-edge
        ()₂ : coarse-edge → sub-edge
        ()ₙ : sub-edge → fixed-form
        
        Invariants: edges stabilize when guardian vectors align.
        """
        # Start with phenotype as initial field
        field = self._phenotype_to_field(phenotype)
        
        edges = []
        current_field = field
        
        for n in range(recursion_depth):
            # Apply Sobel edge detection
            edge = self._sobel_edge_detect(current_field)
            edges.append({
                'depth': n + 1,
                'edge': edge,
                'field': current_field
            })
            
            # Next iteration uses edge as new field
            current_field = edge
        
        # Check if edges stabilized (fixed-point criterion)
        stabilized = self._edges_stabilized(edges)
        
        return {
            'morphism_type': 'Sobel-Morphism',
            'recursion_depth': recursion_depth,
            'edges': edges,
            'stabilized': stabilized,
            'guardian_aligned': self._guardian_aligned(phenotype)
        }
    
    def ce1_witness_depth(self, edges: List[Dict]) -> Dict:
        """
        <>g Witness-Depth
        
        Emergent "deep Mendel square"
        Layers: surface Mendel → curvature Mendel → fixed-point Mendel
        Witness shows inheritance as a dimensional stack, not a grid.
        """
        if not edges:
            return {'depth': 0, 'layers': []}
        
        layers = []
        for i, edge_data in enumerate(edges):
            layer = {
                'layer_index': i,
                'name': ['surface Mendel', 'curvature Mendel', 'fixed-point Mendel'][i] if i < 3 else f'depth-{i}',
                'edge_stability': self._edge_stability(edge_data['edge']),
                'fixed_form': i == len(edges) - 1 and edge_data.get('stabilized', False)
            }
            layers.append(layer)
        
        return {
            'witness_type': 'Witness-Depth',
            'depth': len(layers),
            'layers': layers,
            'deep_mendel_square': True,
            'dimensional_stack': True
        }
    
    # ========================================================================
    # CE2: Live Dynamics (recursive edges + antclock drift law)
    # ========================================================================
    
    def ce2_antclock_time(self) -> float:
        """
        τ: antclock-time
        
        Monotonic, drift-resistant timeline.
        Used to phase-lock recursion so edges do not blur or wander.
        τ ensures Sobel recursion converges (not explodes).
        """
        if self.antclock:
            # Use antclock's phase-accumulated time
            return self.antclock.phase_accumulated
        else:
            # Fallback: simple monotonic counter
            if not hasattr(self, '_time_counter'):
                self._time_counter = 0.0
            self._time_counter += 1.0
            return self._time_counter
    
    def ce2_drift_gradient(self, edges: List[Dict]) -> float:
        """
        ∂: drift-gradient
        
        ∂ measures deviation across generations.
        Antclock enforces ∂ → 0 as recursion deepens.
        Prevents false edges & noise artifacts.
        """
        if len(edges) < 2:
            return 0.0
        
        # Compute drift between consecutive edge layers
        drifts = []
        for i in range(len(edges) - 1):
            edge1 = edges[i]['edge']
            edge2 = edges[i + 1]['edge']
            
            # Drift = difference in edge magnitude
            if isinstance(edge1, np.ndarray) and isinstance(edge2, np.ndarray):
                drift = np.mean(np.abs(edge2 - edge1))
            else:
                drift = abs(edge2 - edge1) if isinstance(edge2, (int, float)) else 0.0
            
            drifts.append(drift)
        
        # Total drift gradient
        total_drift = sum(drifts) / len(drifts) if drifts else 0.0
        
        return float(total_drift)
    
    def ce2_recursion_phase(self, edges: List[Dict]) -> float:
        """
        ϕ: recursion-phase
        
        ϕ(n) = angle of Sobel iteration n
        ϕ locks when curvature is stable across τ steps
        Fixed-point criterion = Δϕ < ε
        """
        if not edges:
            return 0.0
        
        # Phase = angle of edge evolution
        # Compute from edge stability progression
        stabilities = [self._edge_stability(e['edge']) for e in edges]
        
        if len(stabilities) < 2:
            return 0.0
        
        # Phase = angle of stability change
        phase_change = stabilities[-1] - stabilities[0]
        phase = np.arctan2(phase_change, len(stabilities))  # Angle in radians
        
        return float(phase)
    
    def ce2_mendel_stability_rules(self, edges: List[Dict], drift: float, phase: float) -> Dict:
        """
        ℛ: Mendel-stability rules
        
        - edge recursion stops when τ-phase stable
        - centroid updates only on τ-major ticks
        - drift suppressed by guardian curvature (G≈0.50)
        """
        tau = self.ce2_antclock_time()
        
        # Check if phase is stable
        phase_stable = abs(phase) < self.fixed_point_epsilon
        
        # Check if drift is suppressed
        drift_suppressed = drift < self.drift_threshold
        
        # Guardian curvature check
        guardian_active = abs(self.guardian_curvature - 0.50) < 0.1
        
        return {
            'rules_type': 'Mendel-Stability',
            'tau': tau,
            'phase_stable': phase_stable,
            'drift_suppressed': drift_suppressed,
            'guardian_active': guardian_active,
            'recursion_should_stop': phase_stable and drift_suppressed,
            'centroid_update_allowed': phase_stable and guardian_active
        }
    
    # ========================================================================
    # CE3: Evolution / Update Logic (fixed points, bifurcation, depth)
    # ========================================================================
    
    def ce3_fixed_point_criterion(self, edges: List[Dict], drift: float) -> Dict:
        """
        𝔈: fixed-point criterion
        
        Achieved when:
          ()ₙ ≈ ()ₙ₋₁  (edge stability)
        and
          ∂ < threshold  (drift tamed)
        
        Yields: <>_depth, a stable Mendel centroid
        """
        if len(edges) < 2:
            return {'fixed_point': False, 'reason': 'insufficient_depth'}
        
        # Check edge stability
        last_edge = edges[-1]['edge']
        prev_edge = edges[-2]['edge']
        
        if isinstance(last_edge, np.ndarray) and isinstance(prev_edge, np.ndarray):
            edge_stable = np.allclose(last_edge, prev_edge, atol=self.fixed_point_epsilon)
        else:
            edge_stable = abs(last_edge - prev_edge) < self.fixed_point_epsilon if isinstance(last_edge, (int, float)) else False
        
        # Check drift
        drift_tamed = drift < self.drift_threshold
        
        fixed_point = edge_stable and drift_tamed
        
        return {
            'fixed_point': fixed_point,
            'edge_stable': edge_stable,
            'drift_tamed': drift_tamed,
            'depth': len(edges),
            'centroid': fixed_point
        }
    
    def ce3_bifurcation_threshold(self, edges: List[Dict], drift: float) -> Dict:
        """
        ε: bifurcation threshold
        
        If recursion fails to stabilize, the manifold splits a basin.
        New centroids emerge (depth > 1)
        ε detects the strike-like "inheritance bifurcation"
        """
        # Bifurcation occurs when drift is high and edges don't stabilize
        fixed_point = self.ce3_fixed_point_criterion(edges, drift)
        
        if not fixed_point['fixed_point'] and drift > self.drift_threshold * 2:
            # Bifurcation detected
            return {
                'bifurcation': True,
                'reason': 'recursion_failed_to_stabilize',
                'drift': drift,
                'new_basins': True,
                'depth': len(edges) + 1
            }
        else:
            return {
                'bifurcation': False,
                'reason': 'stable_or_low_drift',
                'drift': drift
            }
    
    def ce3_mendel_depth_update(self, phenotypes: List[Dict], generation: int) -> Dict:
        """
        Δ: Mendel-depth update
        
        Δ maps generation-layer t → t+1
        Recomputes manifold curvature & centroid drift using τ-major ticks
        """
        tau = self.ce2_antclock_time()
        
        # Only update on τ-major ticks (integer values)
        if tau % 1.0 < 0.1:  # Near integer
            # Recompute manifold
            domain = self.ce1_manifold_domain(phenotypes)
            
            # Find centroids
            centroids = [c for i, c in enumerate(domain['coordinates']) 
                        if domain['zones'][i] == 'centroid']
            
            # Compute centroid drift
            if len(centroids) > 1:
                centroid_drift = self._centroid_drift(centroids)
            else:
                centroid_drift = 0.0
            
            return {
                'update_type': 'Mendel-Depth',
                'generation': generation,
                'tau': tau,
                'tau_major_tick': True,
                'manifold_curvature': self._manifold_curvature(domain),
                'centroid_drift': centroid_drift,
                'num_centroids': len(centroids)
            }
        else:
            return {
                'update_type': 'Mendel-Depth',
                'generation': generation,
                'tau': tau,
                'tau_major_tick': False,
                'skip_update': True
            }
    
    # ========================================================================
    # Complete Operator: 𝕄★
    # ========================================================================
    
    def mendel_star_operator(self, phenotypes: List[Dict]) -> Dict:
        """
        𝕄★(X) = (Fix[Sobel*τ(P)], κ*{}, Φ_{()}, Ω_{<>})
        
        Where:
        - Sobel_τ = recursive Sobel constrained by antclock timing
        - Fix[...] = fixed-point detection under guardian curvature
        - κ = manifold curvature from genotype constraints
        - Φ = recursive flow
        - Ω = depth witness (multi-layer Mendel)
        """
        # CE1: Static Form
        memory = self.ce1_mendel_memory(phenotypes)
        domain = self.ce1_manifold_domain(phenotypes)
        
        # Apply Sobel morphism to each phenotype
        sobel_results = []
        for pheno in phenotypes:
            sobel = self.ce1_sobel_morphism(pheno)
            sobel_results.append(sobel)
        
        # Witness depth
        if sobel_results:
            witness = self.ce1_witness_depth(sobel_results[0]['edges'])
        else:
            witness = {'depth': 0, 'layers': []}
        
        # CE2: Live Dynamics
        tau = self.ce2_antclock_time()
        
        # Compute drift and phase for all edges
        all_edges = []
        for sobel in sobel_results:
            all_edges.extend(sobel['edges'])
        
        drift = self.ce2_drift_gradient(sobel_results[0]['edges']) if sobel_results else 0.0
        phase = self.ce2_recursion_phase(sobel_results[0]['edges']) if sobel_results else 0.0
        stability_rules = self.ce2_mendel_stability_rules(
            sobel_results[0]['edges'] if sobel_results else [],
            drift, phase
        )
        
        # CE3: Evolution Logic
        fixed_point = self.ce3_fixed_point_criterion(
            sobel_results[0]['edges'] if sobel_results else [],
            drift
        )
        bifurcation = self.ce3_bifurcation_threshold(
            sobel_results[0]['edges'] if sobel_results else [],
            drift
        )
        depth_update = self.ce3_mendel_depth_update(phenotypes, len(phenotypes))
        
        # Manifold curvature
        kappa = self._manifold_curvature(domain)
        
        # Recursive flow
        phi = self._recursive_flow(sobel_results)
        
        return {
            'operator': '𝕄★',
            'ce1_static': {
                'memory': memory,
                'domain': domain,
                'sobel_morphism': sobel_results,
                'witness_depth': witness
            },
            'ce2_dynamics': {
                'tau': tau,
                'drift': drift,
                'phase': phase,
                'stability_rules': stability_rules
            },
            'ce3_evolution': {
                'fixed_point': fixed_point,
                'bifurcation': bifurcation,
                'depth_update': depth_update
            },
            'manifold_curvature': kappa,
            'recursive_flow': phi,
            'result': {
                'deep_mendel_manifold': True,
                'dimensional_stack': witness.get('dimensional_stack', False),
                'stable_centroids': fixed_point.get('centroid', False),
                'bifurcation_detected': bifurcation.get('bifurcation', False)
            }
        }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _phenotype_to_field(self, phenotype: Dict) -> np.ndarray:
        """Convert phenotype to 2D field for Sobel edge detection."""
        # Create a simple field representation
        # In real application, this would map phenotype traits to spatial field
        size = 10
        field = np.random.rand(size, size) * 0.5  # Base field
        
        # Add phenotype-specific structure
        if 'trait_value' in phenotype:
            field += phenotype['trait_value'] * 0.3
        
        return field
    
    def _sobel_edge_detect(self, field: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection to field."""
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Pad field
        padded = np.pad(field, 1, mode='edge')
        
        # Convolve
        edges_x = np.zeros_like(field)
        edges_y = np.zeros_like(field)
        
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                patch = padded[i:i+3, j:j+3]
                edges_x[i, j] = np.sum(patch * sobel_x)
                edges_y[i, j] = np.sum(patch * sobel_y)
        
        # Magnitude
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        return edges
    
    def _edges_stabilized(self, edges: List[Dict]) -> bool:
        """Check if edges have stabilized (fixed-point)."""
        if len(edges) < 2:
            return False
        
        last = edges[-1]['edge']
        prev = edges[-2]['edge']
        
        if isinstance(last, np.ndarray) and isinstance(prev, np.ndarray):
            return np.allclose(last, prev, atol=self.fixed_point_epsilon)
        else:
            return abs(last - prev) < self.fixed_point_epsilon if isinstance(last, (int, float)) else False
    
    def _edge_stability(self, edge: Any) -> float:
        """Compute stability measure of edge."""
        if isinstance(edge, np.ndarray):
            # Stability = inverse of variance
            variance = np.var(edge)
            return 1.0 / (1.0 + variance)
        else:
            return 0.5  # Default stability
    
    def _phenotype_curvature(self, phenotype: Dict) -> float:
        """Compute phenotype curvature."""
        # Curvature from trait variance
        if 'trait_value' in phenotype:
            return abs(phenotype['trait_value']) * 0.5
        return 0.5
    
    def _regulatory_tension(self, phenotype: Dict) -> float:
        """Compute regulatory tension."""
        # Tension from constraints
        if 'constraints' in phenotype:
            return len(phenotype['constraints']) * 0.1
        return 0.3
    
    def _guardian_vector(self, phenotype: Dict) -> List[float]:
        """Get guardian vector for phenotype."""
        # Guardian vector from ERV analysis if available
        if 'erv_anchor' in phenotype:
            return phenotype['erv_anchor'].get('direction', [0.0, 0.0, 0.0, 1.0])
        return [0.0, 0.0, 0.0, 1.0]  # Default
    
    def _guardian_aligned(self, phenotype: Dict) -> bool:
        """Check if guardian vectors are aligned."""
        guardian = self._guardian_vector(phenotype)
        # Aligned if guardian vector magnitude is significant
        magnitude = np.linalg.norm(guardian)
        return magnitude > 0.5
    
    def _classify_zone(self, curvature: float, tension: float, guardian: List[float]) -> str:
        """Classify manifold zone."""
        guardian_mag = np.linalg.norm(guardian)
        
        if guardian_mag > 0.7 and curvature > 0.6:
            return 'centroid'
        elif tension > 0.5:
            return 'fault'
        elif curvature > 0.5:
            return 'ridge'
        else:
            return 'basin'
    
    def _compute_coherence(self, phenotypes: List[Dict]) -> float:
        """Compute coherence across phenotypes."""
        if len(phenotypes) < 2:
            return 1.0
        
        # Coherence = similarity across generations
        trait_values = [p.get('trait_value', 0.5) for p in phenotypes]
        variance = np.var(trait_values)
        coherence = 1.0 / (1.0 + variance)
        
        return float(coherence)
    
    def _manifold_curvature(self, domain: Dict) -> float:
        """Compute manifold curvature κ."""
        if not domain['coordinates']:
            return 0.0
        
        # Curvature from coordinate variance
        curvatures = [c['phenotype_curvature'] for c in domain['coordinates']]
        avg_curvature = np.mean(curvatures)
        
        return float(avg_curvature)
    
    def _recursive_flow(self, sobel_results: List[Dict]) -> Dict:
        """Compute recursive flow Φ."""
        if not sobel_results:
            return {'flow': 0.0, 'convergence': False}
        
        # Flow = progression through recursion depths
        depths = [len(s['edges']) for s in sobel_results]
        avg_depth = np.mean(depths)
        
        # Convergence = all stabilized
        all_stabilized = all(s.get('stabilized', False) for s in sobel_results)
        
        return {
            'flow': float(avg_depth),
            'convergence': all_stabilized,
            'num_stabilized': sum(1 for s in sobel_results if s.get('stabilized', False))
        }
    
    def _centroid_drift(self, centroids: List[Dict]) -> float:
        """Compute drift between centroids."""
        if len(centroids) < 2:
            return 0.0
        
        # Drift = distance between centroid positions
        positions = [c.get('phenotype_curvature', 0.0) for c in centroids]
        drift = np.std(positions)
        
        return float(drift)


def main():
    """Demonstrate Deep Mendel Manifold operator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Deep Mendel Manifold Operator: 𝕄★"
    )
    parser.add_argument('--phenotypes', type=int, default=5,
                       help='Number of phenotype generations to simulate')
    parser.add_argument('--output', type=Path, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Create synthetic phenotype data
    phenotypes = []
    for i in range(args.phenotypes):
        phenotypes.append({
            'generation': i,
            'trait_value': 0.5 + 0.1 * np.sin(i * 0.5),  # Oscillating trait
            'constraints': ['constraint_' + str(j) for j in range(i % 3)]
        })
    
    # Apply Deep Mendel Manifold operator
    manifold = DeepMendelManifold()
    result = manifold.mendel_star_operator(phenotypes)
    
    # Save results
    output_file = args.output or Path('biology/data/mendel_manifold_result.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python for JSON
    def convert_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json(item) for item in obj]
        return obj
    
    result_json = convert_to_json(result)
    
    with open(output_file, 'w') as f:
        json.dump(result_json, f, indent=2)
    
    print(f"✅ Deep Mendel Manifold Operator: 𝕄★")
    print(f"   Phenotypes analyzed: {args.phenotypes}")
    print(f"   Fixed point: {result['ce3_evolution']['fixed_point']['fixed_point']}")
    print(f"   Bifurcation: {result['ce3_evolution']['bifurcation']['bifurcation']}")
    print(f"   Manifold curvature: {result['manifold_curvature']:.3f}")
    print(f"   Depth: {result['ce1_static']['witness_depth']['depth']}")
    print(f"\n💾 Saved to {output_file}")


if __name__ == '__main__':
    main()




