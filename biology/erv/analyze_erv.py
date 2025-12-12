#!/usr/bin/env python3
"""
ERV analysis using Volte systems framework.

Models ERVs as Volte systems where:
- x = lineage genomic architecture state
- Q = species identity / conserved core genes
- S = maladaptive load / instability
- V = exaptation: viral element → function while preserving lineage identity

Integrates with CE framework for causal compositional analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Nash equilibrium analysis
from biology.erv.nash_equilibrium import ERVNashEquilibrium, estimate_composition_gain

# Import directly from module to avoid triggering antclock/__init__.py
# which imports learner.py (requires torch)
import importlib.util
volte_path = Path(__file__).parent.parent.parent / "antclock" / "volte.py"
if not volte_path.exists():
    raise ImportError(f"volte.py not found at {volte_path}. Use './run.sh {__file__}' to ensure proper environment.")

spec = importlib.util.spec_from_file_location("antclock.volte", volte_path)
volte_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(volte_module)
DiscreteVolteSystem = volte_module.DiscreteVolteSystem


class ERVState:
    """Genomic architecture state for ERV analysis."""
    
    def __init__(self, sequence_id: str, sequence: str, 
                 integration_site: Optional[Tuple[int, int]] = None,
                 conserved_regions: Optional[List[Tuple[int, int]]] = None,
                 functional_annotations: Optional[Dict[str, Any]] = None):
        self.sequence_id = sequence_id
        self.sequence = sequence
        self.integration_site = integration_site
        self.conserved_regions = conserved_regions or []
        self.functional_annotations = functional_annotations or {}
        self.length = len(sequence)
    
    def __repr__(self):
        return f"ERVState(id={self.sequence_id}, len={self.length})"


class ERVVolteSystem:
    """
    Volte system for ERV evolution and exaptation.
    
    Models the dynamics of ERV integration and functionalization
    through the Volte framework.
    
    Uses DiscreteVolteSystem from antclock.volte when available,
    falls back to standalone implementation otherwise.
    """
    
    def __init__(self, threshold: float = 0.638):
        """
        Initialize ERV Volte system.
        
        Args:
            threshold: Activation threshold κ (defaults to chi_feg = 0.638)
        """
        self.threshold = threshold
        
        # Set up DiscreteVolteSystem
        self._setup_volte_system()
    
    def _setup_volte_system(self):
        """Set up DiscreteVolteSystem with ERV-specific functions."""
        def step_operator(state: ERVState, control: Dict[str, Any]) -> ERVState:
            """Ordinary evolution F_Δ: minimal change."""
            return ERVState(
                sequence_id=state.sequence_id,
                sequence=state.sequence,
                integration_site=state.integration_site,
                conserved_regions=state.conserved_regions.copy(),
                functional_annotations=state.functional_annotations.copy()
            )
        
        def invariant(state: ERVState) -> Dict[str, Any]:
            """Guardian charge Q."""
            return self.invariant_Q(state)
        
        def stress(state: ERVState, control: Dict[str, Any]) -> float:
            """Stress functional S."""
            return self.stress_S(state, control)
        
        def coherence(state: ERVState) -> float:
            """Coherence functional C."""
            return self.coherence_C(state)
        
        self.volte_system = DiscreteVolteSystem(
            step_operator=step_operator,
            invariant=invariant,
            stress=stress,
            coherence=coherence,
            threshold=self.threshold
        )
    
    def invariant_Q(self, state: ERVState) -> Dict[str, Any]:
        """
        Guardian charge Q: species identity / conserved core genes.
        
        Measures preservation of essential genomic structure:
        - Conserved region coverage
        - Core gene proximity
        - Structural stability
        """
        if not state.conserved_regions:
            return {
                'conserved_coverage': 0.0,
                'stability_score': 0.5,  # Neutral if unknown
                'identity_preserved': True
            }
        
        total_conserved = sum(end - start for start, end in state.conserved_regions)
        coverage = total_conserved / state.length if state.length > 0 else 0.0
        
        # Stability from structural features
        gc_content = self._gc_content(state.sequence)
        stability = 0.5 + 0.3 * abs(gc_content - 0.5)  # Prefer moderate GC
        
        return {
            'conserved_coverage': coverage,
            'stability_score': stability,
            'identity_preserved': coverage > 0.1  # At least 10% conserved
        }
    
    def stress_S(self, state: ERVState, control: Dict[str, Any]) -> float:
        """
        Stress functional S: maladaptive load / instability.
        
        Measures genomic stress from ERV integration:
        - Integration site disruption
        - Sequence instability (repeats, low complexity)
        - Functional interference
        """
        stress = 0.0
        
        # Integration site stress
        if state.integration_site:
            # Stress increases with proximity to genes
            if 'gene_proximity' in control:
                stress += control['gene_proximity'] * 0.3
        
        # Sequence instability
        repeat_density = self._repeat_density(state.sequence)
        stress += repeat_density * 0.4
        
        # Low complexity regions
        low_complexity = self._low_complexity_fraction(state.sequence)
        stress += low_complexity * 0.3
        
        # Normalize to [0, 1]
        return min(stress, 1.0)
    
    def coherence_C(self, state: ERVState) -> float:
        """
        Coherence functional C: internal fit / stability.
        
        Measures structural coherence:
        - Sequence quality
        - Functional annotation consistency
        - Integration stability
        """
        coherence = 0.5  # Base coherence
        
        # Sequence quality
        if state.length > 100:
            coherence += 0.2
        
        # Functional annotations suggest coherence
        if state.functional_annotations:
            if 'exapted' in state.functional_annotations:
                coherence += 0.2
            if 'expression' in state.functional_annotations:
                coherence += 0.1
        
        return min(coherence, 1.0)
    
    def exaptation_V(self, state: ERVState, control: Dict[str, Any]) -> ERVState:
        """
        Volte correction V: exaptation transformation.
        
        Models the reorientation that incorporates viral element
        into functional role while preserving species identity.
        """
        # Create new state with exaptation
        new_state = ERVState(
            sequence_id=state.sequence_id + "_exapted",
            sequence=state.sequence,
            integration_site=state.integration_site,
            conserved_regions=state.conserved_regions.copy(),
            functional_annotations=state.functional_annotations.copy()
        )
        
        # Mark as exapted
        new_state.functional_annotations['exapted'] = True
        new_state.functional_annotations['exaptation_potential'] = self._exaptation_potential(state)
        
        return new_state
    
    def step(self, state: ERVState, control: Dict[str, Any]) -> ERVState:
        """
        Discrete Volte step for ERV evolution.
        
        x_{t+1} = x_t + F_Δ(x_t, u_t) + V_Δ(x_t, u_t)
        
        where F_Δ is ordinary evolution and V_Δ is exaptation.
        
        Uses DiscreteVolteSystem from antclock.volte.
        """
        # Use DiscreteVolteSystem
        # Note: DiscreteVolteSystem expects state + base_step, but ERVState
        # doesn't support addition. We'll handle this manually.
        base_step = self.volte_system.step_operator(state, control)
        stress_level = self.volte_system.stress(state, control)
        
        if stress_level <= self.threshold:
            return base_step  # No Volte correction
        
        # Apply Volte correction
        return self.exaptation_V(base_step, control)
    
    def _gc_content(self, sequence: str) -> float:
        """Calculate GC content."""
        if not sequence:
            return 0.0
        gc = sum(1 for base in sequence.upper() if base in 'GC')
        return gc / len(sequence)
    
    def _repeat_density(self, sequence: str, k: int = 3) -> float:
        """Estimate repeat density from k-mer frequency."""
        if len(sequence) < k:
            return 0.0
        
        kmers = {}
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k].upper()
            kmers[kmer] = kmers.get(kmer, 0) + 1
        
        # High repeat density = many repeated kmers
        repeated = sum(1 for count in kmers.values() if count > 1)
        return repeated / len(kmers) if kmers else 0.0
    
    def _low_complexity_fraction(self, sequence: str) -> float:
        """Estimate low complexity regions."""
        if not sequence:
            return 0.0
        
        # Simple heuristic: homopolymer runs
        max_run = 1
        current_run = 1
        for i in range(1, len(sequence)):
            if sequence[i].upper() == sequence[i-1].upper():
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        # Normalize by sequence length
        return min(max_run / len(sequence), 1.0)
    
    def _exaptation_potential(self, state: ERVState) -> float:
        """Estimate exaptation potential from sequence features."""
        potential = 0.0
        
        # Length suggests functional potential
        if state.length > 500:
            potential += 0.3
        
        # Conserved regions suggest importance
        if state.conserved_regions:
            potential += 0.3
        
        # Structural features
        gc = self._gc_content(state.sequence)
        if 0.3 < gc < 0.7:  # Moderate GC suggests stability
            potential += 0.2
        
        # Integration site stability
        if state.integration_site:
            potential += 0.2
        
        return min(potential, 1.0)


class ERVAnalyzer:
    """Main analyzer for ERV sequences using Volte framework."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.volte_system = ERVVolteSystem()
        self.nash_analyzer = ERVNashEquilibrium()
        self.stress_history = []  # Track stress for Hurst estimation
    
    def parse_fasta(self, fasta_file: Path) -> List[ERVState]:
        """Parse FASTA file into ERVState objects."""
        states = []
        current_id = None
        current_seq = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id and current_seq:
                        states.append(ERVState(
                            sequence_id=current_id,
                            sequence=''.join(current_seq)
                        ))
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # Don't forget last sequence
            if current_id and current_seq:
                states.append(ERVState(
                    sequence_id=current_id,
                    sequence=''.join(current_seq)
                ))
        
        return states
    
    def analyze_erv(self, state: ERVState, control: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze ERV through Volte framework.
        
        Returns CE1-structured analysis:
        - [] memory: state history
        - {} domain: constraints and invariants
        - () transform: evolution step
        - <> witness: Volte conditions
        """
        if control is None:
            control = {}
        
        # Compute Volte components
        Q = self.volte_system.invariant_Q(state)
        S = self.volte_system.stress_S(state, control)
        C = self.volte_system.coherence_C(state)
        
        # Update stress history for Hurst estimation
        self.stress_history.append(S)
        if len(self.stress_history) > 100:  # Keep last 100 values
            self.stress_history = self.stress_history[-100:]
        
        # Check Volte activation
        volte_activated = S > self.volte_system.threshold
        
        # Nash equilibrium analysis
        composition_gain = estimate_composition_gain(S, C, self.volte_system.threshold)
        nash_analysis = self.nash_analyzer.analyze_erv_exaptation(
            composition_gain=composition_gain,
            stress_history=self.stress_history,
            current_stress=S
        )
        
        # CE1 bracket structure
        analysis = {
            # [] Memory: state log
            'memory': {
                'sequence_id': state.sequence_id,
                'length': state.length,
                'integration_site': state.integration_site,
                'timestamp': 'current'
            },
            
            # {} Domain: constraints
            'domain': {
                'invariant_Q': Q,
                'conserved_regions': state.conserved_regions,
                'functional_annotations': state.functional_annotations
            },
            
            # () Transform: evolution step
            'transform': {
                'stress_S': S,
                'coherence_C': C,
                'volte_activated': volte_activated,
                'threshold': self.volte_system.threshold,
                'nash_equilibrium': nash_analysis
            },
            
            # <> Witness: Volte conditions
            'witness': {
                'identity_preserved': Q['identity_preserved'],
                'stress_reduced': False,  # Would be True after Volte step
                'coherence_increased': False,  # Would be True after Volte step
                'exaptation_potential': self.volte_system._exaptation_potential(state)
            }
        }
        
        # If Volte activated, compute next state
        if volte_activated:
            next_state = self.volte_system.step(state, control)
            next_Q = self.volte_system.invariant_Q(next_state)
            next_S = self.volte_system.stress_S(next_state, control)
            next_C = self.volte_system.coherence_C(next_state)
            
            analysis['transform']['next_state'] = {
                'sequence_id': next_state.sequence_id,
                'stress_S': next_S,
                'coherence_C': next_C,
                'invariant_Q': next_Q
            }
            
            analysis['witness']['stress_reduced'] = next_S < S
            analysis['witness']['coherence_increased'] = next_C > C
            analysis['witness']['identity_preserved'] = next_Q['identity_preserved']
        
        return analysis
    
    def analyze_file(self, input_file: Path, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Analyze ERV sequences from FASTA file."""
        states = self.parse_fasta(input_file)
        
        results = {
            'input_file': str(input_file),
            'num_sequences': len(states),
            'analyses': []
        }
        
        for state in states:
            analysis = self.analyze_erv(state)
            results['analyses'].append(analysis)
        
        # Summary statistics
        results['summary'] = {
            'volte_activated_count': sum(1 for a in results['analyses'] 
                                        if a['transform']['volte_activated']),
            'avg_stress': sum(a['transform']['stress_S'] for a in results['analyses']) / len(results['analyses']) if results['analyses'] else 0,
            'avg_coherence': sum(a['transform']['coherence_C'] for a in results['analyses']) / len(results['analyses']) if results['analyses'] else 0,
            'avg_exaptation_potential': sum(a['witness']['exaptation_potential'] for a in results['analyses']) / len(results['analyses']) if results['analyses'] else 0
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Analysis saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ERV sequences using Volte systems framework"
    )
    parser.add_argument('input', type=Path, help='Input FASTA file with ERV sequences')
    parser.add_argument('--output', type=Path, help='Output JSON file for analysis results')
    parser.add_argument('--threshold', type=float, default=0.638,
                       help='Volte activation threshold (default: 0.638 = chi_feg)')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        return
    
    analyzer = ERVAnalyzer()
    analyzer.volte_system.threshold = args.threshold
    
    output_file = args.output or analyzer.data_dir / f"{args.input.stem}_erv_analysis.json"
    
    print(f"🔬 Analyzing ERV sequences from {args.input}...")
    results = analyzer.analyze_file(args.input, output_file)
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Sequences analyzed: {results['num_sequences']}")
    print(f"   Volte activations: {results['summary']['volte_activated_count']}")
    print(f"   Average stress: {results['summary']['avg_stress']:.3f}")
    print(f"   Average coherence: {results['summary']['avg_coherence']:.3f}")
    print(f"   Average exaptation potential: {results['summary']['avg_exaptation_potential']:.3f}")


if __name__ == '__main__':
    main()

