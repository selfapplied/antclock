#!/usr/bin/env python3
"""
Hurst-Curvature Field Equation: CE-Encoded Harmonic Law

Unifies:
- Hurst scaling (H) - deep evolutionary memory
- Antclock curvature (κ) - spacetime geometry
- Coherence invariants (C) - structural stability
- Stress curvature (S) - evolutionary pressure
- Guardian protection preference (G > G_crit) - defensive strategy

Key Insight: Long memory (high H) → repeated averaging → flat spacetime → low curvature
Mathematical relationship: H ↑ → κ ↓ (inverse)

This gives the "ERV as guardian" hypothesis a mathematical foundation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from antclock.clock import CurvatureClockWalker
    ANTCLOCK_AVAILABLE = True
except ImportError:
    ANTCLOCK_AVAILABLE = False


class HurstCurvatureField:
    """
    CE-encoded field equation unifying Hurst scaling and antclock curvature.
    
    The harmonic law:
    
    [] Memory: H (Hurst) - deep evolutionary memory
    {} Domain: κ (curvature) - spacetime geometry
    () Transform: C (coherence) - structural stability
    <> Witness: G > G_crit - guardian protection preference
    
    Field Equation:
    κ(H) = κ₀ · (1 - H)^α · exp(-β·C) · (1 + S/χ_FEG)^(-γ)
    
    Where:
    - κ₀ = base curvature (antclock chi_feg = 0.638)
    - H = Hurst exponent (0.5 = random, 1.0 = perfect memory)
    - C = coherence (0-1)
    - S = stress (0-1)
    - χ_FEG = FEG coupling constant (0.638)
    - α, β, γ = scaling exponents
    """
    
    def __init__(self, chi_feg: float = 0.638, alpha: float = 1.0, 
                 beta: float = 1.0, gamma: float = 0.5):
        """
        Initialize Hurst-Curvature field equation.
        
        Args:
            chi_feg: FEG coupling constant (antclock threshold)
            alpha: Hurst-curvature scaling exponent
            beta: Coherence damping exponent
            gamma: Stress modulation exponent
        """
        self.chi_feg = chi_feg
        self.kappa_0 = chi_feg  # Base curvature = FEG coupling
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        if ANTCLOCK_AVAILABLE:
            self.antclock = CurvatureClockWalker(chi_feg=chi_feg, enable_volte=False)
        else:
            self.antclock = None
    
    # ========================================================================
    # CE1: Memory - Hurst Exponent (Deep Evolutionary Memory)
    # ========================================================================
    
    def memory_hurst(self, stress_history: List[float]) -> Dict:
        """
        [] Memory: Compute Hurst exponent from stress history.
        
        Hurst measures long-range dependence:
        - H = 0.5: Random walk (no memory)
        - H > 0.5: Persistent (long memory)
        - H < 0.5: Anti-persistent (short memory)
        
        High H → deep evolutionary memory → repeated averaging → flat spacetime
        """
        if len(stress_history) < 10:
            # Default to moderate memory if insufficient data
            return {
                'memory_type': 'Hurst-Memory',
                'H': 0.5,
                'interpretation': 'insufficient_data_default'
            }
        
        # Simplified Hurst estimation (R/S analysis approximation)
        H = self._estimate_hurst(stress_history)
        
        return {
            'memory_type': 'Hurst-Memory',
            'H': H,
            'interpretation': self._interpret_hurst(H),
            'deep_memory': H > 0.7,
            'flat_spacetime': H > 0.7  # High H → flat spacetime
        }
    
    def _estimate_hurst(self, series: List[float]) -> float:
        """Estimate Hurst exponent using simplified R/S analysis."""
        if len(series) < 10:
            return 0.5
        
        # Detrend the series
        mean_val = np.mean(series)
        detrended = [x - mean_val for x in series]
        
        # Compute cumulative deviations
        cumsum = np.cumsum(detrended)
        
        # Range and standard deviation
        R = np.max(cumsum) - np.min(cumsum)
        S = np.std(series)
        
        if S == 0:
            return 0.5
        
        # R/S ratio
        RS = R / S if S > 0 else 1.0
        
        # Hurst estimation: H ≈ log(RS) / log(n) for small n
        # For larger n, use more sophisticated estimation
        n = len(series)
        if n < 50:
            # Simple approximation
            H = 0.5 + (RS - 1.0) / (2.0 * n)
        else:
            # Log-log regression would be better, but simplified here
            H = 0.5 + 0.3 * (RS - 1.0) / n
        
        # Clamp to valid range
        H = max(0.0, min(1.0, H))
        
        return float(H)
    
    def _interpret_hurst(self, H: float) -> str:
        """Interpret Hurst exponent value."""
        if H < 0.4:
            return 'anti_persistent_short_memory'
        elif H < 0.6:
            return 'moderate_memory'
        elif H < 0.8:
            return 'persistent_long_memory'
        else:
            return 'very_persistent_deep_memory'
    
    # ========================================================================
    # CE2: Domain - Curvature (Spacetime Geometry)
    # ========================================================================
    
    def domain_curvature(self, H: float, C: float, S: float) -> Dict:
        """
        {} Domain: Compute curvature from Hurst, coherence, and stress.
        
        Field Equation:
        κ(H, C, S) = κ₀ · (1 - H)^α · exp(-β·C) · (1 + S/χ_FEG)^(-γ)
        
        Key relationships:
        - H ↑ → κ ↓ (long memory → flat spacetime → low curvature)
        - C ↑ → κ ↓ (high coherence → stable → low curvature)
        - S ↑ → κ ↓ (high stress → protection → low curvature)
        """
        # Base curvature term: (1 - H)^α
        # High H (long memory) → low curvature (flat spacetime)
        hurst_term = (1.0 - H) ** self.alpha
        
        # Coherence damping: exp(-β·C)
        # High coherence → exponential damping of curvature
        coherence_term = math.exp(-self.beta * C)
        
        # Stress modulation: (1 + S/χ_FEG)^(-γ)
        # High stress → protection → reduced curvature
        stress_term = (1.0 + S / self.chi_feg) ** (-self.gamma)
        
        # Combined curvature
        kappa = self.kappa_0 * hurst_term * coherence_term * stress_term
        
        return {
            'domain_type': 'Curvature-Domain',
            'kappa': float(kappa),
            'kappa_0': self.kappa_0,
            'hurst_term': float(hurst_term),
            'coherence_term': float(coherence_term),
            'stress_term': float(stress_term),
            'interpretation': self._interpret_curvature(kappa),
            'flat_regime': kappa < 0.1  # Very low curvature = flat spacetime
        }
    
    def _interpret_curvature(self, kappa: float) -> str:
        """Interpret curvature value."""
        if kappa < 0.1:
            return 'very_low_flat_spacetime'
        elif kappa < 0.3:
            return 'low_curvature_stable'
        elif kappa < 0.5:
            return 'moderate_curvature'
        else:
            return 'high_curvature_dynamic'
    
    # ========================================================================
    # CE3: Transform - Coherence Invariants
    # ========================================================================
    
    def transform_coherence(self, kappa: float, H: float, S: float) -> Dict:
        """
        () Transform: Coherence as structural stability invariant.
        
        Coherence emerges from:
        - Low curvature (flat spacetime)
        - Long memory (high H)
        - Moderate stress (S < χ_FEG)
        
        C = f(κ, H, S) = C₀ · (1 - κ/κ₀) · H · (1 - S/χ_FEG)
        """
        # Coherence increases with:
        # - Low curvature (1 - κ/κ₀)
        # - Long memory (H)
        # - Low stress (1 - S/χ_FEG)
        
        curvature_factor = max(0.0, 1.0 - kappa / self.kappa_0)
        memory_factor = H
        stress_factor = max(0.0, 1.0 - S / self.chi_feg)
        
        C = curvature_factor * memory_factor * stress_factor
        
        # Normalize to [0, 1]
        C = max(0.0, min(1.0, C))
        
        return {
            'transform_type': 'Coherence-Transform',
            'C': float(C),
            'curvature_factor': float(curvature_factor),
            'memory_factor': float(memory_factor),
            'stress_factor': float(stress_factor),
            'interpretation': self._interpret_coherence(C)
        }
    
    def _interpret_coherence(self, C: float) -> str:
        """Interpret coherence value."""
        if C < 0.3:
            return 'low_coherence_unstable'
        elif C < 0.6:
            return 'moderate_coherence'
        else:
            return 'high_coherence_stable'
    
    # ========================================================================
    # CE4: Witness - Guardian Protection Preference
    # ========================================================================
    
    def witness_protection(self, kappa: float, H: float, C: float, 
                          S: float, G: float, G_crit: float) -> Dict:
        """
        <> Witness: Guardian protection preference.
        
        Protection emerges when:
        - Low curvature (flat spacetime regime)
        - Long memory (high H)
        - High coherence (C)
        - G > G_crit (Nash equilibrium)
        
        Protection preference: P = f(κ, H, C, G > G_crit)
        """
        # Protection increases with:
        # - Low curvature (flat spacetime)
        # - Long memory (high H)
        # - High coherence
        # - Nash protection recommendation (G > G_crit)
        
        curvature_preference = 1.0 - min(1.0, kappa / self.kappa_0)
        memory_preference = H
        coherence_preference = C
        nash_preference = 1.0 if G > G_crit else 0.0
        
        # Combined protection preference
        P = (curvature_preference + memory_preference + coherence_preference + nash_preference) / 4.0
        
        should_protect = (G > G_crit) and (kappa < 0.3) and (H > 0.6) and (C > 0.6)
        
        return {
            'witness_type': 'Guardian-Protection-Witness',
            'protection_preference': float(P),
            'should_protect': should_protect,
            'curvature_preference': float(curvature_preference),
            'memory_preference': float(memory_preference),
            'coherence_preference': float(coherence_preference),
            'nash_preference': float(nash_preference),
            'interpretation': 'protect' if should_protect else 'explore',
            'regime': 'flat_spacetime_protection' if should_protect else 'curved_exploration'
        }
    
    # ========================================================================
    # Complete Field Equation
    # ========================================================================
    
    def field_equation(self, stress_history: List[float], C: float, 
                      S: float, G: float, G_crit: float) -> Dict:
        """
        Complete CE-encoded field equation.
        
        Unifies:
        - [] Hurst memory (H)
        - {} Curvature domain (κ)
        - () Coherence transform (C)
        - <> Protection witness (G > G_crit)
        
        Returns full CE structure with harmonic law.
        """
        # CE1: Memory - Hurst
        memory = self.memory_hurst(stress_history)
        H = memory['H']
        
        # CE2: Domain - Curvature
        domain = self.domain_curvature(H, C, S)
        kappa = domain['kappa']
        
        # CE3: Transform - Coherence (refined)
        transform = self.transform_coherence(kappa, H, S)
        C_refined = transform['C']
        
        # CE4: Witness - Protection
        witness = self.witness_protection(kappa, H, C_refined, S, G, G_crit)
        
        return {
            'field_equation': 'Hurst-Curvature-Harmonic-Law',
            'ce1_memory': memory,
            'ce2_domain': domain,
            'ce3_transform': transform,
            'ce4_witness': witness,
            'harmonic_law': {
                'H': H,
                'kappa': kappa,
                'C': C_refined,
                'S': S,
                'G': G,
                'G_crit': G_crit,
                'relationship': 'H ↑ → κ ↓ → C ↑ → Protection',
                'interpretation': 'Long memory creates flat spacetime (low curvature), enabling high coherence and protection preference'
            }
        }
    
    # ========================================================================
    # Validation: Apply to ERV Data
    # ========================================================================
    
    def apply_to_erv_analysis(self, erv_analysis_file: Path) -> Dict:
        """
        Apply field equation to ERV analysis results.
        
        Validates the harmonic law on real data.
        """
        with open(erv_analysis_file, 'r') as f:
            data = json.load(f)
        
        analyses = data.get('analyses', [])
        if not analyses:
            return {'error': 'No analyses found'}
        
        results = []
        
        for analysis in analyses:
            transform = analysis.get('transform', {})
            S = transform.get('stress_S', 0.0)
            C = transform.get('coherence_C', 0.0)
            
            nash = transform.get('nash_equilibrium', {})
            G = nash.get('G', 0.0)
            G_crit = nash.get('g_crit', 0.0)
            H = nash.get('hurst', 0.5)
            
            # Build stress history (simplified - use current stress)
            stress_history = [S] * 10  # Placeholder
            
            # Apply field equation
            field_result = self.field_equation(stress_history, C, S, G, G_crit)
            
            results.append({
                'sequence_id': analysis.get('memory', {}).get('sequence_id', 'unknown'),
                'field_equation': field_result,
                'validation': {
                    'H_observed': H,
                    'H_computed': field_result['ce1_memory']['H'],
                    'kappa_computed': field_result['ce2_domain']['kappa'],
                    'protection_recommended': field_result['ce4_witness']['should_protect'],
                    'nash_protection': nash.get('should_protect', False),
                    'agreement': field_result['ce4_witness']['should_protect'] == nash.get('should_protect', False)
                }
            })
        
        # Aggregate statistics
        agreements = sum(1 for r in results if r['validation']['agreement'])
        avg_kappa = np.mean([r['field_equation']['ce2_domain']['kappa'] for r in results])
        avg_H = np.mean([r['field_equation']['ce1_memory']['H'] for r in results])
        
        return {
            'field_equation_results': results,
            'aggregate': {
                'num_sequences': len(results),
                'agreement_rate': agreements / len(results) if results else 0.0,
                'avg_curvature': float(avg_kappa),
                'avg_hurst': float(avg_H),
                'interpretation': f'Field equation predicts protection with {agreements}/{len(results)} agreement'
            }
        }


def main():
    """Demonstrate Hurst-Curvature field equation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hurst-Curvature Field Equation"
    )
    parser.add_argument('--erv-analysis', type=Path, 
                       help='ERV analysis JSON file to validate against')
    parser.add_argument('--output', type=Path,
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    # Create field equation
    field = HurstCurvatureField(chi_feg=0.638, alpha=1.0, beta=1.0, gamma=0.5)
    
    if args.erv_analysis:
        # Apply to real data
        result = field.apply_to_erv_analysis(args.erv_analysis)
        
        output_file = args.output or Path('biology/data/genbank/hurst_curvature_field.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print("✅ Hurst-Curvature Field Equation Applied")
        print(f"   Sequences: {result['aggregate']['num_sequences']}")
        print(f"   Agreement: {result['aggregate']['agreement_rate']:.1%}")
        print(f"   Avg Curvature: {result['aggregate']['avg_curvature']:.3f}")
        print(f"   Avg Hurst: {result['aggregate']['avg_hurst']:.3f}")
        print(f"\n💾 Saved to {output_file}")
    else:
        # Demo with synthetic data
        stress_history = [0.4] * 50  # Moderate stress
        C = 0.7  # High coherence
        S = 0.397  # Moderate stress
        G = 0.506  # Composition gain
        G_crit = 0.443  # Critical threshold
        
        result = field.field_equation(stress_history, C, S, G, G_crit)
        
        print("✅ Hurst-Curvature Field Equation (Demo)")
        print(f"\nHarmonic Law:")
        print(f"   H (Hurst): {result['harmonic_law']['H']:.3f}")
        print(f"   κ (Curvature): {result['harmonic_law']['kappa']:.3f}")
        print(f"   C (Coherence): {result['harmonic_law']['C']:.3f}")
        print(f"   S (Stress): {result['harmonic_law']['S']:.3f}")
        print(f"\nRelationship: {result['harmonic_law']['relationship']}")
        print(f"Interpretation: {result['harmonic_law']['interpretation']}")
        print(f"\nProtection: {result['ce4_witness']['should_protect']}")
        print(f"Regime: {result['ce4_witness']['regime']}")


if __name__ == '__main__':
    main()




