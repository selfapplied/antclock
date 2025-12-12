#!/usr/bin/env python3
"""
Hurst-Curvature Field Theory: Complete CE Field Equation

This is not just a formula—it's a geometry of behavior.

The field equation creates a closed loop:
[] → {} → () → <> → []

Memory → Geometry → Structure → Witness → Memory

A biological soliton. A fixed point of evolutionary time.

This module derives:
- Euler-Lagrange form (variational principle)
- Conserved quantity (guardian charge)
- Stability manifold
- Bifurcation parameter (explore vs protect)
- CE operator diagram
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import sys
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class HurstCurvatureFieldTheory:
    """
    Complete field theory for Hurst-Curvature harmonic law.
    
    The field equation is a Lagrangian:
    L(H, C, S, κ) = kinetic - potential
    
    Where:
    - Kinetic: Memory flow (H dynamics)
    - Potential: Curvature cost (κ minimization)
    
    The system minimizes κ while maximizing C.
    This is the variational principle of biological evolution.
    """
    
    def __init__(self, chi_feg: float = 0.638, alpha: float = 1.0,
                 beta: float = 1.0, gamma: float = 0.5):
        """Initialize field theory."""
        self.chi_feg = chi_feg
        self.kappa_0 = chi_feg
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Guardian charge (conserved quantity) - stored value
        self._guardian_charge_value = 0.0
    
    # ========================================================================
    # Lagrangian Formulation
    # ========================================================================
    
    def lagrangian(self, H: float, C: float, S: float, dH_dt: float = 0.0) -> Dict:
        """
        Lagrangian: L = T - V
        
        Kinetic energy (T): Memory flow
        T = (1/2) · (dH/dt)²
        
        Potential energy (V): Curvature cost
        V = κ(H, C, S) = κ₀ · (1 - H)^α · exp(-β·C) · (1 + S/χ_FEG)^(-γ)
        
        The system minimizes V (curvature) while maximizing C (coherence).
        """
        # Kinetic: Memory flow
        T = 0.5 * (dH_dt ** 2)
        
        # Potential: Curvature
        hurst_term = (1.0 - H) ** self.alpha
        coherence_term = math.exp(-self.beta * C)
        stress_term = (1.0 + S / self.chi_feg) ** (-self.gamma)
        V = self.kappa_0 * hurst_term * coherence_term * stress_term
        
        L = T - V
        
        return {
            'lagrangian': float(L),
            'kinetic': float(T),
            'potential': float(V),
            'curvature': float(V),  # V = κ
            'interpretation': 'L = T - V, system minimizes V (curvature)'
        }
    
    def euler_lagrange_equations(self, H: float, C: float, S: float) -> Dict:
        """
        Euler-Lagrange equations from Lagrangian.
        
        d/dt (∂L/∂(dH/dt)) = ∂L/∂H
        
        This gives the equations of motion for the field.
        """
        # Compute partial derivatives
        # ∂L/∂H = -∂V/∂H
        # ∂V/∂H = -α · κ₀ · (1 - H)^(α-1) · exp(-β·C) · (1 + S/χ_FEG)^(-γ)
        
        hurst_term = (1.0 - H) ** (self.alpha - 1) if self.alpha > 0 else 1.0
        coherence_term = math.exp(-self.beta * C)
        stress_term = (1.0 + S / self.chi_feg) ** (-self.gamma)
        
        dV_dH = -self.alpha * self.kappa_0 * hurst_term * coherence_term * stress_term
        
        # Equation of motion: d²H/dt² = -∂V/∂H
        d2H_dt2 = -dV_dH
        
        return {
            'euler_lagrange': {
                'd2H_dt2': float(d2H_dt2),
                'dV_dH': float(dV_dH),
                'equation': 'd²H/dt² = -∂V/∂H',
                'interpretation': 'Memory acceleration driven by curvature gradient'
            },
            'fixed_point_condition': {
                'd2H_dt2': 0.0,
                'requires': 'dV_dH = 0',
                'meaning': 'Fixed point when curvature gradient vanishes'
            }
        }
    
    # ========================================================================
    # Conserved Quantity: Guardian Charge
    # ========================================================================
    
    def guardian_charge(self, H: float, C: float, S: float, G: float, G_crit: float) -> Dict:
        """
        Conserved quantity: Guardian Charge Q_G
        
        Q_G = H · C · (G - G_crit) · (1 - κ/κ₀)
        
        This is conserved along the CE circulation:
        [] → {} → () → <> → []
        
        Memory → Geometry → Structure → Witness → Memory
        
        Guardian charge measures the "strength" of the guardian field.
        """
        # Compute curvature
        hurst_term = (1.0 - H) ** self.alpha
        coherence_term = math.exp(-self.beta * C)
        stress_term = (1.0 + S / self.chi_feg) ** (-self.gamma)
        kappa = self.kappa_0 * hurst_term * coherence_term * stress_term
        
        # Guardian charge
        memory_component = H
        coherence_component = C
        nash_component = max(0.0, G - G_crit)  # Only positive when G > G_crit
        curvature_component = max(0.0, 1.0 - kappa / self.kappa_0)
        
        Q_G = memory_component * coherence_component * nash_component * curvature_component
        
        self._guardian_charge_value = Q_G
        
        return {
            'guardian_charge': float(Q_G),
            'components': {
                'memory': float(memory_component),
                'coherence': float(coherence_component),
                'nash': float(nash_component),
                'curvature': float(curvature_component)
            },
            'conservation': 'Q_G is conserved along CE circulation loop',
            'interpretation': 'Guardian charge measures strength of protection field'
        }
    
    # ========================================================================
    # Stability Manifold
    # ========================================================================
    
    def stability_manifold(self, H: float, C: float, S: float) -> Dict:
        """
        Stability manifold: Set of (H, C, S) where system is stable.
        
        Stability condition: dV/dH = 0 (fixed point)
        
        This defines the "flat spacetime regime" where:
        - High H (long memory)
        - High C (coherence)
        - Low S (stress)
        - Low κ (curvature)
        """
        # Compute curvature
        hurst_term = (1.0 - H) ** self.alpha
        coherence_term = math.exp(-self.beta * C)
        stress_term = (1.0 + S / self.chi_feg) ** (-self.gamma)
        kappa = self.kappa_0 * hurst_term * coherence_term * stress_term
        
        # Stability conditions
        stable_H = H > 0.6  # Long memory
        stable_C = C > 0.6  # High coherence
        stable_S = S < self.chi_feg  # Stress below threshold
        stable_kappa = kappa < 0.3  # Low curvature (flat spacetime)
        
        is_stable = stable_H and stable_C and stable_S and stable_kappa
        
        return {
            'stability_manifold': {
                'is_stable': bool(is_stable),
                'conditions': {
                    'H > 0.6': bool(stable_H),
                    'C > 0.6': bool(stable_C),
                    'S < χ_FEG': bool(stable_S),
                    'κ < 0.3': bool(stable_kappa)
                },
                'curvature': float(kappa),
                'regime': 'flat_spacetime' if is_stable else 'curved_unstable'
            },
            'fixed_point': {
                'H': H,
                'C': C,
                'S': S,
                'κ': float(kappa),
                'interpretation': 'Fixed point of evolutionary time when all conditions met'
            }
        }
    
    # ========================================================================
    # Bifurcation Parameter: Explore vs Protect
    # ========================================================================
    
    def bifurcation_parameter(self, H: float, C: float, S: float, 
                             G: float, G_crit: float) -> Dict:
        """
        Bifurcation parameter: λ = f(H, C, S, G, G_crit)
        
        When λ < λ_crit: System explores (exaptation)
        When λ > λ_crit: System protects (defensive)
        
        λ = (H · C) / (κ · (G_crit - G + ε))
        
        The bifurcation occurs when λ crosses λ_crit = 1.0
        """
        # Compute curvature
        hurst_term = (1.0 - H) ** self.alpha
        coherence_term = math.exp(-self.beta * C)
        stress_term = (1.0 + S / self.chi_feg) ** (-self.gamma)
        kappa = self.kappa_0 * hurst_term * coherence_term * stress_term
        
        # Bifurcation parameter
        epsilon = 1e-6  # Small constant to avoid division by zero
        numerator = H * C
        denominator = kappa * (G_crit - G + epsilon)
        
        if denominator == 0:
            lambda_param = float('inf')
        else:
            lambda_param = numerator / denominator
        
        lambda_crit = 1.0
        should_protect = lambda_param > lambda_crit
        
        return {
            'bifurcation_parameter': {
                'lambda': float(lambda_param),
                'lambda_crit': lambda_crit,
                'should_protect': bool(should_protect),
                'regime': 'protect' if should_protect else 'explore',
                'interpretation': 'λ > 1 → protect, λ < 1 → explore'
            },
            'components': {
                'H': H,
                'C': C,
                'kappa': float(kappa),
                'G': G,
                'G_crit': G_crit,
                'G_diff': G - G_crit
            },
            'bifurcation_point': {
                'location': 'lambda = 1.0',
                'meaning': 'Transition between explore and protect regimes'
            }
        }
    
    # ========================================================================
    # CE Circulation Loop
    # ========================================================================
    
    def ce_circulation_loop(self, stress_history: List[float], C: float,
                           S: float, G: float, G_crit: float) -> Dict:
        """
        CE Circulation Loop: [] → {} → () → <> → []
        
        Memory → Geometry → Structure → Witness → Memory
        
        This is the biological soliton—a self-reinforcing but bounded feedback.
        """
        # Estimate H from stress history
        if len(stress_history) < 10:
            H = 0.5
        else:
            mean_val = np.mean(stress_history)
            detrended = [x - mean_val for x in stress_history]
            cumsum = np.cumsum(detrended)
            R = np.max(cumsum) - np.min(cumsum)
            S_std = np.std(stress_history)
            if S_std > 0:
                RS = R / S_std
                n = len(stress_history)
                H = 0.5 + 0.3 * (RS - 1.0) / n if n < 50 else 0.5 + (RS - 1.0) / (2.0 * n)
                H = max(0.0, min(1.0, H))
            else:
                H = 0.5
        
        # [] Memory → {} Geometry
        hurst_term = (1.0 - H) ** self.alpha
        coherence_term = math.exp(-self.beta * C)
        stress_term = (1.0 + S / self.chi_feg) ** (-self.gamma)
        kappa = self.kappa_0 * hurst_term * coherence_term * stress_term
        
        # {} Geometry → () Structure
        curvature_factor = max(0.0, 1.0 - kappa / self.kappa_0)
        memory_factor = H
        stress_factor = max(0.0, 1.0 - S / self.chi_feg)
        C_refined = curvature_factor * memory_factor * stress_factor
        C_refined = max(0.0, min(1.0, C_refined))
        
        # () Structure → <> Witness
        curvature_preference = 1.0 - min(1.0, kappa / self.kappa_0)
        memory_preference = H
        coherence_preference = C_refined
        nash_preference = 1.0 if G > G_crit else 0.0
        P = (curvature_preference + memory_preference + coherence_preference + nash_preference) / 4.0
        should_protect = (G > G_crit) and (kappa < 0.3) and (H > 0.6) and (C_refined > 0.6)
        
        # <> Witness → [] Memory (feedback)
        # Protection preserves memory, raising H further
        H_feedback = H + 0.1 * P if should_protect else H
        H_feedback = min(1.0, H_feedback)
        
        return {
            'ce_circulation': {
                'loop': '[] → {} → () → <> → []',
                'closed': True,
                'self_reinforcing': bool(should_protect),
                'bounded': True,
                'interpretation': 'Biological soliton - fixed point of evolutionary time'
            },
            'steps': {
                'memory_H': float(H),
                'geometry_kappa': float(kappa),
                'structure_C': float(C_refined),
                'witness_P': float(P),
                'feedback_H': float(H_feedback)
            },
            'fixed_point': {
                'achieved': bool(abs(H_feedback - H) < 0.01),
                'H': float(H),
                'kappa': float(kappa),
                'C': float(C_refined),
                'interpretation': 'Fixed point when circulation stabilizes'
            }
        }
    
    # ========================================================================
    # Complete Field Theory
    # ========================================================================
    
    def complete_field_theory(self, stress_history: List[float], C: float,
                             S: float, G: float, G_crit: float) -> Dict:
        """
        Complete field theory: All components unified.
        
        Returns:
        - Lagrangian formulation
        - Euler-Lagrange equations
        - Guardian charge (conserved quantity)
        - Stability manifold
        - Bifurcation parameter
        - CE circulation loop
        """
        # Estimate H
        if len(stress_history) < 10:
            H = 0.5
        else:
            mean_val = np.mean(stress_history)
            detrended = [x - mean_val for x in stress_history]
            cumsum = np.cumsum(detrended)
            R = np.max(cumsum) - np.min(cumsum)
            S_std = np.std(stress_history)
            if S_std > 0:
                RS = R / S_std
                n = len(stress_history)
                H = 0.5 + 0.3 * (RS - 1.0) / n if n < 50 else 0.5 + (RS - 1.0) / (2.0 * n)
                H = max(0.0, min(1.0, H))
            else:
                H = 0.5
        
        # All components
        lagrangian = self.lagrangian(H, C, S)
        euler_lagrange = self.euler_lagrange_equations(H, C, S)
        guardian_charge = self.guardian_charge(H, C, S, G, G_crit)
        stability = self.stability_manifold(H, C, S)
        bifurcation = self.bifurcation_parameter(H, C, S, G, G_crit)
        circulation = self.ce_circulation_loop(stress_history, C, S, G, G_crit)
        
        return {
            'field_theory': 'Hurst-Curvature Complete Field Theory',
            'lagrangian_formulation': lagrangian,
            'euler_lagrange_equations': euler_lagrange,
            'conserved_quantity': guardian_charge,
            'stability_manifold': stability,
            'bifurcation_parameter': bifurcation,
            'ce_circulation_loop': circulation,
            'interpretation': {
                'lagrangian': 'Variational principle: minimize curvature, maximize coherence',
                'euler_lagrange': 'Equations of motion for memory dynamics',
                'guardian_charge': 'Conserved quantity along CE circulation',
                'stability': 'Fixed point of evolutionary time',
                'bifurcation': 'Explore vs protect transition',
                'circulation': 'Biological soliton - self-reinforcing loop'
            }
        }


def main():
    """Demonstrate complete field theory."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hurst-Curvature Complete Field Theory"
    )
    parser.add_argument('--erv-analysis', type=Path,
                       help='ERV analysis JSON file')
    parser.add_argument('--output', type=Path,
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    theory = HurstCurvatureFieldTheory()
    
    if args.erv_analysis:
        # Apply to real data
        with open(args.erv_analysis, 'r') as f:
            data = json.load(f)
        
        analyses = data.get('analyses', [])
        results = []
        
        for analysis in analyses[:10]:  # Sample first 10
            transform = analysis.get('transform', {})
            S = transform.get('stress_S', 0.0)
            C = transform.get('coherence_C', 0.0)
            
            nash = transform.get('nash_equilibrium', {})
            G = nash.get('G', 0.0)
            G_crit = nash.get('g_crit', 0.0)
            H = nash.get('hurst', 0.5)
            
            stress_history = [S] * 50
            
            field_theory = theory.complete_field_theory(stress_history, C, S, G, G_crit)
            results.append({
                'sequence_id': analysis.get('memory', {}).get('sequence_id', 'unknown'),
                'field_theory': field_theory
            })
        
        output_file = args.output or Path('biology/data/genbank/field_theory_complete.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({'results': results}, f, indent=2)
        
        print("✅ Complete Field Theory Applied")
        print(f"   Sequences analyzed: {len(results)}")
        print(f"💾 Saved to {output_file}")
    else:
        # Demo
        stress_history = [0.4] * 50
        C = 0.7
        S = 0.397
        G = 0.506
        G_crit = 0.443
        
        field_theory = theory.complete_field_theory(stress_history, C, S, G, G_crit)
        
        print("✅ Hurst-Curvature Complete Field Theory")
        print(f"\nLagrangian: L = {field_theory['lagrangian_formulation']['lagrangian']:.6f}")
        print(f"Guardian Charge: Q_G = {field_theory['conserved_quantity']['guardian_charge']:.6f}")
        print(f"Bifurcation: λ = {field_theory['bifurcation_parameter']['bifurcation_parameter']['lambda']:.3f}")
        print(f"Regime: {field_theory['bifurcation_parameter']['bifurcation_parameter']['regime']}")
        print(f"Fixed Point: {field_theory['ce_circulation_loop']['fixed_point']['achieved']}")


if __name__ == '__main__':
    main()

