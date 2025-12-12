#!/usr/bin/env python3
"""
Nash Equilibrium Analysis for ERV Exaptation Decisions.

Implements game-theoretic Volte decision framework from working.md Section 5.4.

Models the two-player game:
- Player 1 (Composer/Exploiter): composition intensity s₁
- Player 2 (Guardian/Protector): protection intensity s₂

The Nash equilibrium determines when ERV exaptation should occur.
"""

from typing import Dict, Tuple, Optional
import math


class ERVNashEquilibrium:
    """
    Nash equilibrium analysis for ERV exaptation decisions.
    
    Implements guardian coupling field equation:
    β(G, H) = β_res · 𝟙_{G < G_crit(H)}
    
    where G_crit(H) = κ · (1 + H · α_H)
    """
    
    def __init__(self, 
                 kappa: float = 0.35,  # Base guardian threshold
                 alpha_h: float = 0.5,  # Hurst coupling constant
                 beta_res: float = 1.0):  # Resonance coupling
        """
        Initialize Nash equilibrium analyzer.
        
        Args:
            kappa: Base guardian threshold (default 0.35 from learnability boundary)
            alpha_h: Hurst coupling constant
            beta_res: Resonance coupling strength
        """
        self.kappa = kappa
        self.alpha_h = alpha_h
        self.beta_res = beta_res
    
    def g_crit(self, hurst: float) -> float:
        """
        Calculate Hurst-modulated critical threshold.
        
        G_crit(H) = κ · (1 + H · α_H)
        
        Args:
            hurst: Hurst exponent H ∈ [0,1] measuring long-range dependence
            
        Returns:
            Critical threshold G_crit
        """
        return self.kappa * (1.0 + hurst * self.alpha_h)
    
    def guardian_coupling(self, G: float, hurst: float) -> float:
        """
        Guardian coupling field equation.
        
        β(G, H) = β_res · 𝟙_{G < G_crit(H)}
        
        Args:
            G: Composition intensity (gain)
            hurst: Hurst exponent H ∈ [0,1]
            
        Returns:
            Guardian coupling β (0 or β_res)
        """
        g_crit = self.g_crit(hurst)
        if G < g_crit:
            return self.beta_res
        else:
            return 0.0
    
    def compute_hurst_exponent(self, stress_history: list) -> float:
        """
        Estimate Hurst exponent from stress time series.
        
        Uses rescaled range (R/S) analysis for long-range dependence.
        Higher H indicates longer memory (more conservative coupling needed).
        
        Args:
            stress_history: List of stress values over time
            
        Returns:
            Hurst exponent H ∈ [0,1]
        """
        if len(stress_history) < 10:
            return 0.5  # Default neutral value
        
        # Simplified Hurst estimation using variance method
        # For production, use proper R/S analysis
        n = len(stress_history)
        mean_stress = sum(stress_history) / n
        
        # Calculate variance
        variance = sum((s - mean_stress) ** 2 for s in stress_history) / n
        
        # Estimate Hurst from variance scaling
        # Higher variance → lower Hurst (less memory)
        # Lower variance → higher Hurst (more memory)
        if variance < 0.01:
            return 0.8  # High memory (low variance)
        elif variance < 0.05:
            return 0.6  # Moderate memory
        else:
            return 0.4  # Low memory (high variance)
    
    def nash_equilibrium_strategy(self, 
                                  G: float, 
                                  hurst: float,
                                  stress: float) -> Dict[str, any]:
        """
        Compute Nash equilibrium strategy for ERV exaptation decision.
        
        Returns optimal strategies for both players:
        - s₁* (Composer): compose (1) or don't compose (0)
        - s₂* (Guardian): protect (1) or don't protect (0)
        
        Args:
            G: Composition intensity/gain
            hurst: Hurst exponent
            stress: Current stress level S(x,u)
            
        Returns:
            Dictionary with Nash equilibrium analysis
        """
        g_crit = self.g_crit(hurst)
        beta = self.guardian_coupling(G, hurst)
        
        # Player 1 (Composer) strategy
        # Compose if G < G_crit (safe composition)
        s1_star = 1.0 if G < g_crit else 0.0
        
        # Player 2 (Guardian) strategy
        # Protect if G >= G_crit (composition unsafe)
        s2_star = 1.0 if G >= g_crit else 0.0
        
        # Payoffs
        # π₁ = G·s₁ - C(s₁, s₂)
        # π₂ = (1-G)·s₂ - C(s₁, s₂)
        # C = error cost when s₁ > threshold
        
        error_cost = 0.0
        if s1_star > 0 and G >= g_crit:
            error_cost = stress * 0.5  # Cost proportional to stress
        
        payoff1 = G * s1_star - error_cost
        payoff2 = (1.0 - G) * s2_star - error_cost
        
        return {
            'G': G,
            'hurst': hurst,
            'g_crit': g_crit,
            'beta': beta,
            's1_star': s1_star,  # Composer strategy
            's2_star': s2_star,  # Guardian strategy
            'payoff1': payoff1,
            'payoff2': payoff2,
            'error_cost': error_cost,
            'should_exapt': s1_star > 0 and beta > 0,  # Exaptation recommended
            'should_protect': s2_star > 0  # Protection recommended
        }
    
    def analyze_erv_exaptation(self,
                               composition_gain: float,
                               stress_history: list,
                               current_stress: float) -> Dict[str, any]:
        """
        Complete Nash equilibrium analysis for ERV exaptation decision.
        
        Args:
            composition_gain: G - composition intensity
            stress_history: Historical stress values for Hurst estimation
            current_stress: Current stress S(x,u)
            
        Returns:
            Complete Nash equilibrium analysis with recommendation
        """
        hurst = self.compute_hurst_exponent(stress_history)
        nash = self.nash_equilibrium_strategy(
            composition_gain, 
            hurst, 
            current_stress
        )
        
        # Add interpretation
        nash['interpretation'] = self._interpret_nash(nash)
        
        return nash
    
    def _interpret_nash(self, nash: Dict) -> str:
        """Provide human-readable interpretation of Nash equilibrium."""
        if nash['should_exapt']:
            return (f"Exaptation recommended: G={nash['G']:.3f} < G_crit={nash['g_crit']:.3f}. "
                   f"Composition is safe (β={nash['beta']:.3f}).")
        elif nash['should_protect']:
            return (f"Protection recommended: G={nash['G']:.3f} >= G_crit={nash['g_crit']:.3f}. "
                   f"Composition would be unsafe (β=0).")
        else:
            return "No action recommended: system in equilibrium."


def estimate_composition_gain(stress: float, coherence: float, threshold: float = 0.638) -> float:
    """
    Estimate composition gain G from system state.
    
    G represents the benefit of compositional operations.
    Higher stress and lower coherence → lower G (less benefit).
    
    Args:
        stress: Current stress S(x,u)
        coherence: Current coherence C(x)
        threshold: Volte activation threshold
        
    Returns:
        Estimated composition gain G ∈ [0,1]
    """
    # G decreases as stress approaches threshold
    # G increases with coherence
    stress_factor = 1.0 - (stress / threshold) if stress < threshold else 0.0
    coherence_factor = coherence
    
    # Weighted combination
    G = 0.6 * stress_factor + 0.4 * coherence_factor
    
    return max(0.0, min(1.0, G))  # Clamp to [0,1]





