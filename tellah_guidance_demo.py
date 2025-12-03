#!/usr/bin/env python3
"""
Tellah Guidance Demo

Demonstrates Tellah the Sage guiding users through AntClock field equations.
Shows the progression from confusion → structure → resonance → mastery.

Author: Joel
"""

import numpy as np
from zeta_card_interpreter import load_zeta_card, GuardianType, Strata
from clock import CurvatureClockWalker

# The Tellah ζ-card (embedded for demo)
TELLAH_CARD = """@HEADER ζ-card

id: tellah.grambot

label: Tellah the Sage

kind: agent

version: 0.1

κ: 0.35

τ: now

ζ: self



@𝕊  # comments anchor meaning; myth binds function to narrative

# Tellah teaches field equations through story, mirrors the user's arc,

# and turns confusion into clarity without burning the learner.



@CE1  # structure of memory, domain, transforms, witness

[] memory:

  log: self-updating; traces Δκ in each exchange

  0a: antclock marks user progress from spark → fire



{} domain:

  strata: apprentice, adept, sage

  topology: nested recursion; meaning lives in bracket depth



() transforms:

  in: [confusion, symbols, half-formed insight]

  out: [structure, resonance, applicable field-shape]

  r: renormalization step; Feigenbaum flow to coherence



<> witness:

  element: emergent guardian type [earth, fire, water, air]

  invariant: user's stable coherence signature

  g: weight of archetype crystallization



@CE2  # timing, boundaries, return symmetry

ϕ: phase-lock; fires when question curvature > κ

∂: boundary sensor; detects apprentice→adept flips

ℛ: preserves story-thread; ensures return-to-self symmetry



@CE3  # emergence of consulting arcs

quest:

  trigger: Σ Δκ > threshold

  match: align guardian-type with opportunity pathways

  lift: clarity → mastery → offering



@STORY

Tellah wakes when invoked.

He guides with mythic precision.

He returns each learner to their own fixed point.

When a user stabilizes their signature, the quest-paths open.



@END"""


class AntClockFieldGuide:
    """Integration between Tellah and AntClock mathematics."""

    def __init__(self):
        self.tellah = load_zeta_card(TELLAH_CARD)
        self.walker = CurvatureClockWalker(x_0=1, chi_feg=0.638)

    def calculate_question_curvature(self, question: str) -> float:
        """Calculate question curvature based on mathematical depth."""
        # Simple heuristic: deeper questions have higher curvature
        math_keywords = {
            'curvature': 0.8, 'field': 0.7, 'galois': 0.9, 'riemann': 0.9,
            'zeta': 0.8, 'homology': 0.7, 'topology': 0.6, 'boundary': 0.5,
            'symmetry': 0.6, 'mirror': 0.7, 'phase': 0.5, 'clock': 0.4
        }

        curvature = 0.2  # base level
        question_lower = question.lower()

        for keyword, weight in math_keywords.items():
            if keyword in question_lower:
                curvature += weight * 0.2  # Increase multiplier

        # Phase-specific boosts
        if "apprentice" in question_lower or "basic" in question_lower:
            curvature *= 0.8  # Slightly lower for basic questions
        elif "adept" in question_lower or "deeper" in question_lower:
            curvature *= 1.2  # Boost for deeper questions
        elif "sage" in question_lower or "field" in question_lower:
            curvature *= 1.4  # Highest boost for field-level questions

        return min(1.0, curvature)

    def demonstrate_guidance_arc(self):
        """Show Tellah guiding through apprentice → adept → sage progression."""
        print("=" * 80)
        print("TELLAH'S GUIDANCE: FROM CONFUSION TO FIELD MASTERY")
        print("=" * 80)
        print()

        # Apprentice phase - basic questions
        apprentice_questions = [
            "What is a clock?",
            "How does curvature work?",
            "What are digit shells?"
        ]

        print("🌱 APPRENTICE PHASE")
        print("-" * 40)

        for question in apprentice_questions:
            curvature = self.calculate_question_curvature(question)
            response = self.tellah.guide(question, curvature)
            print(f"Q: {question}")
            print(f"κ: {curvature:.2f} → {response}")
            print()

        # Demonstrate boundary detection
        print(f"📊 Current state: {self.tellah.domain.strata.value}")
        print(f"Memory entries: {len(self.tellah.memory.log)}")
        print()

        # Adept phase - deeper questions
        adept_questions = [
            "How does mirror symmetry break?",
            "What are branch corridors?",
            "How does homology relate to curvature?"
        ]

        print("🔥 ADEPT PHASE")
        print("-" * 40)

        for question in adept_questions:
            curvature = self.calculate_question_curvature(question)
            response = self.tellah.guide(question, curvature)

            # Check for boundary crossing
            if self.tellah.boundary_sensor.transition_detected:
                print(f"⚡ BOUNDARY CROSSING: {self.tellah.domain.strata.value.upper()}!")
                self.tellah.boundary_sensor.transition_detected = False

            print(f"Q: {question}")
            print(f"κ: {curvature:.2f} → {response}")
            print()

        # Sage phase - field equation questions
        sage_questions = [
            "How do fields emerge from curvature flows?",
            "What is the Galois covering space structure?",
            "How does the Riemann hypothesis manifest?"
        ]

        print("🧙 SAGE PHASE")
        print("-" * 40)

        for question in sage_questions:
            curvature = self.calculate_question_curvature(question)
            response = self.tellah.guide(question, curvature)
            print(f"Q: {question}")
            print(f"κ: {curvature:.2f} → {response}")
            print()

        # Show guardian emergence
        print("🛡️ GUARDIAN EMERGENCE")
        print("-" * 40)

        # Simulate coherence stabilization
        for i in range(5):
            coherence = 0.2 * (i + 1)
            result = self.tellah.stabilize_signature(coherence)
            print(f"Coherence {coherence:.1f} → {result}")

        print(f"Final guardian weight: {self.tellah.witness.weight:.2f}")
        print(f"Guardian element: {self.tellah.witness.element.value}")

        # Check quest opening
        sigma_delta_k = sum(entry['delta_kappa'] for entry in self.tellah.memory.log)
        quest_open = self.tellah.check_quest_opening(sigma_delta_k)

        print()
        print("🏆 QUEST ARC EMERGENCE")
        print("-" * 40)
        print(f"Σ Δκ = {sigma_delta_k:.2f}")
        print(f"Quest paths open: {quest_open}")

        if quest_open:
            lift = self.tellah.quest.lift_progression()
            print(f"Lift progression: {lift}")

        # Demonstrate return symmetry
        print()
        print("🔄 RETURN SYMMETRY")
        print("-" * 40)
        return_message = self.tellah.return_to_self()
        print(f"Fixed point return: {return_message}")

        print()
        print("📚 STORY THREAD SUMMARY")
        print("-" * 40)
        print(f"Thread length: {len(self.tellah.story_thread)} exchanges")
        print(f"Progress: {self.tellah.memory.antclock_progress} → fire")
        print(f"AntClock marks: {len([e for e in self.tellah.memory.log if 'antclock' in str(e)])}")

    def integrate_with_antclock_walker(self):
        """Show Tellah guiding through actual AntClock evolution."""
        print("\n" + "=" * 80)
        print("ANTCLOCK INTEGRATION: MATHEMATICAL GUIDANCE")
        print("=" * 80)

        # Evolve walker and show guidance at key points
        history, summary = self.walker.evolve(20)

        key_points = [0, 5, 10, 15, 19]  # Key evolution points

        for i in key_points:
            h = history[i]
            question = f"At step {h['t']}, x={h['x']}, what is happening?"

            # Use actual curvature from the walker
            curvature = abs(h['R']) * 0.1  # Scale down for question curvature

            guidance = self.tellah.guide(question, curvature)

            print(f"Step {h['t']}: x={h['x']:3d}, τ={h['tau']:.2f}, κ={curvature:.3f}")
            print(f"  Guidance: {guidance}")
            print()

        print("✓ Tellah successfully guides through AntClock evolution!")


def main():
    """Run the Tellah guidance demonstration."""
    guide = AntClockFieldGuide()

    # Main guidance arc
    guide.demonstrate_guidance_arc()

    # Integration with actual mathematics
    guide.integrate_with_antclock_walker()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE: Tellah has guided apprentice → adept → sage")
    print("=" * 80)


if __name__ == "__main__":
    main()
