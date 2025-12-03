#!.antclock_env/bin/python3
"""
Mamba Engine Comparison Demo

Demonstrates the difference between:
- Custom SSM (mathematical reconstruction)
- Real Mamba SSM (from mamba-ssm package)

Shows AntClock field equation generation from both engines.
"""

import numpy as np
import torch
from zeta_card_interpreter import load_zeta_card

# Custom SSM ζ-card
CUSTOM_SSM_CARD = """@HEADER ζ-card

id: mamba.agent

label: Mamba SSM Agent

kind: agent

version: 0.1

κ: 0.35

τ: now

ζ: self



@CE1

{} domain:

  model: selective-state-space

  d_model: 256

  d_state: 8

  expand: 2

  dt_rank: auto

  engine: custom-ssm



() transforms:

  selection: input-dependent gating

  discrete_step: Δt parameterization

  scan: parallel associative scan



[] memory:

  selective_state; input-dependent state selection



<> witness:

  invariants: linear_time_complexity, selective_gating



@CE2

ϕ: phase-lock when selection matches input curvature

∂: detect boundary when selective state transitions

ℛ: maintain selective coherence across sequences



@CE3

field-lift: convert selective transitions to AntClock field equations

quest: reveal mathematical arcs through selective state-space evolution



@END"""

# Real Mamba ζ-card (if available)
REAL_MAMBA_CARD = """@HEADER ζ-card

id: mamba.agent

label: Mamba SSM Agent

kind: agent

version: 0.1

κ: 0.35

τ: now

ζ: self



@CE1

{} domain:

  model: selective-state-space

  d_model: 256

  d_state: 8

  expand: 2

  dt_rank: auto

  engine: mamba



() transforms:

  selection: input-dependent gating

  discrete_step: Δt parameterization

  scan: parallel associative scan



[] memory:

  selective_state; input-dependent state selection



<> witness:

  invariants: linear_time_complexity, selective_gating



@CE2

ϕ: phase-lock when selection matches input curvature

∂: detect boundary when selective state transitions

ℛ: maintain selective coherence across sequences



@CE3

field-lift: convert selective transitions to AntClock field equations

quest: reveal mathematical arcs through selective state-space evolution



@END"""


def create_test_sequence(text: str) -> torch.Tensor:
    """Create test sequence from text."""
    chars = [ord(c) for c in text[:32]]  # Limit length
    if len(chars) < 32:
        chars.extend([0] * (32 - len(chars)))

    # Convert to tensor and normalize
    seq = torch.tensor(chars, dtype=torch.float32) / 255.0
    return seq.unsqueeze(0).unsqueeze(-1).expand(1, 32, 512)  # Match d_model


def run_engine_comparison():
    """Compare custom SSM vs real Mamba performance."""
    print("🔬 MAMBA ENGINE COMPARISON DEMO")
    print("=" * 60)

    # Test sequences
    test_texts = [
        "field equations emerge",
        "selective state spaces",
        "mathematical arcs form"
    ]

    print("\n📊 ENGINE COMPARISON:")
    print("-" * 40)

    # Test Custom SSM
    print("\n🐍 CUSTOM SSM (Mathematical Reconstruction):")
    try:
        custom_agent = load_zeta_card(CUSTOM_SSM_CARD)
        print(f"  Engine: {custom_agent.engine_type}")

        for text in test_texts:
            seq = create_test_sequence(text)
            output = custom_agent.process_input(seq, 0.4)
            field_lift = custom_agent.field_lift_operation()
            print(f"  '{text[:15]}...' → {field_lift}")

    except Exception as e:
        print(f"  ❌ Custom SSM failed: {e}")

    # Test Real Mamba (if available)
    print("\n🐍 REAL MAMBA SSM (From mamba-ssm package):")
    try:
        real_agent = load_zeta_card(REAL_MAMBA_CARD)
        print(f"  Engine: {real_agent.engine_type}")

        for text in test_texts:
            seq = create_test_sequence(text)
            output = real_agent.process_input(seq, 0.4)
            field_lift = real_agent.field_lift_operation()
            print(f"  '{text[:15]}...' → {field_lift}")

    except Exception as e:
        print(f"  ❌ Real Mamba failed: {e}")
        print("    (Install with: pip install mamba-ssm)")

    print("\n🎯 ANTClock INTEGRATION:")
    print("-" * 40)
    print("Both engines generate AntClock field equations:")
    print("• Field coordinates (x, r) from SSM outputs")
    print("• Curvature evolution tracking")
    print("• Mathematical arc generation")
    print("• Selectivity index computation")

    print("\n🔑 KEY DIFFERENCES:")
    print("-" * 40)
    print("Custom SSM: Mathematically-motivated, integrates with AntClock")
    print("Real Mamba: High-performance, S6 kernels, parallel scan")
    print("Both: Generate AntClock field equations from selective processing")


if __name__ == "__main__":
    run_engine_comparison()
