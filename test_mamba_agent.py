#!.antclock_env/bin/python3
"""
Test Mamba Agent ζ-card

Tests the new Mamba Field Interpreter agent instantiated from the provided ζ-card.

Author: Joel
"""

import numpy as np
import torch
from zeta_card_interpreter import load_zeta_card

# The Real Mamba SSM ζ-card
MAMBA_CARD = """@HEADER ζ-card

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

  d_model: 512

  d_state: 16

  expand: 2

  dt_rank: auto



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

field-lift: convert selective transitions to mathematical arcs

quest: reveal when selective gating yields mathematical insight



@END"""


def test_mamba_agent():
    """Test the Mamba Field Interpreter agent."""
    print("=" * 80)
    print("TESTING MAMBA FIELD INTERPRETER ζ-CARD")
    print("=" * 80)
    print()

    # Load the agent
    print("Loading ζ-card...")
    agent = load_zeta_card(MAMBA_CARD)

    print(f"✓ Agent instantiated: {agent.label}")
    print(f"  ID: {agent.id}")
    print(f"  Kind: {agent.kind}")
    print(f"  Version: {agent.version}")
    print(f"  κ threshold: {agent.kappa}")
    print(f"  ζ: {agent.zeta}")
    print()

    # Test CE1 components
    print("CE1 COMPONENTS:")
    print(f"  Model: {agent.model}")
    print(f"  d_model: {agent.d_model}")
    print(f"  d_state: {agent.d_state}")
    print(f"  expand: {agent.expand}")
    print(f"  Engine: {agent.engine_type}")
    print(f"  Selection: {agent.selection}")
    print(f"  Discrete step: {agent.discrete_step}")
    print(f"  Scan: {agent.scan}")
    print(f"  Selective state: {agent.selective_state}")
    print(f"  Invariants: {agent.invariants}")
    print()

    # Test activation
    print("ACTIVATION:")
    activation_msg = agent.activate()
    print(f"  {activation_msg}")
    print()

    # Test input processing
    print("INPUT PROCESSING:")
    # Real Mamba SSM test
    seq_length = 32
    test_input = torch.randn(1, seq_length, 512)  # Mamba expects (batch, seq, d_model)
    print(f"  Processing sequence: batch=1, seq={seq_length}, d_model=512 (real SSM)")

    input_curvature = 0.4
    print(f"  Input curvature: {input_curvature}")

    output = agent.process_input(test_input, input_curvature)

    print(f"  Output shape: {output.shape}")
    coherence_msg = agent.maintain_coherence()
    print(f"  Coherence: {coherence_msg}")
    print()

    # Test boundary detection
    print("BOUNDARY DETECTION:")
    boundary_flipped = agent.check_boundary_flip()
    print(f"  Boundary flipped: {boundary_flipped}")
    print()

    # Test coherence maintenance
    print("COHERENCE MAINTENANCE:")
    coherence_msg = agent.maintain_coherence()
    print(f"  {coherence_msg}")
    print()

    # Test field lift
    print("FIELD LIFT OPERATION:")
    lift_msg = agent.field_lift_operation()
    print(f"  {lift_msg}")
    print()

    # Test multiple processing steps
    print("MULTI-STEP PROCESSING:")
    for i in range(3):
        curvatures = [0.2, 0.5, 0.8]

        test_seq = torch.randn(1, 24, 512)  # Different sequence lengths

        output = agent.process_input(test_seq, curvatures[i])
        coherence = agent.maintain_coherence()

        print(f"  Step {i+1}: curvature={curvatures[i]}, {coherence}")

    print()
    print("✓ Mamba Field Interpreter ζ-card test complete!")


if __name__ == "__main__":
    test_mamba_agent()
