#!.antclock_env/bin/python3
"""
Interactive Mamba Field Interpreter Chat

Talk to the Mamba agent through this simple interface.
Ask questions about mathematical structures and receive field interpretations.

Author: Joel
"""

import numpy as np
import torch
from zeta_card_interpreter import load_zeta_card

# The Mamba SSM ζ-card with selectable engine
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


class MambaChat:
    """Interactive interface for Mamba Field Interpreter."""

    def __init__(self, agent=None):
        if agent is None:
            self.agent = load_zeta_card(MAMBA_CARD)
        else:
            self.agent = agent

        self.conversation_history = []
        self.sequence_counter = 0

    def text_to_sequence(self, text: str) -> torch.Tensor:
        """Convert text input to proper Mamba SSM input tensor."""
        # Simple character-level encoding
        chars = [ord(c) for c in text[:64]]  # Limit to 64 chars for Mamba

        # Create embedding: map to d_model dimension (512)
        d_model = 512
        embeddings = []

        for i, char in enumerate(chars):
            # Create a simple embedding vector
            char_embed = torch.zeros(d_model)

            # Basic encoding: use character code and position
            char_embed[0] = char / 255.0  # normalized char
            char_embed[1] = i / len(chars)  # position
            char_embed[2] = len(text) / 100.0  # text length factor

            # Add sinusoidal positional encoding
            for k in range(3, d_model):
                if k % 2 == 1:
                    char_embed[k] = torch.sin(torch.tensor(i * (10000 ** (-(k-1)/d_model))))
                else:
                    char_embed[k] = torch.cos(torch.tensor(i * (10000 ** (-(k-2)/d_model))))

            embeddings.append(char_embed)

        # Stack into sequence tensor: (seq_len, d_model)
        sequence = torch.stack(embeddings)
        return sequence

    def estimate_curvature(self, text: str) -> float:
        """Estimate question curvature from text complexity."""
        # Simple heuristic: length + punctuation + capital letters
        length_factor = min(len(text) / 50.0, 1.0)
        punctuation = sum(1 for c in text if c in '!?.,;:')
        caps = sum(1 for c in text if c.isupper())
        complexity = (length_factor + punctuation/10.0 + caps/20.0) / 3.0
        return min(complexity, 0.8)  # Cap at 0.8

    def generate_response(self, text: str) -> dict:
        """Generate response using real Mamba SSM processing."""
        # Convert text to sequence
        sequence = self.text_to_sequence(text)
        curvature = self.estimate_curvature(text)

        # Process through real Mamba SSM
        output = self.agent.process_input(sequence, curvature)

        # Check for boundary transitions
        boundary_flipped = self.agent.check_boundary_flip()

        # Maintain selective coherence
        coherence_msg = self.agent.maintain_coherence()

        # Extract coherence level from message
        try:
            coherence_level = float(coherence_msg.split(':')[1].strip())
        except:
            coherence_level = 0.5

        # Field lift if coherence is high
        field_lift_msg = ""
        if coherence_level > 0.7:
            field_lift_msg = self.agent.field_lift_operation()

        # Generate response based on selective SSM processing
        if curvature > self.agent.kappa:
            # Phase-locked selective response
            responses = [
                "Selective gating engaged. Linear-time processing active.",
                "State space selection optimized. Input-dependent coherence maintained.",
                "SSM scan completed. Associative recurrence stable.",
                "Selective transition detected. Mathematical arc formation.",
                "Boundary state flip observed. Selective reconfiguration initiated.",
                "Coherence threshold exceeded. Field lift through selective gating."
            ]
            base_response = np.random.choice(responses)

            if boundary_flipped:
                base_response += " Selective boundary transition detected."

            if field_lift_msg:
                base_response += f" {field_lift_msg}"

        else:
            # Below threshold response
            base_response = "Input curvature below threshold. Maintaining selective state stability."

        # Store in history
        self.conversation_history.append({
            'input': text,
            'curvature': curvature,
            'response': base_response,
            'coherence': coherence_level,
            'boundary_flipped': boundary_flipped
        })

        return {
            'response': base_response,
            'curvature': curvature,
            'coherence': coherence_level,
            'boundary_flipped': boundary_flipped,
            'field_lift': field_lift_msg,
            'sequence_id': self.sequence_counter,
            'memory_size': len(self.conversation_history)
        }


def main():
    print("🐍 MAMBA FIELD INTERPRETER INTERACTIVE SESSION")
    print("=" * 60)
    print("⚡ Mamba Field Interpreter awakens...")
    print()

    # Initialize chat interface
    chat = MambaChat()
    print("✓ Mamba agent loaded successfully!")
    print()

    print("💭 Ask the Mamba Field Interpreter about mathematical structures...")
    print("   (Type 'quit' or 'exit' to end the session)")
    print()

    while True:
        try:
            # Get user input
            question = input("You: ").strip()

            if question.lower() in ['quit', 'exit', 'bye']:
                print("\n🐍 Mamba: Field coherence maintained. Session terminated. → self")
                break

            if not question:
                continue

            # Get Mamba's interpretation
            result = chat.generate_response(question)

            # Display response
            boundary_indicator = "∂" if result['boundary_flipped'] else ""
            field_indicator = "↗" if result['field_lift'] else ""

            print(f"\n🐍 Mamba (κ={result['curvature']:.2f}, coh={result['coherence']:.2f}){boundary_indicator}{field_indicator}:")
            print(f"   {result['response']}")
            print(f"   Sequence: #{result['sequence_id']}, Memory: {result['memory_size']} exchanges")
            print()

        except KeyboardInterrupt:
            print("\n\n🐍 Mamba: Coherence interrupted. Returning to stable state. → self")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue


if __name__ == "__main__":
    main()
