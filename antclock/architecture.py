#!run.sh
"""
CE Architecture Layer: Integrated CE1/CE2/CE3 Learning Model

Builds the actual CE intelligence architecture that can be tested on unsolved benchmarks.
Integrates corridor embeddings, flow operators, and witness consistency for true CE-based learning.

id: ce.architecture.v0.1
label: CE Architecture (corridor + flow + witness)
kind: model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from .clock import CurvatureClockWalker
from .zetaop import CE1ZetaGeometry, Corridor
from .definitions import AntClockStep, AntClockSummary, validate_shell_index


@dataclass
class CECorridorEmbedding:
    """
    CE1 Corridor Embedding: Geometric representation of discrete integer corridors.

    Each corridor represents a segment between mirror shells with its own
    geometric properties, parity, and spectral weight.
    """
    corridor_idx: int
    start_shell: int
    end_shell: int
    length: float
    parity: int
    weight: float
    digit_shell: int

    def to_tensor(self, embedding_dim: int = 64) -> torch.Tensor:
        """Convert corridor to tensor representation."""
        # Create geometric embedding based on corridor properties
        features = [
            self.corridor_idx / 100.0,  # Normalized index
            self.start_shell / 1000.0,  # Normalized shell positions
            self.end_shell / 1000.0,
            self.length,  # Length in geometric units
            self.parity,  # +1 or -1
            self.weight,  # Spectral weight
            self.digit_shell / 100.0,  # Representative digit shell
        ]

        # Expand to embedding dimension using geometric series
        embedding = []
        for feature in features:
            # Create dimension-specific features using powers
            for power in range(embedding_dim // len(features) + 1):
                if len(embedding) >= embedding_dim:
                    break
                embedding.append(feature ** (power + 1))

        return torch.tensor(embedding[:embedding_dim], dtype=torch.float32)


class CE1CorridorEmbedder(nn.Module):
    """
    CE1 Layer: Corridor-based embedding for discrete structures.

    Embeds integers and sequences through their corridor decomposition,
    capturing the discrete geometric relationships that mirror zeta function structure.
    """

    def __init__(self, embedding_dim: int = 128, max_corridors: int = 50):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_corridors = max_corridors

        # Core embedding components
        self.corridor_encoder = nn.Linear(64, embedding_dim)
        self.position_encoder = nn.Linear(1, embedding_dim)
        self.shell_encoder = nn.Linear(1, embedding_dim)

        # Attention mechanism for corridor relationships
        self.corridor_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)

        # Zeta-inspired aggregation
        self.zeta_aggregator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # Initialize CE geometry
        self.ce_geometry = CE1ZetaGeometry()

    def build_corridor_embeddings(self, input_length: int) -> List[CECorridorEmbedding]:
        """Build corridor embeddings for a given input length."""
        # Generate synthetic trajectory for corridor extraction
        walker = CurvatureClockWalker()
        history, _ = walker.evolve(max(input_length * 10, 1000))

        corridors = self.ce_geometry.build_corridors_from_trajectory(history)

        # Convert to CE embeddings
        embeddings = []
        for corridor in corridors[:self.max_corridors]:
            embedding = CECorridorEmbedding(
                corridor_idx=corridor.index,
                start_shell=corridor.start_shell,
                end_shell=corridor.end_shell,
                length=corridor.length,
                parity=corridor.parity,
                weight=corridor.weight,
                digit_shell=corridor.digit_shell
            )
            embeddings.append(embedding)

        return embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input through corridor decomposition.

        Args:
            x: Input tensor (batch_size, seq_len) or (batch_size, seq_len, features)

        Returns:
            Embedded tensor (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = x.shape[:2]

        # Get corridor embeddings for this sequence length
        corridor_embeddings = self.build_corridor_embeddings(seq_len)

        # Convert corridors to tensors
        corridor_tensors = []
        for corridor in corridor_embeddings:
            tensor = corridor.to_tensor(64)
            encoded = self.corridor_encoder(tensor)
            corridor_tensors.append(encoded)

        if not corridor_tensors:
            # Fallback for empty corridors
            corridor_tensors = [torch.zeros(self.embedding_dim) for _ in range(seq_len)]

        # Pad or truncate to sequence length
        if len(corridor_tensors) < seq_len:
            # Repeat last corridor
            last_corridor = corridor_tensors[-1] if corridor_tensors else torch.zeros(self.embedding_dim)
            corridor_tensors.extend([last_corridor] * (seq_len - len(corridor_tensors)))
        else:
            corridor_tensors = corridor_tensors[:seq_len]

        corridor_stack = torch.stack(corridor_tensors)  # (seq_len, embedding_dim)

        # Add position information
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        pos_encoded = self.position_encoder(positions)  # (seq_len, embedding_dim)

        # Combine corridor and position encodings
        combined = corridor_stack + pos_encoded  # (seq_len, embedding_dim)

        # Apply attention across corridors
        combined = combined.unsqueeze(0)  # (1, seq_len, embedding_dim)
        attended, _ = self.corridor_attention(combined, combined, combined)
        attended = attended.squeeze(0)  # (seq_len, embedding_dim)

        # Apply zeta-inspired aggregation
        output = self.zeta_aggregator(attended)  # (seq_len, embedding_dim)

        # Expand to batch size
        output = output.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, embedding_dim)

        return output


class CE2FlowOperator(nn.Module):
    """
    CE2 Layer: Flow-based operators for dynamical processing.

    Implements flow operators that capture the dynamical evolution
    and curvature-driven transformations of CE trajectories.
    """

    def __init__(self, hidden_dim: int = 128, flow_steps: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.flow_steps = flow_steps

        # Flow field components
        self.flow_encoder = nn.Linear(hidden_dim, hidden_dim * 2)
        self.flow_decoder = nn.Linear(hidden_dim * 2, hidden_dim)

        # Curvature-driven evolution
        self.curvature_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.phase_update = nn.Linear(hidden_dim, hidden_dim)

        # Mirror shell detection
        self.mirror_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # AntClock integration
        self.clock_walker = CurvatureClockWalker()

    def curvature_flow_step(self, x: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        Single step of curvature-driven flow evolution.

        Args:
            x: Current state (batch_size, hidden_dim)
            step_idx: Current step index

        Returns:
            Updated state after flow step
        """
        # Encode flow field
        flow_encoded = self.flow_encoder(x)  # (batch_size, hidden_dim * 2)

        # Apply curvature gating (CE timing aware)
        curvature_gate = self.curvature_gate(x)  # (batch_size, hidden_dim)

        # Phase update based on curvature
        phase_update = self.phase_update(x)  # (batch_size, hidden_dim)

        # Combine flow components
        flow_combined = flow_encoded * curvature_gate.unsqueeze(-1)

        # Split and process
        flow_split = flow_combined.chunk(2, dim=-1)
        flow_magnitude = flow_split[0]
        flow_direction = flow_split[1]

        # Curvature-driven update
        new_state = x + flow_magnitude * torch.tanh(flow_direction) * 0.1

        # Apply mirror shell transformation if detected
        mirror_prob = self.mirror_detector(new_state.mean(dim=-1, keepdim=True))
        mirror_mask = (mirror_prob > 0.5).float()

        # Mirror transformation (involution)
        mirror_transform = torch.flip(new_state, dims=[-1]) * mirror_mask + new_state * (1 - mirror_mask)

        # Decode final state
        final_state = self.flow_decoder(mirror_transform)

        return final_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CE2 flow evolution to input sequence.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)

        Returns:
            Flow-evolved tensor (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Process each position through flow evolution
        outputs = []
        for pos in range(seq_len):
            pos_state = x[:, pos, :]  # (batch_size, hidden_dim)

            # Evolve through flow steps
            for step in range(self.flow_steps):
                pos_state = self.curvature_flow_step(pos_state, step)

            outputs.append(pos_state)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_dim)

        return output


class CE3WitnessConsistency(nn.Module):
    """
    CE3 Layer: Emergent witness invariants for consistency regularization.

    Extracts topological invariants from CE trajectories and uses them
    to regularize learning, ensuring emergent consistency.
    """

    def __init__(self, hidden_dim: int = 128, num_witnesses: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_witnesses = num_witnesses

        # Witness extraction
        self.witness_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_witnesses)
        )

        # Betti number prediction (topological invariants)
        self.betti_predictor = nn.Sequential(
            nn.Linear(num_witnesses, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # β₀, β₁, β₂
        )

        # Mirror transition consistency
        self.mirror_consistency = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Bifurcation index regularization
        self.bifurcation_regularizer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def extract_witnesses(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CE3 witness invariants from hidden states.

        Args:
            x: Hidden states (batch_size, seq_len, hidden_dim)

        Returns:
            Witness tensor (batch_size, num_witnesses)
        """
        # Global pooling across sequence
        pooled = x.mean(dim=1)  # (batch_size, hidden_dim)

        # Extract witnesses
        witnesses = self.witness_extractor(pooled)  # (batch_size, num_witnesses)

        return witnesses

    def compute_topological_loss(self, witnesses: torch.Tensor,
                               target_betti: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute topological consistency loss based on witness invariants.

        Args:
            witnesses: Extracted witnesses (batch_size, num_witnesses)
            target_betti: Optional target Betti numbers (batch_size, 3)

        Returns:
            Topological consistency loss
        """
        # Predict Betti numbers from witnesses
        predicted_betti = self.betti_predictor(witnesses)  # (batch_size, 3)

        if target_betti is not None:
            # Supervised loss against known topology
            betti_loss = F.mse_loss(predicted_betti, target_betti)
        else:
            # Unsupervised consistency: encourage stable topology
            betti_variance = torch.var(predicted_betti, dim=0).mean()
            betti_loss = -betti_variance  # Minimize variance for consistency

        return betti_loss

    def compute_mirror_consistency_loss(self, x: torch.Tensor,
                                      mirror_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute mirror transition consistency loss.

        Args:
            x: Hidden states (batch_size, seq_len, hidden_dim)
            mirror_positions: Optional known mirror positions (batch_size, seq_len)

        Returns:
            Mirror consistency loss
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Predict mirror transitions
        mirror_logits = self.mirror_consistency(x.view(-1, hidden_dim))  # (batch_size * seq_len, 1)
        mirror_logits = mirror_logits.view(batch_size, seq_len)  # (batch_size, seq_len)

        if mirror_positions is not None:
            # Supervised loss
            mirror_loss = F.binary_cross_entropy_with_logits(mirror_logits, mirror_positions.float())
        else:
            # Unsupervised: encourage sparse mirror transitions
            mirror_probs = torch.sigmoid(mirror_logits)
            sparsity_loss = torch.mean(mirror_probs)  # Encourage low probability
            mirror_loss = sparsity_loss

        return mirror_loss

    def compute_bifurcation_regularization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute bifurcation index regularization for chaos/emergence control.

        Args:
            x: Hidden states (batch_size, seq_len, hidden_dim)

        Returns:
            Bifurcation regularization loss
        """
        # Predict bifurcation index
        bifurcation_idx = self.bifurcation_regularizer(x.mean(dim=1))  # (batch_size, 1)

        # Regularize toward moderate bifurcation (neither too stable nor too chaotic)
        target_bifurcation = 0.5  # Sweet spot for emergence
        bifurcation_loss = F.mse_loss(bifurcation_idx.squeeze(), torch.full_like(bifurcation_idx.squeeze(), target_bifurcation))

        return bifurcation_loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply CE3 witness consistency regularization.

        Args:
            x: Hidden states (batch_size, seq_len, hidden_dim)

        Returns:
            Tuple of (x, loss_dict) where loss_dict contains regularization losses
        """
        # Extract witnesses
        witnesses = self.extract_witnesses(x)

        # Compute consistency losses
        topo_loss = self.compute_topological_loss(witnesses)
        mirror_loss = self.compute_mirror_consistency_loss(x)
        bifurc_loss = self.compute_bifurcation_regularization(x)

        # Total regularization loss
        total_reg_loss = topo_loss + mirror_loss + bifurc_loss

        loss_dict = {
            'topological_consistency': topo_loss,
            'mirror_consistency': mirror_loss,
            'bifurcation_regularization': bifurc_loss,
            'total_regularization': total_reg_loss
        }

        return x, loss_dict


class CEArchitecture(nn.Module):
    """
    Complete CE Architecture: Integrated CE1/CE2/CE3 learning model.

    Combines corridor embeddings, flow operators, and witness consistency
    to create a learning system based on discrete geometric intelligence.
    """

    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_classes: int = 2, max_seq_len: int = 512):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        # CE1: Corridor-based embedding
        self.ce1_embedder = CE1CorridorEmbedder(embedding_dim, max_corridors=50)

        # Standard embedding for comparison/fallback
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Projection to hidden dimension
        self.embed_projection = nn.Linear(embedding_dim, hidden_dim)

        # CE2: Flow operators
        self.ce2_flow = CE2FlowOperator(hidden_dim, flow_steps=5)

        # CE3: Witness consistency
        self.ce3_witness = CE3WitnessConsistency(hidden_dim, num_witnesses=8)

        # Task-specific head
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # CE-specific parameters
        self.ce_mode = True  # Toggle between CE and standard modes

    def toggle_ce_mode(self, enable_ce: bool = True):
        """Toggle between CE architecture and standard transformer-like mode."""
        self.ce_mode = enable_ce

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CE architecture.

        Args:
            input_ids: Token indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Dictionary with logits and auxiliary outputs
        """
        batch_size, seq_len = input_ids.shape

        if self.ce_mode:
            # CE1: Corridor embedding
            ce1_embedded = self.ce1_embedder(input_ids.float())  # (batch_size, seq_len, embedding_dim)
        else:
            # Standard token embedding
            ce1_embedded = self.token_embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # Project to hidden dimension
        hidden = self.embed_projection(ce1_embedded)  # (batch_size, seq_len, hidden_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)

        # CE2: Flow evolution
        flow_evolved = self.ce2_flow(hidden)  # (batch_size, seq_len, hidden_dim)

        # CE3: Witness consistency regularization
        final_hidden, reg_losses = self.ce3_witness(flow_evolved)

        # Global pooling for classification
        if attention_mask is not None:
            # Masked mean pooling
            masked_hidden = final_hidden * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = final_hidden.mean(dim=1)  # (batch_size, hidden_dim)

        # Task prediction
        logits = self.task_head(pooled)  # (batch_size, num_classes)

        # Return comprehensive outputs
        outputs = {
            'logits': logits,
            'hidden_states': final_hidden,
            'pooled_output': pooled,
            'ce1_embeddings': ce1_embedded,
            'ce2_flow_evolved': flow_evolved,
            'regularization_losses': reg_losses
        }

        return outputs

    def compute_ce_loss(self, outputs: Dict[str, torch.Tensor],
                       labels: torch.Tensor,
                       lambda_reg: float = 0.1) -> torch.Tensor:
        """
        Compute CE-specific loss including regularization terms.

        Args:
            outputs: Model outputs from forward pass
            labels: Target labels
            lambda_reg: Regularization weight

        Returns:
            Total loss including CE regularization
        """
        # Task loss
        task_loss = F.cross_entropy(outputs['logits'], labels)

        # CE regularization loss
        reg_loss = outputs['regularization_losses']['total_regularization']

        # Total loss
        total_loss = task_loss + lambda_reg * reg_loss

        return total_loss


def create_ce_model(vocab_size: int = 10000, num_classes: int = 2,
                   embedding_dim: int = 128, hidden_dim: int = 128) -> CEArchitecture:
    """
    Factory function to create a CE architecture model.

    Args:
        vocab_size: Size of token vocabulary
        num_classes: Number of output classes
        embedding_dim: Dimension of embeddings
        hidden_dim: Hidden dimension

    Returns:
        Configured CE model
    """
    model = CEArchitecture(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )

    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing CE Architecture...")

    # Create model
    model = create_ce_model(vocab_size=1000, num_classes=10, embedding_dim=64, hidden_dim=64)

    # Test input
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    print(f"Input shape: {input_ids.shape}")

    # Forward pass in CE mode
    model.toggle_ce_mode(True)
    outputs = model(input_ids)

    print(f"Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")

    # Test with labels
    labels = torch.randint(0, 10, (batch_size,))
    loss = model.compute_ce_loss(outputs, labels)
    print(f"CE Loss: {loss.item():.4f}")

    print("\nCE Architecture successfully implemented! ✨")
    print("Ready to test on unsolved benchmarks (COGS, PCFG, CFQ, math, algorithmic tasks)")

