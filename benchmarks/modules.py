#!run.sh
"""
CE Modules: Core CE Neural Network Components

Contains the fundamental neural network modules for CE (Corridor + Flow + Witness) architecture:
- CEEnhancedLSTM: LSTM with CE regularization
- MirrorOperator: Mirror symmetry operations
- CurvatureCouplingLayer: Curvature-aware coupling
- GuardianThreshold: CE timing threshold mechanisms
- ZetaRegularization: Functional equation regularization

These modules implement the discrete geometric intelligence principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class CEConfig:
    """Configuration for CE modules."""
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    kappa: float = 0.35  # Curvature threshold
    chi: float = 0.638  # FEG ratio
    zeta_strength: float = 1.0


class MirrorOperator(nn.Module):
    """
    Mirror Operator: Implements discrete mirror symmetry operations.

    Based on the principle that mirror phases (n mod 4) determine
    the symmetry properties of shell structures.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Mirror phase detectors
        self.phase_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 4)  # 4 mirror phases
        )

        # Mirror transformation matrices (learned)
        self.mirror_transform = nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize with reflection-like properties
        with torch.no_grad():
            # Create reflection matrix (reverse order)
            reflection = torch.eye(hidden_size)
            reflection = torch.flip(reflection, dims=[0])
            self.mirror_transform.weight.data = reflection * 0.9

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mirror operator.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of (transformed_x, mirror_loss)
        """
        batch_size, seq_len, hidden_size = x.shape

        # Detect mirror phases
        phases = self.phase_detector(x.mean(dim=1))  # (batch_size, 4)
        phase_probs = F.softmax(phases, dim=-1)

        # Apply mirror transformation
        mirrored = self.mirror_transform(x)

        # Compute mirror consistency loss
        # Mirror operation should be involution: M^2 = I
        mirror_squared = self.mirror_transform(mirrored)
        mirror_consistency = F.mse_loss(mirror_squared, x)

        return mirrored, mirror_consistency


class CurvatureCouplingLayer(nn.Module):
    """
    Curvature Coupling Layer: Implements curvature-aware information flow.

    Based on the principle that information flow should respect
    the curvature fields defined by mirror phase boundaries.
    """

    def __init__(self, hidden_size: int, kappa: float = 0.35):
        super().__init__()
        self.hidden_size = hidden_size
        self.kappa = kappa

        # Curvature field computation
        self.curvature_encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Coupling strength modulation
        self.coupling_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Evolution operator
        self.evolution_op = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply curvature coupling.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of (coupled_x, curvature_loss)
        """
        batch_size, seq_len, hidden_size = x.shape

        # Compute local curvature fields
        curvatures = []
        for i in range(seq_len):
            if i == 0:
                # Boundary condition
                local_curvature = self.curvature_encoder(
                    torch.cat([x[:, i], x[:, i]], dim=-1)
                )
            else:
                local_curvature = self.curvature_encoder(
                    torch.cat([x[:, i-1], x[:, i]], dim=-1)
                )
            curvatures.append(local_curvature)

        curvature_field = torch.stack(curvatures, dim=1)  # (batch_size, seq_len, 1)

        # Apply curvature-aware coupling
        coupling_strength = self.coupling_gate(x)  # (batch_size, seq_len, hidden_size)

        # Modulate by curvature
        kappa_tensor = torch.full_like(curvature_field, self.kappa)
        evolution = self.evolution_op(x)

        # Curvature-modulated update
        dx = kappa_tensor.unsqueeze(-1) * evolution
        coupled_x = x + coupling_strength * dx

        # Compute curvature consistency loss
        # Adjacent positions should have smooth curvature transitions
        curvature_diff = torch.abs(curvature_field[:, 1:] - curvature_field[:, :-1])
        curvature_smoothness = torch.mean(curvature_diff)

        return coupled_x, curvature_smoothness


class GuardianThreshold(nn.Module):
    """
    Guardian Threshold: CE timing mechanism for early stopping and regularization.

    Implements the Kappa Guardian principle: stop when zeta loss exceeds threshold,
    indicating loss of geometric awareness.
    """

    def __init__(self, threshold: float = 0.35, patience: int = 5):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.reset()

    def reset(self):
        """Reset guardian state."""
        self.violation_count = 0
        self.best_zeta = float('inf')
        self.early_stop_triggered = False

    def forward(self, zeta_loss: float) -> Tuple[bool, float]:
        """
        Check if guardian threshold is violated.

        Args:
            zeta_loss: Current zeta regularization loss

        Returns:
            Tuple of (should_stop, guardian_score)
        """
        if self.early_stop_triggered:
            return True, 0.0

        # Update best zeta
        if zeta_loss < self.best_zeta:
            self.best_zeta = zeta_loss
            self.violation_count = 0
        else:
            self.violation_count += 1

        # Check threshold violation
        threshold_violation = zeta_loss > self.threshold

        if threshold_violation:
            self.violation_count += 1

        # Check early stopping condition
        if self.violation_count >= self.patience:
            self.early_stop_triggered = True

        # Guardian score (lower is better, indicates awareness)
        guardian_score = zeta_loss / (self.threshold + 1e-8)

        return self.early_stop_triggered, guardian_score


class ZetaRegularization(nn.Module):
    """
    Zeta Regularization: Enforces functional equation constraints.

    Implements the CE principle that trajectories should satisfy
    discrete functional equations derived from zeta function structure.
    """

    def __init__(self, hidden_size: int, zeta_strength: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.zeta_strength = zeta_strength

        # Functional equation layers
        self.complex_proj = nn.Linear(hidden_size, 2)  # Project to complex plane
        self.equation_encoder = nn.Sequential(
            nn.Linear(4, hidden_size),  # Takes 2 complex numbers (4 real values)
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def functional_equation_loss(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute functional equation regularization loss.

        Args:
            points: Points in complex plane (batch_size, 2) representing (real, imag)

        Returns:
            Functional equation loss
        """
        # Simple functional equation: f(z) + f(1-z) = constant
        # For complex numbers, this becomes a constraint on the representation

        batch_size = points.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=points.device)

        # Sample pairs of points
        indices = torch.randperm(batch_size, device=points.device)[:min(10, batch_size)]
        selected_points = points[indices]  # (num_pairs, 2)

        # Compute f(z) and f(1-z) proxy
        z = selected_points
        one_minus_z = 1.0 - selected_points

        # Encode functional relationship
        equation_input = torch.cat([z, one_minus_z], dim=-1)  # (num_pairs, 4)
        equation_output = self.equation_encoder(equation_input)  # (num_pairs, 1)

        # Loss: functional equation should be approximately constant
        if equation_output.shape[0] > 1:
            equation_variance = torch.var(equation_output, dim=0)
            loss = equation_variance.mean()
        else:
            loss = torch.tensor(0.0, device=points.device)

        return loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply zeta regularization.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)

        Returns:
            Tuple of (x, zeta_loss)
        """
        original_shape = x.shape
        if len(original_shape) == 2:
            # Handle (batch_size, hidden_size) case
            batch_size, hidden_size = original_shape
            seq_len = 1
            x_reshaped = x.unsqueeze(1)  # (batch_size, 1, hidden_size)
        elif len(original_shape) == 3:
            # Handle (batch_size, seq_len, hidden_size) case
            batch_size, seq_len, hidden_size = original_shape
            x_reshaped = x
        else:
            # Handle higher dimensional case (flatten extra dimensions)
            batch_size = original_shape[0]
            seq_len = int(torch.prod(torch.tensor(original_shape[1:-1])))
            hidden_size = original_shape[-1]
            x_reshaped = x.view(batch_size, seq_len, hidden_size)

        # Project to complex plane
        complex_coords = self.complex_proj(x_reshaped.view(-1, hidden_size))  # (batch*seq_len, 2)
        complex_coords = complex_coords.view(batch_size, seq_len, 2)  # (batch_size, seq_len, 2)

        # Compute functional equation loss across sequence
        zeta_loss = 0.0
        for i in range(seq_len):
            loss_i = self.functional_equation_loss(complex_coords[:, i])  # (batch_size, 2) -> scalar
            zeta_loss += loss_i

        zeta_loss = zeta_loss / seq_len if seq_len > 0 else zeta_loss
        zeta_loss = zeta_loss * self.zeta_strength

        return x, zeta_loss


class CEEnhancedLSTM(nn.Module):
    """
    CE-Enhanced LSTM: LSTM with integrated CE regularization.

    Combines standard LSTM with CE modules for geometry-aware processing.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.1, ce_config: Optional[CEConfig] = None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # CE configuration
        self.ce_config = ce_config or CEConfig(hidden_size=hidden_size)

        # Standard LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTMCell(input_dim, hidden_size))

        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # CE enhancement modules
        self.mirror_op = MirrorOperator(hidden_size)
        self.curvature_layer = CurvatureCouplingLayer(hidden_size, self.ce_config.kappa)
        self.zeta_reg = ZetaRegularization(hidden_size, self.ce_config.zeta_strength)

        # CE timing
        self.guardian = GuardianThreshold(self.ce_config.kappa)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through CE-enhanced LSTM.

        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            Tuple of (outputs, (h_n, c_n), zeta_loss)
        """
        batch_size, seq_len, input_size = x.shape

        # Initialize hidden states
        h_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            # Process through LSTM layers
            for layer_idx, lstm_cell in enumerate(self.lstm_layers):
                if layer_idx == 0:
                    h_t[layer_idx], c_t[layer_idx] = lstm_cell(x_t, (h_t[layer_idx], c_t[layer_idx]))
                else:
                    h_t[layer_idx], c_t[layer_idx] = lstm_cell(h_t[layer_idx-1], (h_t[layer_idx], c_t[layer_idx]))

                # Apply dropout between layers
                if layer_idx < self.num_layers - 1:
                    h_t[layer_idx] = self.dropout_layer(h_t[layer_idx])

            # Get final layer output
            lstm_output = h_t[-1]  # (batch_size, hidden_size)

            # Apply CE enhancements
            # 1. Mirror operation
            mirrored, mirror_loss = self.mirror_op(lstm_output.unsqueeze(1))
            mirrored = mirrored.squeeze(1)

            # 2. Curvature coupling
            coupled, curvature_loss = self.curvature_layer(mirrored.unsqueeze(1))
            coupled = coupled.squeeze(1)

            # 3. Zeta regularization
            _, zeta_loss = self.zeta_reg(coupled.unsqueeze(1))

            # Combine losses
            total_zeta_loss = mirror_loss + curvature_loss + zeta_loss

            # Check guardian threshold
            should_stop, guardian_score = self.guardian(total_zeta_loss.item())

            outputs.append(coupled)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)

        # Final hidden states
        h_n = h_t[-1].unsqueeze(0)  # (1, batch_size, hidden_size)
        c_n = c_t[-1].unsqueeze(0)  # (1, batch_size, hidden_size)

        # Total zeta loss across sequence
        _, final_zeta_loss = self.zeta_reg(outputs)

        return outputs, (h_n, c_n), final_zeta_loss


# Convenience functions for creating CE modules
def create_ce_lstm(input_size: int, hidden_size: int, **kwargs) -> CEEnhancedLSTM:
    """Create CE-enhanced LSTM with default configuration."""
    return CEEnhancedLSTM(input_size, hidden_size, **kwargs)


def create_ce_modules(hidden_size: int) -> Dict[str, nn.Module]:
    """Create complete set of CE modules."""
    return {
        'mirror_op': MirrorOperator(hidden_size),
        'curvature_layer': CurvatureCouplingLayer(hidden_size),
        'zeta_reg': ZetaRegularization(hidden_size),
        'guardian': GuardianThreshold()
    }


# Test the modules
if __name__ == "__main__":
    print("Testing CE Modules...")

    # Test CEEnhancedLSTM
    batch_size, seq_len, input_size, hidden_size = 2, 5, 10, 32

    ce_lstm = CEEnhancedLSTM(input_size, hidden_size)
    x = torch.randn(batch_size, seq_len, input_size)

    print(f"Input shape: {x.shape}")

    try:
        outputs, (h_n, c_n), zeta_loss = ce_lstm(x)
        print(f"Output shape: {outputs.shape}")
        print(f"Hidden state shape: {h_n.shape}")
        print(f"Zeta loss: {zeta_loss.item():.4f}")
        print("✅ CEEnhancedLSTM working!")
    except Exception as e:
        print(f"❌ CEEnhancedLSTM error: {e}")
        import traceback
        traceback.print_exc()

    # Test individual modules
    mirror_op = MirrorOperator(hidden_size)
    curvature_layer = CurvatureCouplingLayer(hidden_size)
    zeta_reg = ZetaRegularization(hidden_size)

    x_test = torch.randn(batch_size, seq_len, hidden_size)

    try:
        mirrored, mirror_loss = mirror_op(x_test)
        print(f"Mirror op - Output shape: {mirrored.shape}, Loss: {mirror_loss.item():.4f}")

        coupled, curvature_loss = curvature_layer(x_test)
        print(f"Curvature layer - Output shape: {coupled.shape}, Loss: {curvature_loss.item():.4f}")

        _, zeta_loss = zeta_reg(x_test)
        print(f"Zeta reg - Loss: {zeta_loss.item():.4f}")

        print("✅ All CE modules working!")
    except Exception as e:
        print(f"❌ Module error: {e}")
        import traceback
        traceback.print_exc()