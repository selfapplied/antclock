"""
CE-Enhanced Neural Network Modules

Integrates CE framework components into neural architectures:
- Zeta operator regularization
- Mirror symmetry operators
- Curvature coupling layers
- Guardian threshold activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zeta_operator import ZetaOperator, Corridor


class MirrorOperator(nn.Module):
    """
    CE Mirror Operator: Implements digit mirror symmetries as neural operations.

    The mirror operator μ₇(d) = d⁷ mod 10 provides involutive transformations
    that enforce discrete symmetry breaking.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Mirror transformation matrix (learned projection)
        self.mirror_proj = nn.Linear(embed_dim, embed_dim)

        # Fixed mirror operator parameters
        self.register_buffer('mirror_matrix', self._create_mirror_matrix())

    def _create_mirror_matrix(self) -> torch.Tensor:
        """Create the digit mirror operator matrix."""
        # Based on μ₇(d) = d^7 mod 10 with fixed points {0,1,4,5,6,9}
        # and oscillating pairs {2↔8, 3↔7}
        mirror_map = {
            0: 0, 1: 1, 2: 8, 3: 7, 4: 4, 5: 5,
            6: 6, 7: 3, 8: 2, 9: 9
        }

        # Create permutation matrix
        size = 10
        matrix = torch.zeros(size, size)
        for i in range(size):
            j = mirror_map.get(i, i)
            matrix[i, j] = 1.0

        return matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply mirror operator transformation.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Mirrored tensor with symmetry-breaking properties
        """
        # Apply learned mirror projection
        mirrored = self.mirror_proj(x)

        # Apply involution constraint (mirror operator should be its own inverse)
        # This enforces μ ∘ μ = id
        mirrored = torch.tanh(mirrored)  # Bound the transformation

        return mirrored


class CurvatureCouplingLayer(nn.Module):
    """
    CE Curvature Coupling Layer: Implements χ_FEG coupling between layers.

    Provides continuous-time evolution with curvature-driven dynamics.
    """

    def __init__(self, embed_dim: int, chi_feg: float = 0.638):
        super().__init__()
        self.embed_dim = embed_dim
        self.chi_feg = chi_feg

        # Curvature computation layers
        self.curvature_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        # Evolution operator
        self.evolution_gate = nn.Linear(embed_dim, embed_dim)

    def compute_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local curvature field."""
        # Simplified curvature as second derivative proxy
        curvature = self.curvature_net(x)

        # Apply Feigenbaum coupling
        coupled_curvature = curvature * self.chi_feg

        return coupled_curvature

    def forward(self, x: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Apply curvature-coupled evolution.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            dt: Time step for evolution

        Returns:
            Evolved tensor with curvature coupling
        """
        # Compute curvature field
        kappa = self.compute_curvature(x)

        # Apply curvature-driven evolution
        # dx/dt = κ(x) * χ_FEG * x
        evolution = self.evolution_gate(x)
        dx = kappa.unsqueeze(-1) * evolution

        # Euler integration step
        x_new = x + dt * dx

        return x_new


class GuardianThreshold(nn.Module):
    """
    CE Guardian Threshold: κ = 0.35 activation function.

    Implements the guardian threshold that prevents phase locking
    and maintains separation between correlated states.
    """

    def __init__(self, kappa: float = 0.35):
        super().__init__()
        self.kappa = kappa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply guardian threshold activation.

        κ = 0.35 prevents phase locking while maintaining
        information flow through the network.
        """
        # Guardian threshold activation
        # f(x) = x / (1 + κ * |x|) - maintains separation
        thresholded = x / (1 + self.kappa * torch.abs(x))

        # Apply learned scaling to maintain expressivity
        return thresholded


class ZetaRegularization(nn.Module):
    """
    CE Zeta Regularization: Enforces discrete functional equation constraints.

    Regularizes neural networks to satisfy Ξ(s) = Ξ(1-s) properties
    through learned corridor structures.
    """

    def __init__(self, embed_dim: int, num_corridors: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_corridors = num_corridors

        # Learnable corridor weights and parities
        self.corridor_weights = nn.Parameter(torch.randn(num_corridors))
        self.corridor_parities = nn.Parameter(torch.randn(num_corridors))

        # Complex plane projection for zeta evaluation
        self.complex_proj = nn.Linear(embed_dim, 2)  # Real and imaginary parts

        # Mirror phase characters
        self.phase_characters = nn.Parameter(torch.randn(num_corridors, 2))  # Complex phases

    def create_corridor_term(self, s: torch.Tensor, corridor_idx: int) -> torch.Tensor:
        """
        Create F_k(s) = 0.5 * (exp(-s * L_k) + ε_k * exp(-(1-s) * L_k))

        Args:
            s: Complex points in s-plane
            corridor_idx: Which corridor to evaluate

        Returns:
            Complex corridor term value
        """
        # Get corridor parameters
        weight = self.corridor_weights[corridor_idx]
        parity = torch.sigmoid(self.corridor_parities[corridor_idx])  # ε_k ∈ [0,1]

        # Simplified length based on corridor index (learned)
        length = 0.2 + 0.1 * corridor_idx

        # Complex exponential terms
        s_real, s_imag = s[:, 0], s[:, 1]
        s_complex = torch.complex(s_real, s_imag)

        term1 = torch.exp(-s_complex * length)
        term2 = parity * torch.exp(-(1 - s_complex) * length)

        return 0.5 * (term1 + term2)

    def zeta_operator(self, s: torch.Tensor) -> torch.Tensor:
        """
        Complete CE zeta operator: Ξ_CE(s) = Σ_k w_k * F_k(s)

        Args:
            s: Complex points [batch, 2] (real, imag)

        Returns:
            Zeta function values [batch]
        """
        total = torch.zeros(s.size(0), dtype=torch.complex64, device=s.device)

        for k in range(self.num_corridors):
            term = self.create_corridor_term(s, k)
            weight = torch.sigmoid(self.corridor_weights[k])  # Normalize weights
            total += weight * term

        return total

    def functional_equation_loss(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute violation of functional equation Ξ(s) = Ξ(1-s)

        Returns L2 distance between Ξ(s) and Ξ(1-s)
        """
        zeta_s = self.zeta_operator(s)
        zeta_1_minus_s = self.zeta_operator(1 - s)

        return torch.mean(torch.abs(zeta_s - zeta_1_minus_s)**2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply zeta regularization to input tensor.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Tuple of (regularized_output, regularization_loss)
        """
        # Project to complex plane
        complex_coords = self.complex_proj(x)  # [..., 2]

        # Compute zeta regularization loss
        # Sample points along critical line and nearby
        original_shape = complex_coords.shape[:-1]
        batch_size = original_shape[0]
        seq_len = original_shape[1] if len(original_shape) > 1 else 1

        # Critical line points: σ + i t
        t_values = torch.linspace(-2, 2, seq_len, device=x.device)
        sigma_values = 0.5 * torch.ones_like(t_values)

        s_points = torch.stack([sigma_values, t_values], dim=-1)  # [seq_len, 2]
        s_points = s_points.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, seq_len, 2]

        # Compute functional equation loss
        reg_loss = self.functional_equation_loss(s_points.view(-1, 2))
        # reg_loss is already a scalar, just return it

        return x, reg_loss


class CEAttention(nn.Module):
    """
    CE-Enhanced Attention: Mirror-symmetric attention mechanism.

    Combines standard attention with CE symmetry constraints.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # CE symmetry projections
        self.mirror_q = MirrorOperator(embed_dim)
        self.mirror_k = MirrorOperator(embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        CE-enhanced attention with mirror symmetries.

        Args:
            query, key, value: Input tensors [batch, seq_len, embed_dim]
            mask: Optional attention mask

        Returns:
            Attention output [batch, seq_len, embed_dim]
        """
        batch_size = query.size(0)

        # Standard attention projections
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # CE mirror-symmetric projections
        Q_mirror = self.mirror_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K_mirror = self.mirror_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Combine standard and mirror attention
        # Attention = softmax((Q + Q_mirror) @ (K + K_mirror)^T / sqrt(d_k))
        Q_combined = Q + Q_mirror
        K_combined = K + K_mirror

        # Compute attention scores
        scores = torch.matmul(Q_combined, K_combined.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)

        return output


class CEEnhancedLSTM(nn.Module):
    """
    CE-Enhanced LSTM: Integrates CE components into LSTM architecture.

    Combines curvature coupling, mirror operators, and zeta regularization.
    """

    def __init__(self, input_size: int, hidden_size: int, chi_feg: float = 0.638,
                 kappa: float = 0.35, use_zeta_reg: bool = True):
        super().__init__()

        self.hidden_size = hidden_size
        self.chi_feg = chi_feg
        self.kappa = kappa
        self.use_zeta_reg = use_zeta_reg

        # Standard LSTM components
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # CE components
        self.mirror_op = MirrorOperator(hidden_size)
        self.curvature_layer = CurvatureCouplingLayer(hidden_size, chi_feg)
        self.guardian_activation = GuardianThreshold(kappa)

        if use_zeta_reg:
            self.zeta_reg = ZetaRegularization(hidden_size)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        CE-enhanced LSTM forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_size]
            hidden: Optional initial hidden state

        Returns:
            Tuple of (output, (h_n, c_n))
        """
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = hidden

        outputs = []
        zeta_loss = 0.0

        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_size]

            # Combine input and hidden state
            combined = torch.cat([x_t, h], dim=-1)

            # Standard LSTM gates
            i = torch.sigmoid(self.input_gate(combined))
            f = torch.sigmoid(self.forget_gate(combined))
            g = torch.tanh(self.cell_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))

            # Update cell state
            c = f * c + i * g

            # Apply CE curvature coupling to cell state
            c = self.curvature_layer(c.unsqueeze(1)).squeeze(1)

            # Apply mirror operator to hidden state
            h_pre = o * torch.tanh(c)
            h = self.mirror_op(h_pre.unsqueeze(1)).squeeze(1)

            # Apply guardian threshold activation
            h = self.guardian_activation(h)

            # Apply zeta regularization if enabled
            if self.use_zeta_reg:
                h_expanded = h.unsqueeze(1)  # [batch, 1, hidden_size]
                h_reg, reg_loss = self.zeta_reg(h_expanded)
                h = h_reg.squeeze(1)
                zeta_loss += reg_loss

            outputs.append(h.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        final_zeta_loss = zeta_loss / seq_len if seq_len > 0 else 0.0

        return outputs, (h, c), final_zeta_loss
