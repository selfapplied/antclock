"""
CE-Enhanced COGS Model

Integrates CE framework components into COGS semantic parsing architecture:
- Mirror operators for symmetry-aware parsing
- Curvature coupling for semantic composition
- Zeta regularization for logical consistency
- Guardian thresholds for predicate separation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .cogs import COGSBenchmark, COGSDataset, COGSVocab, COGSExample
from .ce_modules import (
    MirrorOperator, CurvatureCouplingLayer, GuardianThreshold,
    ZetaRegularization, CEAttention, CEEnhancedLSTM
)


class CEEnhancedCOGSModel(nn.Module):
    """
    CE-Enhanced COGS Model with integrated CE components.

    Incorporates:
    - Bidirectional CE-LSTM encoder
    - CE attention for semantic composition
    - Curvature-coupled decoder
    - Zeta regularization for logical consistency
    """

    def __init__(self, vocab_sentence: COGSVocab, vocab_lf: COGSVocab,
                 embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2,
                 chi_feg: float = 0.638, kappa: float = 0.35,
                 use_ce_attention: bool = True, use_zeta_reg: bool = True):
        super().__init__()

        self.vocab_sentence = vocab_sentence
        self.vocab_lf = vocab_lf
        self.chi_feg = chi_feg
        self.kappa = kappa
        self.use_ce_attention = use_ce_attention
        self.use_zeta_reg = use_zeta_reg

        # Encoder with CE enhancements
        self.encoder_embed = nn.Embedding(len(vocab_sentence), embed_dim)

        # Bidirectional CE-enhanced encoder
        self.encoder_lstm = CEEnhancedLSTM(
            embed_dim, hidden_dim, chi_feg, kappa, use_zeta_reg
        )

        # CE attention mechanism for semantic composition
        if use_ce_attention:
            self.encoder_attention = CEAttention(hidden_dim)
        else:
            self.encoder_attention = None

        # Decoder
        self.decoder_embed = nn.Embedding(len(vocab_lf), embed_dim)

        # CE-enhanced decoder with attention
        self.decoder_lstm = CEEnhancedLSTM(
            embed_dim + hidden_dim, hidden_dim, chi_feg, kappa, use_zeta_reg  # +hidden_dim for attention
        )

        # Output projection with guardian threshold
        self.output_proj = nn.Linear(hidden_dim, len(vocab_lf))
        self.guardian_activation = GuardianThreshold(kappa)

        # Zeta regularization for logical consistency
        if use_zeta_reg:
            self.zeta_reg = ZetaRegularization(hidden_dim)

    def encode(self, sentence_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Encode sentence with CE-enhanced bidirectional processing.

        Returns:
            Tuple of (encoder_outputs, final_hidden, zeta_loss)
        """
        embeds = self.encoder_embed(sentence_ids)

        # Forward pass
        fwd_outputs, (fwd_hidden, fwd_cell), zeta_loss_fwd = self.encoder_lstm(embeds)

        # Backward pass (reverse sequence)
        rev_embeds = torch.flip(embeds, dims=[1])
        rev_outputs, (rev_hidden, rev_cell), zeta_loss_rev = self.encoder_lstm(rev_embeds)
        rev_outputs = torch.flip(rev_outputs, dims=[1])  # Flip back

        # Combine bidirectional outputs
        encoder_outputs = torch.cat([fwd_outputs, rev_outputs], dim=-1)

        # Apply CE attention for semantic composition
        if self.encoder_attention is not None:
            attended_outputs = self.encoder_attention(
                encoder_outputs, encoder_outputs, encoder_outputs
            )
            encoder_outputs = encoder_outputs + attended_outputs  # Residual

        # Combine final hidden states
        final_hidden = torch.cat([fwd_hidden, rev_hidden], dim=-1)

        total_zeta_loss = zeta_loss_fwd + zeta_loss_rev

        return encoder_outputs, final_hidden, total_zeta_loss

    def decode_step(self, input_token: torch.Tensor, hidden: torch.Tensor,
                   encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Single CE-enhanced decoder step with attention."""
        embed = self.decoder_embed(input_token.unsqueeze(1))

        # Get attention context from encoder
        if self.encoder_attention is not None:
            # Cross-attention: decoder queries encoder
            context = self.encoder_attention(
                hidden.unsqueeze(1), encoder_outputs, encoder_outputs
            ).squeeze(1)
        else:
            # Simple average pooling as fallback
            context = encoder_outputs.mean(dim=1)

        # Concatenate embedding and context
        decoder_input = torch.cat([embed.squeeze(1), context], dim=-1).unsqueeze(1)

        # CE-enhanced LSTM decoding
        decoder_output, new_hidden, zeta_loss = self.decoder_lstm(decoder_input, (hidden, hidden))

        # Project to vocabulary with guardian threshold
        logits = self.output_proj(decoder_output.squeeze(1))
        logits = self.guardian_activation(logits)

        return logits, new_hidden, zeta_loss

    def forward(self, sentence_ids: torch.Tensor, lf_ids: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, float]:
        """
        Forward pass with CE regularization.

        Returns:
            Tuple of (outputs, total_zeta_loss)
        """
        batch_size = sentence_ids.size(0)
        max_len = lf_ids.size(1) if lf_ids is not None else 50

        # Encode with CE components
        encoder_outputs, hidden, zeta_loss_enc = self.encode(sentence_ids)

        # Decode
        outputs = []
        input_token = torch.full((batch_size,), self.vocab_lf.sos_idx,
                               device=sentence_ids.device)

        total_zeta_loss = zeta_loss_enc

        for t in range(max_len):
            logits, hidden, zeta_loss_dec = self.decode_step(
                input_token, hidden, encoder_outputs
            )
            outputs.append(logits.unsqueeze(1))
            total_zeta_loss += zeta_loss_dec

            # Teacher forcing
            if lf_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = lf_ids[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        outputs = torch.cat(outputs, dim=1)

        # Apply final zeta regularization for logical consistency
        if self.use_zeta_reg and lf_ids is not None:
            final_zeta_loss = self.zeta_reg(outputs)[1]
            total_zeta_loss += final_zeta_loss

        return outputs, total_zeta_loss


class CEEnhancedCOGSBenchmark(COGSBenchmark):
    """
    CE-Enhanced COGS Benchmark with integrated CE components.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2,
                 learning_rate: float = 1e-3, batch_size: int = 32,
                 chi_feg: float = 0.638, kappa: float = 0.35,
                 use_ce_attention: bool = True, use_zeta_reg: bool = True,
                 zeta_reg_weight: float = 0.1):
        super().__init__(embed_dim, hidden_dim, num_layers, learning_rate, batch_size)

        self.chi_feg = chi_feg
        self.kappa = kappa
        self.use_ce_attention = use_ce_attention
        self.use_zeta_reg = use_zeta_reg
        self.zeta_reg_weight = zeta_reg_weight

    def create_model(self) -> CEEnhancedCOGSModel:
        """Create a CE-enhanced COGS model."""
        return CEEnhancedCOGSModel(
            self.vocab_sentence, self.vocab_lf,
            self.embed_dim, self.hidden_dim, self.num_layers,
            self.chi_feg, self.kappa,
            self.use_ce_attention, self.use_zeta_reg
        )

    def train_epoch(self, model: CEEnhancedCOGSModel, optimizer: optim.Optimizer,
                   criterion: nn.CrossEntropyLoss, device: str = 'cpu') -> Tuple[float, float]:
        """Train for one epoch with CE regularization."""
        model.train()
        total_loss = 0
        total_zeta_loss = 0

        for batch in self.train_loader:
            sentence_ids = batch['sentence_ids'].to(device)
            lf_ids = batch['lf_ids'].to(device)

            optimizer.zero_grad()

            # Forward pass with CE regularization
            outputs, zeta_loss = model(sentence_ids, lf_ids, teacher_forcing_ratio=0.5)

            # Standard cross-entropy loss
            ce_loss = criterion(outputs.view(-1, len(self.vocab_lf)),
                              lf_ids.view(-1))

            # Combined loss with zeta regularization
            total_batch_loss = ce_loss + self.zeta_reg_weight * zeta_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += ce_loss.item()
            total_zeta_loss += zeta_loss.item()

        avg_loss = total_loss / len(self.train_loader)
        avg_zeta_loss = total_zeta_loss / len(self.train_loader)

        return avg_loss, avg_zeta_loss

    def train_model(self, model: CEEnhancedCOGSModel, num_epochs: int = 100,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """Train CE-enhanced model and return training history."""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_lf.pad_idx)

        train_losses = []
        zeta_losses = []
        dev_accuracies = []

        for epoch in tqdm(range(num_epochs), desc="Training CE-COGS"):
            # Train
            train_loss, zeta_loss = self.train_epoch(model, optimizer, criterion, device)
            train_losses.append(train_loss)
            zeta_losses.append(zeta_loss)

            # Evaluate on dev set
            if epoch % 10 == 0:
                dev_metrics = self.evaluate(model, 'dev', device)
                dev_accuracies.append(dev_metrics['accuracy'])
                print(f"Epoch {epoch}: CE Loss={train_loss:.4f}, "
                      f"Zeta Loss={zeta_loss:.4f}, Dev Acc={dev_metrics['accuracy']:.4f}")

        return {
            'train_losses': train_losses,
            'zeta_losses': zeta_losses,
            'dev_accuracies': dev_accuracies
        }


def run_ce_cogs_experiment(num_epochs: int = 100, device: str = 'cpu',
                          chi_feg: float = 0.638, kappa: float = 0.35) -> Dict[str, float]:
    """Run CE-enhanced COGS experiment."""
    print("🔬 Running CE-Enhanced COGS Experiment...")
    print(f"Parameters: χ_FEG={chi_feg}, κ={kappa}")

    benchmark = COGSBenchmark()
    ce_benchmark = CEEnhancedCOGSBenchmark(chi_feg=chi_feg, kappa=kappa)

    # Create and train CE-enhanced model
    model = ce_benchmark.create_model()
    history = ce_benchmark.train_model(model, num_epochs, device)

    # Final evaluation on both dev and test
    dev_metrics = ce_benchmark.evaluate(model, 'dev', device)
    test_metrics = ce_benchmark.evaluate(model, 'test', device)

    results = {
        'train_loss_final': history['train_losses'][-1],
        'zeta_loss_final': history['zeta_losses'][-1],
        'dev_accuracy': dev_metrics['accuracy'],
        'test_accuracy': test_metrics['accuracy'],
        'dev_correct': dev_metrics['correct'],
        'dev_total': dev_metrics['total'],
        'test_correct': test_metrics['correct'],
        'test_total': test_metrics['total'],
        'chi_feg': chi_feg,
        'kappa': kappa
    }

    print(f"Baseline Loss: {results['train_loss_final']:.4f}")
    print(f"Dev Accuracy: {results['dev_accuracy']:.1%} ({results['dev_correct']}/{results['dev_total']})")
    print(f"Test Accuracy: {results['test_accuracy']:.1%} ({results['test_correct']}/{results['test_total']})")

    return results


def ablation_study_cogs(device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """Perform ablation study on CE components for COGS."""
    print("🔬 Performing COGS Ablation Study...")

    configurations = [
        {'name': 'baseline', 'ce_attention': False, 'zeta_reg': False},
        {'name': 'ce_attention_only', 'ce_attention': True, 'zeta_reg': False},
        {'name': 'zeta_reg_only', 'ce_attention': False, 'zeta_reg': True},
        {'name': 'full_ce', 'ce_attention': True, 'zeta_reg': True},
    ]

    results = {}

    for config in configurations:
        print(f"\nTesting configuration: {config['name']}")

        benchmark = CEEnhancedCOGSBenchmark(
            use_ce_attention=config['ce_attention'],
            use_zeta_reg=config['zeta_reg']
        )

        model = benchmark.create_model()
        history = benchmark.train_model(model, num_epochs=50, device=device)
        metrics = benchmark.evaluate(model, 'test', device)

        results[config['name']] = {
            'test_accuracy': metrics['accuracy'],
            'train_loss_final': history['train_losses'][-1],
            'zeta_loss_final': history['zeta_losses'][-1] if 'zeta_losses' in history else 0.0
        }

        print(f"  Accuracy: {metrics['accuracy']:.1%}")

    return results


if __name__ == "__main__":
    # Test CE-enhanced COGS
    results = run_ce_cogs_experiment(num_epochs=50)
    print(f"\nCE-COGS Results: {results}")

    # Run ablation study
    ablation_results = ablation_study_cogs()
    print(f"\nAblation Study Results: {ablation_results}")
