"""
CE-Enhanced PCFG Model

Integrates CE framework components into PCFG parsing architecture:
- Mirror operators for symmetry-aware parsing
- Curvature coupling for compositional dynamics
- Zeta regularization for grammatical consistency
- Guardian thresholds for structure preservation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .pcfg import PCFGBenchmark, PCFGModel, PCFGVocab, PCFGExample
from .ce_modules import (
    MirrorOperator, CurvatureCouplingLayer, GuardianThreshold,
    ZetaRegularization, CEAttention, CEEnhancedLSTM
)
try:
    from .ce_timing import create_ce_timed_trainer
except ImportError:
    from ce_timing import create_ce_timed_trainer


class CEEnhancedPCFGModel(nn.Module):
    """
    CE-Enhanced PCFG Model with integrated CE components.

    Incorporates:
    - CE-enhanced encoder with mirror symmetries
    - Curvature-coupled attention for composition
    - Zeta regularization for grammatical consistency
    """

    def __init__(self, vocab_sentence: PCFGVocab, vocab_tree: PCFGVocab,
                 embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2,
                 chi_feg: float = 0.638, kappa: float = 0.35,
                 use_ce_attention: bool = True, use_zeta_reg: bool = True):
        super().__init__()

        self.vocab_sentence = vocab_sentence
        self.vocab_tree = vocab_tree
        self.chi_feg = chi_feg
        self.kappa = kappa
        self.use_ce_attention = use_ce_attention
        self.use_zeta_reg = use_zeta_reg

        # CE-enhanced encoder
        self.encoder_embed = nn.Embedding(len(vocab_sentence), embed_dim)

        # CE-enhanced encoder LSTM
        self.encoder_lstm = CEEnhancedLSTM(
            embed_dim, hidden_dim, chi_feg, kappa, use_zeta_reg
        )

        # CE attention mechanism for semantic composition
        if use_ce_attention:
            self.encoder_attention = CEAttention(hidden_dim)
        else:
            self.encoder_attention = None

        # Decoder
        self.decoder_embed = nn.Embedding(len(vocab_tree), embed_dim)

        # CE-enhanced decoder LSTM
        self.decoder_lstm = CEEnhancedLSTM(
            embed_dim + hidden_dim, hidden_dim, chi_feg, kappa, use_zeta_reg
        )

        # Output projection with guardian threshold
        self.output_proj = nn.Linear(hidden_dim, len(vocab_tree))
        self.guardian_activation = GuardianThreshold(kappa)

        # Zeta regularization for grammatical consistency
        if use_zeta_reg:
            self.zeta_reg = ZetaRegularization(hidden_dim)

    def encode(self, sentence_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Encode sentence with CE enhancements."""
        embeds = self.encoder_embed(sentence_ids)

        # Apply CE-enhanced LSTM encoding
        encoder_outputs, (hidden, cell), zeta_loss_enc = self.encoder_lstm(embeds)

        # Apply CE attention for semantic composition
        if self.encoder_attention is not None:
            attended_outputs = self.encoder_attention(
                encoder_outputs, encoder_outputs, encoder_outputs
            )
            encoder_outputs = encoder_outputs + attended_outputs  # Residual

        return encoder_outputs, hidden, zeta_loss_enc

    def decode_step(self, input_token: torch.Tensor, hidden: torch.Tensor,
                   encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Single CE-enhanced decoder step."""
        embed = self.decoder_embed(input_token.unsqueeze(1))

        # Get attention context from encoder
        if self.encoder_attention is not None:
            # Cross-attention between decoder and encoder
            context = self.encoder_attention(
                hidden.unsqueeze(1), encoder_outputs, encoder_outputs
            ).squeeze(1)
        else:
            # Simple average pooling
            context = encoder_outputs.mean(dim=1)

        # Concatenate embedding and context
        decoder_input = torch.cat([embed.squeeze(1), context], dim=-1).unsqueeze(1)

        # CE-enhanced LSTM decoding
        decoder_output, new_hidden, zeta_loss_dec = self.decoder_lstm(decoder_input, (hidden, hidden))

        # Project to vocabulary with guardian threshold
        logits = self.output_proj(decoder_output.squeeze(1))
        logits = self.guardian_activation(logits)

        return logits, new_hidden, zeta_loss_dec

    def forward(self, sentence_ids: torch.Tensor, tree_ids: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, float]:
        """
        Forward pass with CE regularization.

        Returns:
            Tuple of (outputs, total_zeta_loss)
        """
        batch_size = sentence_ids.size(0)
        max_len = tree_ids.size(1) if tree_ids is not None else 50

        # Encode with CE components
        encoder_outputs, hidden, zeta_loss_enc = self.encode(sentence_ids)

        # Decode
        outputs = []
        input_token = torch.full((batch_size,), self.vocab_tree.sos_idx,
                               device=sentence_ids.device)

        total_zeta_loss = zeta_loss_enc

        for t in range(max_len):
            logits, hidden, zeta_loss_dec = self.decode_step(
                input_token, hidden, encoder_outputs
            )
            outputs.append(logits.unsqueeze(1))
            total_zeta_loss += zeta_loss_dec

            # Teacher forcing
            if tree_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tree_ids[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        outputs = torch.cat(outputs, dim=1)

        # Apply final zeta regularization for grammatical consistency
        if self.use_zeta_reg and tree_ids is not None:
            final_zeta_loss = self.zeta_reg(outputs)[1]
            total_zeta_loss += final_zeta_loss

        return outputs, total_zeta_loss


class CEEnhancedPCFGBenchmark(PCFGBenchmark):
    """
    CE-Enhanced PCFG Benchmark with integrated CE components.
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

    def create_model(self) -> CEEnhancedPCFGModel:
        """Create a CE-enhanced PCFG model."""
        return CEEnhancedPCFGModel(
            self.vocab_sentence, self.vocab_tree,
            self.embed_dim, self.hidden_dim, self.num_layers,
            self.chi_feg, self.kappa,
            self.use_ce_attention, self.use_zeta_reg
        )

    def train_epoch(self, model: CEEnhancedPCFGModel, optimizer: optim.Optimizer,
                   criterion: nn.CrossEntropyLoss, device: str = 'cpu') -> Tuple[float, float]:
        """Train for one epoch with CE regularization."""
        model.train()
        total_loss = 0
        total_zeta_loss = 0

        for batch in self.train_loader:
            sentence_ids = batch['sentence_ids'].to(device)
            tree_ids = batch['tree_ids'].to(device)

            optimizer.zero_grad()

            # Forward pass with CE regularization
            outputs, zeta_loss = model(sentence_ids, tree_ids, teacher_forcing_ratio=0.5)

            # Standard cross-entropy loss
            ce_loss = criterion(outputs.view(-1, len(self.vocab_tree)),
                              tree_ids.view(-1))

            # Combined loss with zeta regularization
            total_batch_loss = ce_loss + self.zeta_reg_weight * zeta_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += ce_loss.item()
            total_zeta_loss += zeta_loss.item()

        avg_loss = total_loss / len(self.train_loader)
        avg_zeta_loss = total_zeta_loss / len(self.train_loader)

        return avg_loss, avg_zeta_loss

    def train_epoch_ce_timed(self, model: CEEnhancedPCFGModel, ce_timer,
                           criterion: nn.CrossEntropyLoss, device: str = 'cpu',
                           epoch: int = 0) -> Tuple[float, float]:
        """Train for one epoch with CE timing acceleration."""
        model.train()
        total_loss = 0
        total_zeta_loss = 0

        for batch in self.train_loader:
            sentence_ids = batch['sentence_ids'].to(device)
            tree_ids = batch['tree_ids'].to(device)

            # Forward pass with CE regularization
            outputs, zeta_loss = model(sentence_ids, tree_ids, teacher_forcing_ratio=0.5)

            # Standard cross-entropy loss
            ce_loss = criterion(outputs.view(-1, len(self.vocab_tree)),
                              tree_ids.view(-1))

            # Combined loss with zeta regularization
            total_batch_loss = ce_loss + self.zeta_reg_weight * zeta_loss

            # CE timing step
            timing_info = ce_timer.training_step(total_batch_loss, zeta_loss.item())

            total_loss += ce_loss.item()
            total_zeta_loss += zeta_loss.item()

            # CE early stopping check
            if timing_info['should_stop']:
                print(f"⚡ CE Early Stopping triggered mid-epoch (zeta awareness stabilized)")
                break

        avg_loss = total_loss / len(self.train_loader)
        avg_zeta_loss = total_zeta_loss / len(self.train_loader)

        return avg_loss, avg_zeta_loss

    def evaluate(self, model: CEEnhancedPCFGModel, device: str = 'cpu') -> Dict[str, float]:
        """Evaluate CE model on test set."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                sentence_ids = batch['sentence_ids'].to(device)

                # CE model returns (outputs, zeta_loss) tuple
                outputs, _ = model(sentence_ids, teacher_forcing_ratio=0.0)

                # Get predictions (exclude <sos> token)
                predictions = outputs.argmax(dim=-1)[:, 1:]  # Skip <sos>

                # Get targets (exclude <sos> token)
                targets = batch['tree_ids'][:, 1:].to(device)

                # Create mask for non-pad tokens
                mask = targets != self.vocab_tree.pad_idx

                correct += ((predictions == targets) & mask).sum().item()
                total += mask.sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}

    def train_model(self, model: CEEnhancedPCFGModel, num_epochs: int = 100,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """Train CE-enhanced model with CE timing acceleration."""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_tree.pad_idx)

        # Create CE timing accelerator
        ce_timer = create_ce_timed_trainer(model, optimizer)

        train_losses = []
        zeta_losses = []
        dev_accuracies = []
        timing_info_history = []

        for epoch in tqdm(range(num_epochs), desc="Training CE-PCFG"):
            # Train with CE timing awareness
            train_loss, zeta_loss = self.train_epoch_ce_timed(
                model, ce_timer, criterion, device, epoch
            )
            train_losses.append(train_loss)
            zeta_losses.append(zeta_loss)

            # Store timing information
            timing_info_history.append(ce_timer.timing_stats.copy())

            # Evaluate on dev set (more frequent with CE timing)
            if epoch % 5 == 0:
                dev_metrics = self.evaluate(model, 'dev', device)
                dev_accuracies.append(dev_metrics['accuracy'])
                current_lr = ce_timer.lr_scheduler.get_lr()
                phase_awareness = ce_timer.awareness_optimizer.phase_awareness
                print(f"Epoch {epoch}: CE Loss={train_loss:.4f}, "
                      f"Zeta Loss={zeta_loss:.4f}, Dev Acc={dev_metrics['accuracy']:.4f}, "
                      f"LR={current_lr:.6f}, Awareness={phase_awareness:.3f}")

            # CE timing early stopping
            if ce_timer.early_stopper.early_stop:
                print(f"🎯 CE Early Stopping at epoch {epoch} (zeta awareness stabilized)")
                break

        return {
            'train_losses': train_losses,
            'zeta_losses': zeta_losses,
            'dev_accuracies': dev_accuracies,
            'timing_stats': timing_info_history
        }


def run_ce_pcfg_experiment(num_epochs: int = 50, device: str = 'cpu',
                          chi_feg: float = 0.638, kappa: float = 0.35) -> Dict[str, float]:
    """Run CE-enhanced PCFG experiment."""
    print("🔬 Running CE-Enhanced PCFG Experiment...")
    print(f"Parameters: χ_FEG={chi_feg}, κ={kappa}")

    benchmark = CEEnhancedPCFGBenchmark(chi_feg=chi_feg, kappa=kappa)

    # Create and train CE-enhanced model
    model = benchmark.create_model()
    history = benchmark.train_model(model, num_epochs, device)

    # Final evaluation
    dev_metrics = benchmark.evaluate(model, 'dev', device)
    test_metrics = benchmark.evaluate(model, 'test', device)

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


if __name__ == "__main__":
    # Test CE-enhanced PCFG
    results = run_ce_pcfg_experiment(num_epochs=30)
    print(f"\nCE-PCFG Results: {results}")
