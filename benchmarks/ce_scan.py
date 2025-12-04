"""
CE-Enhanced SCAN Model

Integrates CE framework components into SCAN seq2seq architecture:
- Mirror operators for symmetry-aware attention
- Curvature coupling for temporal dynamics
- Zeta regularization for functional equation constraints
- Guardian thresholds for phase separation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

try:
    from .scan import SCANBenchmark, SCANDataset, SCANVocab, SCANCommand
    from .ce_modules import (
        MirrorOperator, CurvatureCouplingLayer, GuardianThreshold,
        ZetaRegularization, CEAttention, CEEnhancedLSTM
    )
    from .ce_timing import create_ce_timed_trainer
except ImportError:
    from scan import SCANBenchmark, SCANDataset, SCANVocab, SCANCommand
    from ce_modules import (
        MirrorOperator, CurvatureCouplingLayer, GuardianThreshold,
        ZetaRegularization, CEAttention, CEEnhancedLSTM
    )
    from ce_timing import create_ce_timed_trainer


class CEEnhancedSCANModel(nn.Module):
    """
    CE-Enhanced SCAN Model with integrated CE components.

    Incorporates:
    - Mirror-symmetric attention
    - Curvature-coupled LSTM layers
    - Zeta regularization
    - Guardian threshold activations
    """

    def __init__(self, vocab_command: SCANVocab, vocab_action: SCANVocab,
                 embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2,
                 chi_feg: float = 0.638, kappa: float = 0.35,
                 use_ce_attention: bool = True, use_zeta_reg: bool = True):
        super().__init__()

        self.vocab_command = vocab_command
        self.vocab_action = vocab_action
        self.chi_feg = chi_feg
        self.kappa = kappa
        self.use_ce_attention = use_ce_attention
        self.use_zeta_reg = use_zeta_reg

        # Encoder with CE components
        self.encoder_embed = nn.Embedding(len(vocab_command), embed_dim)

        # CE-enhanced encoder LSTM
        self.encoder_lstm = CEEnhancedLSTM(
            embed_dim, hidden_dim, chi_feg, kappa, use_zeta_reg
        )

        # Optional CE attention mechanism
        if use_ce_attention:
            self.encoder_attention = CEAttention(hidden_dim)
        else:
            self.encoder_attention = None

        # Decoder
        self.decoder_embed = nn.Embedding(len(vocab_action), embed_dim)

        # CE-enhanced decoder LSTM
        self.decoder_lstm = CEEnhancedLSTM(
            embed_dim, hidden_dim, chi_feg, kappa, use_zeta_reg
        )

        # Output projection with guardian threshold
        self.output_proj = nn.Linear(hidden_dim, len(vocab_action))
        self.guardian_activation = GuardianThreshold(kappa)

        # Zeta regularization for the complete model
        if use_zeta_reg:
            self.zeta_reg = ZetaRegularization(hidden_dim)

    def encode(self, command_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Encode command sequence with CE enhancements."""
        embeds = self.encoder_embed(command_ids)

        # Apply CE-enhanced LSTM encoding
        encoder_outputs, (hidden, cell), zeta_loss_enc = self.encoder_lstm(embeds)

        # Apply CE attention if enabled
        if self.encoder_attention is not None:
            # Self-attention on encoder outputs
            attended_outputs = self.encoder_attention(
                encoder_outputs, encoder_outputs, encoder_outputs
            )
            encoder_outputs = encoder_outputs + attended_outputs  # Residual connection

        return encoder_outputs, (hidden, cell), zeta_loss_enc

    def decode_step(self, input_token: torch.Tensor, hidden: Tuple,
                   encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Single CE-enhanced decoder step."""
        embed = self.decoder_embed(input_token.unsqueeze(1))

        # CE-enhanced LSTM decoding
        decoder_output, hidden, zeta_loss_dec = self.decoder_lstm(embed, hidden)

        # Apply attention to encoder outputs if available
        if self.encoder_attention is not None and encoder_outputs is not None:
            # Cross-attention between decoder and encoder
            attended = self.encoder_attention(
                decoder_output, encoder_outputs, encoder_outputs
            )
            decoder_output = decoder_output + attended

        # Project to vocabulary with guardian threshold
        logits = self.output_proj(decoder_output.squeeze(1))
        logits = self.guardian_activation(logits)

        return logits, hidden, zeta_loss_dec

    def forward(self, command_ids: torch.Tensor, action_ids: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, float]:
        """
        Forward pass with CE regularization.

        Returns:
            Tuple of (outputs, total_zeta_loss)
        """
        batch_size = command_ids.size(0)
        max_len = action_ids.size(1) if action_ids is not None else 50

        # Encode with CE components
        encoder_outputs, (hidden, cell), zeta_loss_enc = self.encode(command_ids)

        # Decode
        outputs = []
        input_token = torch.full((batch_size,), self.vocab_action.sos_idx,
                               device=command_ids.device)

        total_zeta_loss = zeta_loss_enc

        for t in range(max_len):
            logits, (hidden, cell), zeta_loss_dec = self.decode_step(
                input_token, (hidden, cell), encoder_outputs
            )
            outputs.append(logits.unsqueeze(1))
            total_zeta_loss += zeta_loss_dec

            # Teacher forcing
            if action_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = action_ids[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        outputs = torch.cat(outputs, dim=1)

        # Apply final zeta regularization to hidden states if enabled
        if self.use_zeta_reg and action_ids is not None:
            # Use final hidden state for zeta regularization
            final_zeta_loss = self.zeta_reg(hidden.unsqueeze(1))[1]
            total_zeta_loss += final_zeta_loss

        return outputs, total_zeta_loss


class CEEnhancedSCANBenchmark(SCANBenchmark):
    """
    CE-Enhanced SCAN Benchmark with integrated CE components.
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2,
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

    def create_model(self) -> CEEnhancedSCANModel:
        """Create a CE-enhanced SCAN model."""
        return CEEnhancedSCANModel(
            self.vocab_command, self.vocab_action,
            self.embed_dim, self.hidden_dim, self.num_layers,
            self.chi_feg, self.kappa,
            self.use_ce_attention, self.use_zeta_reg
        )

    def train_epoch(self, model: CEEnhancedSCANModel, optimizer: optim.Optimizer,
                   criterion: nn.CrossEntropyLoss, device: str = 'cpu') -> Tuple[float, float]:
        """Train for one epoch with CE regularization."""
        model.train()
        total_loss = 0
        total_zeta_loss = 0

        for batch in self.train_loader:
            command_ids = batch['command_ids'].to(device)
            action_ids = batch['action_ids'].to(device)

            optimizer.zero_grad()

            # Forward pass with CE regularization
            outputs, zeta_loss = model(command_ids, action_ids, teacher_forcing_ratio=0.5)

            # Standard cross-entropy loss
            ce_loss = criterion(outputs.view(-1, len(self.vocab_action)),
                              action_ids.view(-1))

            # Combined loss with zeta regularization
            total_batch_loss = ce_loss + self.zeta_reg_weight * zeta_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += ce_loss.item()
            total_zeta_loss += zeta_loss.item()

        avg_loss = total_loss / len(self.train_loader)
        avg_zeta_loss = total_zeta_loss / len(self.train_loader)

        return avg_loss, avg_zeta_loss

    def evaluate(self, model: CEEnhancedSCANModel, device: str = 'cpu') -> Dict[str, float]:
        """Evaluate CE model on test set."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                command_ids = batch['command_ids'].to(device)

                # CE model returns (outputs, zeta_loss) tuple
                outputs, _ = model(command_ids, teacher_forcing_ratio=0.0)

                # Get predictions (exclude <sos> token)
                predictions = outputs.argmax(dim=-1)[:, 1:]  # Skip <sos>

                # Get targets (exclude <sos> token)
                targets = batch['action_ids'][:, 1:].to(device)

                # Create mask for non-pad tokens
                mask = targets != self.vocab_action.pad_idx

                correct += ((predictions == targets) & mask).sum().item()
                total += mask.sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}

    def train_epoch_ce_timed(self, model: CEEnhancedSCANModel, ce_timer,
                           criterion: nn.CrossEntropyLoss, device: str = 'cpu',
                           epoch: int = 0) -> Tuple[float, float]:
        """Train for one epoch with CE timing acceleration."""
        model.train()
        total_loss = 0
        total_zeta_loss = 0
        steps_this_epoch = 0

        for batch in self.train_loader:
            command_ids = batch['command_ids'].to(device)
            action_ids = batch['action_ids'].to(device)

            # Forward pass with CE regularization
            outputs, zeta_loss = model(command_ids, action_ids, teacher_forcing_ratio=0.5)

            # Standard cross-entropy loss
            ce_loss = criterion(outputs.view(-1, len(self.vocab_action)),
                              action_ids.view(-1))

            # Combined loss with zeta regularization
            total_batch_loss = ce_loss + self.zeta_reg_weight * zeta_loss

            # CE timing-aware optimization
            timing_info = ce_timer.training_step(total_batch_loss, zeta_loss.item())

            # Only count as a step if CE timing actually performed optimization
            if timing_info['step_taken']:
                steps_this_epoch += 1

            total_loss += ce_loss.item()
            total_zeta_loss += zeta_loss.item()

            # CE early stopping check
            if timing_info['should_stop']:
                print(f"⚡ CE Early Stopping triggered mid-epoch (zeta awareness stabilized)")
                break

        avg_loss = total_loss / len(self.train_loader)
        avg_zeta_loss = total_zeta_loss / len(self.train_loader)

        return avg_loss, avg_zeta_loss

    def train_model(self, model: CEEnhancedSCANModel, num_epochs: int = 100,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """Train CE-enhanced model with CE timing acceleration."""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_action.pad_idx)

        # Create CE timing accelerator
        ce_timer = create_ce_timed_trainer(model, optimizer)

        train_losses = []
        zeta_losses = []
        test_accuracies = []
        timing_info = []

        for epoch in tqdm(range(num_epochs), desc="Training CE-SCAN"):
            # Train with CE timing awareness
            train_loss, zeta_loss = self.train_epoch_ce_timed(
                model, ce_timer, criterion, device, epoch
            )
            train_losses.append(train_loss)
            zeta_losses.append(zeta_loss)

            # Store timing information
            timing_info.append(ce_timer.timing_stats.copy())

            # Evaluate
            if epoch % 5 == 0:  # More frequent evaluation with CE timing
                metrics = self.evaluate(model, device)
                test_accuracies.append(metrics['accuracy'])
                current_lr = ce_timer.lr_scheduler.get_lr()
                phase_awareness = ce_timer.awareness_optimizer.phase_awareness
                print(f"Epoch {epoch}: CE Loss={train_loss:.4f}, "
                      f"Zeta Loss={zeta_loss:.4f}, Test Acc={metrics['accuracy']:.4f}, "
                      f"LR={current_lr:.6f}, Awareness={phase_awareness:.3f}")

            # CE timing early stopping
            if ce_timer.early_stopper.early_stop:
                print(f"🎯 CE Early Stopping at epoch {epoch} (zeta loss stabilized)")
                break

        return {
            'train_losses': train_losses,
            'zeta_losses': zeta_losses,
            'test_accuracies': test_accuracies
        }


def run_ce_scan_experiment(num_epochs: int = 100, device: str = 'cpu',
                          chi_feg: float = 0.638, kappa: float = 0.35) -> Dict[str, float]:
    """Run CE-enhanced SCAN experiment."""
    print("🔬 Running CE-Enhanced SCAN Experiment...")
    print(f"Parameters: χ_FEG={chi_feg}, κ={kappa}")

    benchmark = CEEnhancedSCANBenchmark(chi_feg=chi_feg, kappa=kappa)

    # Create and train CE-enhanced model
    model = benchmark.create_model()
    history = benchmark.train_model(model, num_epochs, device)

    # Final evaluation
    final_metrics = benchmark.evaluate(model, device)

    results = {
        'train_loss_final': history['train_losses'][-1],
        'zeta_loss_final': history['zeta_losses'][-1],
        'test_accuracy': final_metrics['accuracy'],
        'test_correct': final_metrics['correct'],
        'test_total': final_metrics['total'],
        'chi_feg': chi_feg,
        'kappa': kappa
    }

    print(f"Baseline Loss: {results['train_loss_final']:.4f}")
    print(f"Zeta Regularization Loss: {results['zeta_loss_final']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.1%} ({results['test_correct']}/{results['test_total']})")

    return results


def ablation_study_scan(device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """Perform ablation study on CE components for SCAN."""
    print("🔬 Performing SCAN Ablation Study...")

    configurations = [
        {'name': 'baseline', 'ce_attention': False, 'zeta_reg': False},
        {'name': 'ce_attention_only', 'ce_attention': True, 'zeta_reg': False},
        {'name': 'zeta_reg_only', 'ce_attention': False, 'zeta_reg': True},
        {'name': 'full_ce', 'ce_attention': True, 'zeta_reg': True},
    ]

    results = {}

    for config in configurations:
        print(f"\nTesting configuration: {config['name']}")

        benchmark = CEEnhancedSCANBenchmark(
            use_ce_attention=config['ce_attention'],
            use_zeta_reg=config['zeta_reg']
        )

        model = benchmark.create_model()
        history = benchmark.train_model(model, num_epochs=50, device=device)
        metrics = benchmark.evaluate(model, device)

        results[config['name']] = {
            'test_accuracy': metrics['accuracy'],
            'train_loss_final': history['train_losses'][-1],
            'zeta_loss_final': history['zeta_losses'][-1] if 'zeta_losses' in history else 0.0
        }

        print(f"  Accuracy: {metrics['accuracy']:.1%}")

    return results


if __name__ == "__main__":
    # Test CE-enhanced SCAN
    results = run_ce_scan_experiment(num_epochs=50)
    print(f"\nCE-SCAN Results: {results}")

    # Run ablation study
    ablation_results = ablation_study_scan()
    print(f"\nAblation Study Results: {ablation_results}")
