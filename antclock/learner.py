#!run.sh
"""
CE Learner: Integrated CE Architecture + CE Timing System

Complete CE intelligence system that combines:
- CE1 Corridor Embeddings (discrete geometry)
- CE2 Flow Operators (dynamical evolution)
- CE3 Witness Consistency (emergent invariants)
- CE Timing (awareness-aware training)

Ready to test on unsolved benchmarks: COGS, PCFG, CFQ, math, algorithmic tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json

from .architecture import CEArchitecture, create_ce_model
from benchmarks.timing import (
    KappaGuardianEarlyStopper,
    ChiFEGScheduler,
    AwarenessLoopOptimizer,
    PhaseLockedTrainer
)


@dataclass
class CELearningConfig:
    """Configuration for CE learning system."""

    # Model architecture
    vocab_size: int = 10000
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_classes: int = 2
    max_seq_len: int = 512

    # CE timing parameters
    kappa_guardian: float = 0.35
    chi_feg: float = 0.638
    awareness_patience: int = 5

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 100
    lambda_regularization: float = 0.1

    # CE mode settings
    use_ce_architecture: bool = True
    use_ce_timing: bool = True

    # Evaluation
    eval_every: int = 10
    save_every: int = 50


class CETrainerMetrics:
    """Tracks CE-specific training metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.epoch_losses = []
        self.ce_regularization_losses = []
        self.topological_consistencies = []
        self.mirror_consistencies = []
        self.bifurcation_regularizations = []
        self.learning_rates = []
        self.zeta_losses = []
        self.training_times = []
        self.convergence_epochs = None

    def update(self, loss: float, ce_reg_loss: float, topo_consist: float,
               mirror_consist: float, bifurc_reg: float, lr: float, zeta_loss: float):
        """Update metrics for current training step."""
        self.epoch_losses.append(loss)
        self.ce_regularization_losses.append(ce_reg_loss)
        self.topological_consistencies.append(topo_consist)
        self.mirror_consistencies.append(mirror_consist)
        self.bifurcation_regularizations.append(bifurc_reg)
        self.learning_rates.append(lr)
        self.zeta_losses.append(zeta_loss)

    def finalize_training(self, total_time: float, converged_at_epoch: Optional[int] = None):
        """Finalize training metrics."""
        self.training_times.append(total_time)
        self.convergence_epochs = converged_at_epoch

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'epoch_losses': self.epoch_losses,
            'ce_regularization_losses': self.ce_regularization_losses,
            'topological_consistencies': self.topological_consistencies,
            'mirror_consistencies': self.mirror_consistencies,
            'bifurcation_regularizations': self.bifurcation_regularizations,
            'learning_rates': self.learning_rates,
            'zeta_losses': self.zeta_losses,
            'training_times': self.training_times,
            'convergence_epochs': self.convergence_epochs
        }


class CELearner:
    """
    Complete CE Learning System

    Integrates CE architecture (CE1/CE2/CE3) with CE timing mechanisms
    for intelligent, geometry-aware learning.
    """

    def __init__(self, config: CELearningConfig):
        self.config = config

        # Initialize CE architecture
        self.model = create_ce_model(
            vocab_size=config.vocab_size,
            num_classes=config.num_classes,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim
        )

        # Toggle CE mode
        self.model.toggle_ce_mode(config.use_ce_architecture)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Initialize CE timing system (if enabled)
        if config.use_ce_timing:
            self.early_stopper = KappaGuardianEarlyStopper(
                kappa=config.kappa_guardian,
                patience=config.awareness_patience
            )
            self.lr_scheduler = ChiFEGScheduler(
                self.optimizer,
                chi_feg=config.chi_feg,
                base_lr=config.learning_rate
            )
            self.awareness_optimizer = AwarenessLoopOptimizer(
                self.optimizer,
                awareness_threshold=config.kappa_guardian
            )
        else:
            self.early_stopper = None
            self.lr_scheduler = None
            self.awareness_optimizer = None

        # Metrics tracking
        self.metrics = CETrainerMetrics()

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')

    def compute_zeta_loss(self, outputs: Dict[str, torch.Tensor]) -> float:
        """
        Compute zeta loss for CE timing awareness.

        This is a proxy for the model's "awareness" of geometric structure.
        """
        # Use CE regularization as zeta loss proxy
        if 'regularization_losses' in outputs:
            zeta_loss = outputs['regularization_losses']['total_regularization'].item()
        else:
            # Fallback: use task loss variance as proxy
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
            zeta_loss = entropy

        return zeta_loss

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with CE timing integration.

        Args:
            batch: Tuple of (input_ids, labels)

        Returns:
            Dictionary of step metrics
        """
        input_ids, labels = batch

        # Forward pass
        self.model.train()
        outputs = self.model(input_ids)

        # Compute CE loss (task + regularization)
        loss = self.model.compute_ce_loss(
            outputs,
            labels,
            lambda_reg=self.config.lambda_regularization
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Apply CE timing-aware optimization
        if self.config.use_ce_timing and self.awareness_optimizer is not None:
            zeta_loss = self.compute_zeta_loss(outputs)
            self.awareness_optimizer.step(zeta_loss)
        else:
            self.optimizer.step()

        # Update CE timing systems
        if self.config.use_ce_timing:
            zeta_loss = self.compute_zeta_loss(outputs)

            # Update learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(zeta_loss)

            # Check early stopping
            should_stop = False
            if self.early_stopper is not None:
                should_stop = self.early_stopper(zeta_loss)

        # Collect metrics
        current_lr = self.optimizer.param_groups[0]['lr']
        zeta_loss = self.compute_zeta_loss(outputs)

        if 'regularization_losses' in outputs:
            reg_losses = outputs['regularization_losses']
            topo_consist = reg_losses['topological_consistency'].item()
            mirror_consist = reg_losses['mirror_consistency'].item()
            bifurc_reg = reg_losses['bifurcation_regularization'].item()
            ce_reg_loss = reg_losses['total_regularization'].item()
        else:
            topo_consist = mirror_consist = bifurc_reg = ce_reg_loss = 0.0

        # Update metrics
        self.metrics.update(
            loss=loss.item(),
            ce_reg_loss=ce_reg_loss,
            topo_consist=topo_consist,
            mirror_consist=mirror_consist,
            bifurc_reg=bifurc_reg,
            lr=current_lr,
            zeta_loss=zeta_loss
        )

        return {
            'loss': loss.item(),
            'ce_reg_loss': ce_reg_loss,
            'learning_rate': current_lr,
            'zeta_loss': zeta_loss,
            'early_stop': should_stop if self.config.use_ce_timing else False
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                outputs = self.model(input_ids)

                # Task loss only (no regularization for validation)
                logits = outputs['logits']
                loss = nn.functional.cross_entropy(logits, labels)

                # Accuracy
                preds = torch.argmax(logits, dim=-1)
                correct = (preds == labels).sum().item()

                total_loss += loss.item() * len(labels)
                total_correct += correct
                total_samples += len(labels)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Full training loop with CE timing integration.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            save_path: Optional path to save model and metrics

        Returns:
            Training results dictionary
        """
        start_time = time.time()
        best_val_accuracy = 0.0

        print(f"Starting CE training with config:")
        print(f"  CE Architecture: {self.config.use_ce_architecture}")
        print(f"  CE Timing: {self.config.use_ce_timing}")
        print(f"  Max epochs: {self.config.max_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print()

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training epoch
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                step_metrics = self.train_step(batch)
                epoch_loss += step_metrics['loss']
                num_batches += 1

                # Check early stopping
                if step_metrics.get('early_stop', False):
                    print(f"CE Kappa Guardian early stopping at epoch {epoch}")
                    break

            avg_epoch_loss = epoch_loss / num_batches
            epoch_time = time.time() - epoch_start_time

            # Validation
            val_results = {}
            if val_loader is not None and (epoch + 1) % self.config.eval_every == 0:
                val_results = self.validate(val_loader)
                val_accuracy = val_results.get('val_accuracy', 0.0)

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    if save_path:
                        self.save_model(save_path, suffix="_best")

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = f"Epoch {epoch+1}/{self.config.max_epochs} | "
                log_msg += f"Loss: {avg_epoch_loss:.4f} | "
                log_msg += f"Time: {epoch_time:.2f}s"

                if val_results:
                    log_msg += f" | Val Loss: {val_results['val_loss']:.4f} | "
                    log_msg += f"Val Acc: {val_results['val_accuracy']:.4f}"

                if self.config.use_ce_timing and self.lr_scheduler:
                    current_lr = self.lr_scheduler.get_lr()
                    log_msg += f" | LR: {current_lr:.6f}"

                print(log_msg)

            # Check early stopping
            if self.config.use_ce_timing and self.early_stopper and self.early_stopper.early_stop:
                break

            # Save checkpoint
            if save_path and (epoch + 1) % self.config.save_every == 0:
                self.save_model(save_path, suffix=f"_epoch_{epoch+1}")

        # Finalize training
        total_time = time.time() - start_time
        self.metrics.finalize_training(total_time, self.current_epoch)

        # Save final model and metrics
        if save_path:
            self.save_model(save_path, suffix="_final")
            self.save_metrics(save_path)

        # Training results
        results = {
            'total_time': total_time,
            'final_epoch': self.current_epoch,
            'best_val_accuracy': best_val_accuracy,
            'final_loss': self.metrics.epoch_losses[-1] if self.metrics.epoch_losses else float('inf'),
            'config': self.config.__dict__,
            'metrics': self.metrics.to_dict()
        }

        print("\nCE Training completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final epoch: {self.current_epoch}")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")

        return results

    def save_model(self, path: str, suffix: str = ""):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'epoch': self.current_epoch,
            'metrics': self.metrics.to_dict()
        }, f"{path}{suffix}.pt")

    def save_metrics(self, path: str):
        """Save training metrics."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{path}_metrics.json", 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> 'CELearner':
        """Load CE learner from checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        config = CELearningConfig(**checkpoint['config'])
        learner = cls(config)

        learner.model.load_state_dict(checkpoint['model_state_dict'])
        learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        learner.current_epoch = checkpoint['epoch']
        learner.metrics = CETrainerMetrics()
        learner.metrics.__dict__.update(checkpoint['metrics'])

        return learner


def create_ce_learner_for_task(task_name: str, vocab_size: int = 10000,
                              num_classes: int = 2) -> CELearner:
    """
    Factory function to create CE learner configured for specific tasks.

    Args:
        task_name: Name of the task ('cogs', 'pcfg', 'cfq', 'math', 'scan')
        vocab_size: Size of vocabulary
        num_classes: Number of output classes

    Returns:
        Configured CE learner
    """
    # Task-specific configurations
    configs = {
        'cogs': CELearningConfig(
            vocab_size=vocab_size, num_classes=num_classes,
            embedding_dim=256, hidden_dim=256, max_seq_len=128,
            use_ce_architecture=True, use_ce_timing=True,
            lambda_regularization=0.2, kappa_guardian=0.3
        ),
        'pcfg': CELearningConfig(
            vocab_size=vocab_size, num_classes=num_classes,
            embedding_dim=128, hidden_dim=128, max_seq_len=64,
            use_ce_architecture=True, use_ce_timing=True,
            lambda_regularization=0.15, chi_feg=0.638
        ),
        'cfq': CELearningConfig(
            vocab_size=vocab_size, num_classes=num_classes,
            embedding_dim=512, hidden_dim=512, max_seq_len=256,
            use_ce_architecture=True, use_ce_timing=True,
            lambda_regularization=0.1, kappa_guardian=0.25
        ),
        'math': CELearningConfig(
            vocab_size=vocab_size, num_classes=num_classes,
            embedding_dim=128, hidden_dim=128, max_seq_len=128,
            use_ce_architecture=True, use_ce_timing=True,
            lambda_regularization=0.1, chi_feg=0.638
        ),
        'scan': CELearningConfig(
            vocab_size=vocab_size, num_classes=num_classes,
            embedding_dim=64, hidden_dim=64, max_seq_len=32,
            use_ce_architecture=True, use_ce_timing=True,
            lambda_regularization=0.05, kappa_guardian=0.35
        )
    }

    config = configs.get(task_name, configs['cogs'])
    return CELearner(config)


# Example usage and testing
if __name__ == "__main__":
    print("Testing CE Learner integration...")

    # Create synthetic dataset
    class SyntheticDataset(Dataset):
        def __init__(self, size=1000, seq_len=32, vocab_size=100, num_classes=10):
            self.data = torch.randint(0, vocab_size, (size, seq_len))
            self.labels = torch.randint(0, num_classes, (size,))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # Create CE learner for testing
    config = CELearningConfig(
        vocab_size=100, num_classes=10, embedding_dim=64, hidden_dim=64,
        max_seq_len=32, batch_size=16, max_epochs=5, eval_every=1
    )
    learner = CELearner(config)

    # Create data loaders
    train_dataset = SyntheticDataset(size=200, seq_len=32, vocab_size=100, num_classes=10)
    val_dataset = SyntheticDataset(size=50, seq_len=32, vocab_size=100, num_classes=10)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    print("Starting CE training test...")
    results = learner.train(train_loader, val_loader)

    print("\nCE Learner integration test completed!")
    print(f"Results: {results['best_val_accuracy']:.4f} validation accuracy")
    print("Ready to test on unsolved benchmarks ✨")
