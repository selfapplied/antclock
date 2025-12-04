#!/usr/bin/env python3
"""
CE Timing System: Awareness-aware training mechanisms

Implements the CE timing framework:
- Kappa Guardian: Early stopping based on zeta loss threshold
- Chi-FEG Scheduler: Learning rate scheduling based on golden ratio
- Awareness Loop Optimizer: Adaptive optimization based on geometric awareness
- Phase Locked Trainer: Synchronized training with CE phases

These mechanisms provide intelligent timing for CE-based learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class TimingConfig:
    """Configuration for CE timing mechanisms."""
    kappa_guardian: float = 0.35  # Zeta loss threshold
    chi_feg: float = 0.638  # Golden ratio learning rate factor
    awareness_patience: int = 5  # Patience for awareness monitoring
    base_lr: float = 1e-3
    min_lr: float = 1e-6


class KappaGuardianEarlyStopper:
    """
    Kappa Guardian: Early stopping mechanism based on zeta loss threshold.

    Monitors the zeta regularization loss and stops training when it exceeds
    the kappa threshold, indicating loss of geometric awareness.
    """

    def __init__(self, kappa: float = 0.35, patience: int = 5):
        self.kappa = kappa
        self.patience = patience
        self.reset()

    def reset(self):
        """Reset the guardian state."""
        self.violation_count = 0
        self.best_zeta = float('inf')
        self.early_stop = False
        self.epochs_since_improvement = 0

    def __call__(self, zeta_loss: float) -> bool:
        """
        Check if early stopping should be triggered.

        Args:
            zeta_loss: Current zeta regularization loss

        Returns:
            True if training should stop
        """
        if self.early_stop:
            return True

        # Update best zeta
        if zeta_loss < self.best_zeta:
            self.best_zeta = zeta_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        # Check kappa threshold violation
        if zeta_loss > self.kappa:
            self.violation_count += 1
        else:
            self.violation_count = max(0, self.violation_count - 1)

        # Check early stopping conditions
        if self.violation_count >= self.patience or self.epochs_since_improvement >= self.patience * 2:
            self.early_stop = True

        return self.early_stop

    def get_status(self) -> Dict[str, Any]:
        """Get current guardian status."""
        return {
            'early_stop': self.early_stop,
            'violation_count': self.violation_count,
            'best_zeta': self.best_zeta,
            'epochs_since_improvement': self.epochs_since_improvement
        }


class ChiFEGScheduler:
    """
    Chi-FEG Learning Rate Scheduler: Golden ratio-based learning rate scheduling.

    Adjusts learning rate based on the golden ratio (φ ≈ 0.618) and
    zeta loss awareness, providing intelligent learning rate adaptation.
    """

    def __init__(self, optimizer: optim.Optimizer, chi_feg: float = 0.638,
                 base_lr: float = 1e-3, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.chi_feg = chi_feg
        self.base_lr = base_lr
        self.min_lr = min_lr

        self.current_lr = base_lr
        self.step_count = 0

        # Golden ratio sequence for scheduling
        self.phi = (1 + math.sqrt(5)) / 2  # ≈ 1.618
        self.phi_conjugate = self.phi - 1  # ≈ 0.618

    def step(self, zeta_loss: Optional[float] = None):
        """
        Update learning rate based on zeta loss and golden ratio schedule.

        Args:
            zeta_loss: Current zeta regularization loss (optional)
        """
        self.step_count += 1

        if zeta_loss is not None:
            # Zeta-aware learning rate adjustment
            awareness_factor = min(1.0, zeta_loss / 0.5)  # Scale zeta loss to [0,1]
            lr_multiplier = self.chi_feg ** awareness_factor
        else:
            # Standard golden ratio decay
            lr_multiplier = self.phi_conjugate ** (self.step_count / 10)

        # Update learning rate
        new_lr = max(self.min_lr, self.base_lr * lr_multiplier)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.current_lr = new_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset(self):
        """Reset scheduler state."""
        self.current_lr = self.base_lr
        self.step_count = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr


class AwarenessLoopOptimizer:
    """
    Awareness Loop Optimizer: Adaptive optimization based on geometric awareness.

    Modifies gradients based on zeta loss awareness, providing CE-aware
    optimization that respects geometric constraints.
    """

    def __init__(self, optimizer: optim.Optimizer, awareness_threshold: float = 0.35):
        self.optimizer = optimizer
        self.awareness_threshold = awareness_threshold
        self.awareness_history = []

    def step(self, zeta_loss: float):
        """
        Take an optimization step with awareness modulation.

        Args:
            zeta_loss: Current zeta regularization loss
        """
        # Compute awareness level
        awareness_level = max(0.0, 1.0 - (zeta_loss / self.awareness_threshold))
        self.awareness_history.append(awareness_level)

        # Keep only recent history
        if len(self.awareness_history) > 10:
            self.awareness_history = self.awareness_history[-10:]

        # Compute awareness stability (low variance = stable awareness)
        if len(self.awareness_history) > 1:
            awareness_stability = 1.0 - torch.tensor(self.awareness_history).std().item()
            awareness_stability = max(0.0, min(1.0, awareness_stability))
        else:
            awareness_stability = 0.5

        # Modulate gradients based on awareness
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    # Scale gradients by awareness level
                    awareness_scale = 0.5 + 0.5 * awareness_level * awareness_stability
                    param.grad.data *= awareness_scale

        # Take optimizer step
        self.optimizer.step()

    def get_awareness_stats(self) -> Dict[str, float]:
        """Get awareness statistics."""
        if not self.awareness_history:
            return {'mean': 0.0, 'std': 0.0, 'current': 0.0}

        awareness_tensor = torch.tensor(self.awareness_history)
        return {
            'mean': awareness_tensor.mean().item(),
            'std': awareness_tensor.std().item(),
            'current': self.awareness_history[-1]
        }


class PhaseLockedTrainer:
    """
    Phase Locked Trainer: Synchronized training with CE phases.

    Coordinates training phases with CE geometric phases (n mod 4),
    providing phase-aware learning that respects discrete symmetries.
    """

    def __init__(self, optimizer: optim.Optimizer, lr_scheduler: Optional[Any] = None,
                 early_stopper: Optional[Any] = None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.early_stopper = early_stopper

        self.current_phase = 0
        self.phase_history = []

    def step(self, loss: float, zeta_loss: Optional[float] = None) -> Dict[str, Any]:
        """
        Take a training step with phase locking.

        Args:
            loss: Current task loss
            zeta_loss: Current zeta regularization loss

        Returns:
            Step information dictionary
        """
        # Update phase (cycle through mirror phases)
        self.current_phase = (self.current_phase + 1) % 4
        self.phase_history.append(self.current_phase)

        # Phase-aware learning rate modulation
        phase_factor = 1.0
        if self.current_phase == 3:  # Tangent singularity phase
            phase_factor = 0.7  # Reduce learning rate at boundaries
        elif self.current_phase == 0:  # Stable phase
            phase_factor = 1.2  # Increase learning rate in stable regions

        # Apply phase modulation
        for param_group in self.optimizer.param_groups:
            base_lr = param_group['base_lr'] if 'base_lr' in param_group else param_group['lr']
            param_group['lr'] = base_lr * phase_factor

        # Zero gradients
        self.optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        # Update schedulers
        step_info = {'phase': self.current_phase, 'phase_factor': phase_factor}

        if self.lr_scheduler is not None and zeta_loss is not None:
            self.lr_scheduler.step(zeta_loss)
            step_info['learning_rate'] = self.lr_scheduler.get_lr()

        if self.early_stopper is not None and zeta_loss is not None:
            should_stop = self.early_stopper(zeta_loss)
            step_info['early_stop'] = should_stop

        return step_info

    def get_phase_stats(self) -> Dict[str, Any]:
        """Get phase statistics."""
        if not self.phase_history:
            return {'current_phase': 0, 'phase_distribution': {}}

        phase_counts = {}
        for phase in range(4):
            phase_counts[phase] = self.phase_history.count(phase)

        return {
            'current_phase': self.current_phase,
            'phase_distribution': phase_counts,
            'total_steps': len(self.phase_history)
        }


# Convenience functions for creating CE timing systems
def create_ce_timing_system(optimizer: optim.Optimizer, config: TimingConfig) -> Dict[str, Any]:
    """
    Create complete CE timing system.

    Args:
        optimizer: PyTorch optimizer
        config: Timing configuration

    Returns:
        Dictionary with timing components
    """
    early_stopper = KappaGuardianEarlyStopper(
        kappa=config.kappa_guardian,
        patience=config.awareness_patience
    )

    lr_scheduler = ChiFEGScheduler(
        optimizer=optimizer,
        chi_feg=config.chi_feg,
        base_lr=config.base_lr,
        min_lr=config.min_lr
    )

    awareness_optimizer = AwarenessLoopOptimizer(
        optimizer=optimizer,
        awareness_threshold=config.kappa_guardian
    )

    phase_trainer = PhaseLockedTrainer(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        early_stopper=early_stopper
    )

    return {
        'early_stopper': early_stopper,
        'lr_scheduler': lr_scheduler,
        'awareness_optimizer': awareness_optimizer,
        'phase_trainer': phase_trainer,
        'config': config
    }


# Test the timing mechanisms
if __name__ == "__main__":
    print("Testing CE Timing Mechanisms...")

    # Create dummy optimizer
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create timing system
    config = TimingConfig()
    timing_system = create_ce_timing_system(optimizer, config)

    print("✅ CE Timing System created!")
    print(f"Kappa Guardian threshold: {config.kappa_guardian}")
    print(f"Chi-FEG ratio: {config.chi_feg}")

    # Test components
    early_stopper = timing_system['early_stopper']
    lr_scheduler = timing_system['lr_scheduler']

    # Simulate training steps
    for step in range(10):
        zeta_loss = 0.3 + 0.1 * torch.randn(1).item()  # Simulate zeta loss

        # Test early stopping
        should_stop = early_stopper(zeta_loss)

        # Test LR scheduling
        lr_scheduler.step(zeta_loss)

        print(f"Step {step}: Zeta={zeta_loss:.3f}, LR={lr_scheduler.get_lr():.6f}, Stop={should_stop}")

        if should_stop:
            print("Early stopping triggered!")
            break

    print("✅ CE Timing Mechanisms working!")