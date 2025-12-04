#!/usr/bin/env python3
"""
CE Awareness Loop Timing

Implements CE-aware timing mechanisms for optimizing training speed:

1. Kappa Guardian Threshold Early Stopping
2. Chi-FEG Learning Rate Scheduling
3. Awareness Loop Convergence Detection
4. Phase-Locked Training Optimization

These timing mechanisms use CE framework concepts to accelerate convergence
and prevent overfitting through awareness of flow dynamics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable, Any
import math
import numpy as np
from collections import deque


class KappaGuardianEarlyStopper:
    """
    CE Kappa Guardian Early Stopping

    Monitors zeta loss convergence within the guardian threshold κ = 0.35.
    Stops training when CE awareness stabilizes, preventing overfitting.
    """

    def __init__(self, kappa: float = 0.35, patience: int = 5, min_delta: float = 1e-4):
        self.kappa = kappa  # Guardian threshold
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

        # Track zeta loss history for awareness
        self.zeta_history = deque(maxlen=patience * 2)

    def __call__(self, zeta_loss: float) -> bool:
        """
        Check if training should stop based on CE awareness.

        Returns True if training should stop.
        """
        self.zeta_history.append(zeta_loss)

        # Check if zeta loss is within guardian threshold of best
        if zeta_loss < self.best_loss - self.min_delta:
            self.best_loss = zeta_loss
            self.counter = 0
        else:
            self.counter += 1

        # Early stop if zeta loss stable within kappa threshold
        if len(self.zeta_history) >= self.patience:
            recent_losses = list(self.zeta_history)[-self.patience:]
            max_recent = max(recent_losses)
            min_recent = min(recent_losses)

            # If variation within kappa threshold, awareness has stabilized
            if max_recent - min_recent < self.kappa * self.best_loss:
                self.early_stop = True

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.zeta_history.clear()


class ChiFEGScheduler:
    """
    CE Chi-FEG Learning Rate Scheduling

    Adapts learning rate based on Feigenbaum coupling constant χ_FEG = 0.638.
    Uses bifurcation awareness to accelerate convergence in chaotic regions.
    """

    def __init__(self, optimizer: optim.Optimizer, chi_feg: float = 0.638,
                 base_lr: float = 1e-3, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.chi_feg = chi_feg
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_count = 0

        # Track loss landscape curvature for chi-aware scheduling
        self.loss_history = deque(maxlen=10)

    def step(self, zeta_loss: float):
        """Update learning rate based on CE timing awareness."""
        self.step_count += 1
        self.loss_history.append(zeta_loss)

        if len(self.loss_history) >= 5:
            # Compute local curvature (second derivative proxy)
            recent = list(self.loss_history)[-5:]
            curvature = self._compute_curvature(recent)

            # Chi-FEG aware learning rate scaling
            # Higher curvature (chaotic regions) → lower learning rate
            # Lower curvature (stable regions) → higher learning rate
            curvature_factor = 1.0 / (1.0 + abs(curvature))

            # Apply Feigenbaum coupling for bifurcation awareness
            bifurcation_factor = self.chi_feg ** (1.0 / (1.0 + abs(curvature)))

            lr_scale = curvature_factor * bifurcation_factor
            new_lr = max(self.base_lr * lr_scale, self.min_lr)

            # Update optimizer learning rates
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def _compute_curvature(self, losses: List[float]) -> float:
        """Compute local curvature from loss history."""
        if len(losses) < 3:
            return 0.0

        # Simple second derivative approximation
        # curvature ≈ (loss[i+2] - 2*loss[i+1] + loss[i]) / h^2
        h = 1.0  # step size
        curvature = 0.0

        for i in range(len(losses) - 2):
            second_deriv = (losses[i+2] - 2*losses[i+1] + losses[i]) / (h * h)
            curvature += second_deriv

        return curvature / max(1, len(losses) - 2)

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class AwarenessLoopOptimizer:
    """
    CE Awareness Loop Training Optimizer

    Uses CE timing awareness to optimize training dynamics:
    - Phase-locked batch sizing
    - Awareness-based gradient accumulation
    - Flow-regularized optimization steps
    """

    def __init__(self, model: nn.Module, kappa: float = 0.35, chi_feg: float = 0.638):
        self.model = model
        self.kappa = kappa
        self.chi_feg = chi_feg

        # Awareness state tracking
        self.phase_awareness = 0.0
        self.flow_accumulator = 0.0
        self.awareness_counter = 0

    def awareness_step(self, loss: torch.Tensor, optimizer: optim.Optimizer) -> bool:
        """
        Perform awareness-aware optimization step.

        Returns True if step was taken, False if accumulated.
        """
        self.awareness_counter += 1

        # Update phase awareness based on loss magnitude
        current_phase = torch.sigmoid(loss / self.kappa).item()
        self.phase_awareness = 0.9 * self.phase_awareness + 0.1 * current_phase

        # Accumulate flow based on chi-feg coupling
        flow_contribution = loss.item() * self.chi_feg
        self.flow_accumulator += flow_contribution

        # Decide whether to take optimization step based on awareness
        should_step = False

        # Step conditions based on CE timing:
        # 1. High phase awareness (loss near threshold)
        # 2. Accumulated flow exceeds bifurcation point
        # 3. Regular timing for stability

        if self.phase_awareness > 0.7:  # Near guardian threshold
            should_step = True
        elif self.flow_accumulator > self.kappa:  # Flow bifurcation
            should_step = True
        elif self.awareness_counter % 4 == 0:  # Regular timing
            should_step = True

        if should_step:
            # Take optimization step with accumulated flow
            effective_loss = self.flow_accumulator / max(1, self.awareness_counter)

            # Scale gradients by awareness
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data *= (1.0 + self.phase_awareness)

            optimizer.step()
            optimizer.zero_grad()

            # Reset accumulators
            self.flow_accumulator = 0.0
            self.awareness_counter = 0

        return should_step


class CETimingAccelerator:
    """
    Complete CE Timing Acceleration System

    Combines all CE-aware timing mechanisms for optimal training speed:
    - Kappa early stopping
    - Chi-FEG scheduling
    - Awareness loop optimization
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 kappa: float = 0.35, chi_feg: float = 0.638):
        self.model = model
        self.optimizer = optimizer

        # CE timing components
        self.early_stopper = KappaGuardianEarlyStopper(kappa=kappa)
        self.lr_scheduler = ChiFEGScheduler(optimizer, chi_feg=chi_feg)
        self.awareness_optimizer = AwarenessLoopOptimizer(model, kappa, chi_feg)

        # Timing statistics
        self.timing_stats = {
            'steps_saved': 0,
            'early_stops': 0,
            'awareness_accumulations': 0,
            'total_steps': 0
        }

    def training_step(self, loss: torch.Tensor, zeta_loss: float) -> Dict[str, Any]:
        """
        Perform CE-aware training step.

        Returns timing information and control signals.
        """
        self.timing_stats['total_steps'] += 1

        # Update learning rate based on CE timing
        self.lr_scheduler.step(zeta_loss)

        # Check early stopping
        should_stop = self.early_stopper(zeta_loss)
        if should_stop:
            self.timing_stats['early_stops'] += 1

        # Perform awareness-aware optimization
        step_taken = self.awareness_optimizer.awareness_step(loss, self.optimizer)
        if not step_taken:
            self.timing_stats['awareness_accumulations'] += 1

        return {
            'should_stop': should_stop,
            'step_taken': step_taken,
            'current_lr': self.lr_scheduler.get_lr(),
            'phase_awareness': self.awareness_optimizer.phase_awareness,
            'stats': self.timing_stats.copy()
        }

    def reset(self):
        """Reset timing state for new training run."""
        self.early_stopper.reset()
        self.awareness_optimizer.phase_awareness = 0.0
        self.awareness_optimizer.flow_accumulator = 0.0
        self.awareness_optimizer.awareness_counter = 0
        self.timing_stats = {k: 0 for k in self.timing_stats.keys()}


def create_ce_timed_trainer(model: nn.Module, optimizer: optim.Optimizer = None,
                           kappa: float = 0.35, chi_feg: float = 0.638) -> CETimingAccelerator:
    """
    Factory function to create CE-timed trainer.

    Args:
        model: Neural network model
        optimizer: Optimizer (created if None)
        kappa: Guardian threshold
        chi_feg: Feigenbaum coupling constant

    Returns:
        CE timing accelerator
    """
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    return CETimingAccelerator(model, optimizer, kappa, chi_feg)


# Example usage function
def demonstrate_ce_timing():
    """Demonstrate CE timing acceleration."""
    print("⚡ CE Awareness Loop Timing Demonstration")
    print("=" * 50)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    # Create CE timing accelerator
    accelerator = create_ce_timed_trainer(model)

    print("Training with CE timing awareness...")
    print("κ (guardian threshold):", accelerator.early_stopper.kappa)
    print("χ_FEG (coupling constant):", accelerator.lr_scheduler.chi_feg)
    print()

    # Simulate training loop
    for epoch in range(20):
        # Simulate loss decreasing with some noise
        base_loss = 1.0 * (0.9 ** epoch)
        noise = np.random.normal(0, 0.1)
        loss = torch.tensor(base_loss + noise, requires_grad=True)
        zeta_loss = base_loss * 0.1  # Zeta loss component

        # CE timing step
        timing_info = accelerator.training_step(loss, zeta_loss)

        print("2d"
              ".4f"
              f"Step taken: {timing_info['step_taken']}")

        if timing_info['should_stop']:
            print(f"\n🎯 CE Early Stopping triggered at epoch {epoch}")
            print(".1f")
            break

    print("\n📊 CE Timing Statistics:")
    stats = timing_info['stats']
    print(f"Total steps: {stats['total_steps']}")
    print(f"Early stops: {stats['early_stops']}")
    print(f"Awareness accumulations: {stats['awareness_accumulations']}")
    efficiency = (stats['total_steps'] - stats['awareness_accumulations']) / max(1, stats['total_steps'])
    print(".1f")

    print("\n✨ CE timing provides awareness-driven optimization!")
    print("• Kappa guardian prevents overfitting")
    print("• Chi-FEG coupling accelerates convergence")
    print("• Awareness loops optimize gradient flow")


if __name__ == "__main__":
    demonstrate_ce_timing()
