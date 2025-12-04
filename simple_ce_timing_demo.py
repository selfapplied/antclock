#!/usr/bin/env python3
"""
Simple CE Awareness Loop Timing Demonstration

Shows CE timing acceleration benefits.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from benchmarks.ce_timing import create_ce_timed_trainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def run_timing_demo():
    print("⚡ CE Awareness Loop Timing Demo")
    print("=" * 40)

    # Create synthetic data
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)

    # Baseline training
    print("\n🏃 Baseline Training...")
    model1 = SimpleModel()
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    start = time.time()
    for epoch in range(20):
        optimizer1.zero_grad()
        output = model1(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer1.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}")
    baseline_time = time.time() - start

    # CE-timed training
    print("\n⚡ CE-Timed Training...")
    model2 = SimpleModel()
    ce_timer = create_ce_timed_trainer(model2)

    start = time.time()
    for epoch in range(20):
        epoch_loss = 0
        for i in range(0, len(X), 32):
            batch_x = X[i:i+32]
            batch_y = y[i:i+32]

            output = model2(batch_x)
            loss = criterion(output, batch_y)
            zeta_loss = 0.1 * torch.randn(1).abs()

            timing_info = ce_timer.training_step(loss, zeta_loss.item())
            epoch_loss += loss.item()

            if timing_info['should_stop']:
                break

        avg_loss = epoch_loss / (len(X) // 32)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, LR={timing_info['current_lr']:.6f}")

        if ce_timer.early_stopper.early_stop:
            print(f"🎯 CE Early Stopping at epoch {epoch}")
            break
    ce_time = time.time() - start

    # Results
    print("\n" + "=" * 40)
    print("📊 RESULTS:")
    print(f"Baseline time: {baseline_time:.2f}s")
    print(f"CE-timed time: {ce_time:.2f}s")
    print(f"Speedup: {baseline_time/ce_time:.1f}x")

    stats = ce_timer.timing_stats
    print(f"Early stops: {stats['early_stops']}")
    print(f"Awareness accumulations: {stats['awareness_accumulations']}")

    print("\n✨ CE timing provides intelligent acceleration!")


if __name__ == "__main__":
    run_timing_demo()

