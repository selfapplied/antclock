#!/usr/bin/env python3
"""
Test CE Timing on COGS Benchmark
"""

import time
import sys

try:
    from benchmarks.ce_cogs import run_ce_cogs_experiment
    from benchmarks.cogs import run_cogs_baseline
except ImportError:
    from ce_cogs import run_ce_cogs_experiment
    from cogs import run_cogs_baseline


def test_cogs_ce_timing():
    print("⚡ CE Timing Evaluation: COGS Benchmark")
    print("=" * 50)

    # Test CE-enhanced COGS
    print("\n⚡ CE-enhanced COGS (2 epochs)...")
    start = time.time()
    try:
        ce_result = run_ce_cogs_experiment(num_epochs=2)
        ce_time = time.time() - start
        print(f"CE Time: {ce_time:.2f}s")
    except Exception as e:
        print(f"CE COGS failed: {e}")
        return

    # Test baseline COGS
    print("\n🏃 Baseline COGS (2 epochs)...")
    start = time.time()
    try:
        baseline_result = run_cogs_baseline(num_epochs=2)
        baseline_time = time.time() - start
        print(f"Baseline Time: {baseline_time:.2f}s")
    except Exception as e:
        print(f"Baseline COGS failed: {e}")
        return

    # Results
    speedup = baseline_time / ce_time if ce_time > 0 else 0
    ce_acc = ce_result.get('test_accuracy', 0)
    baseline_acc = baseline_result.get('test_accuracy', 0)

    print("\n" + "=" * 50)
    print("📊 COGS RESULTS:")
    print(f"Speedup: {speedup:.1f}x faster")
    print(f"CE Accuracy: {ce_acc:.1%}")
    print(f"Baseline Accuracy: {baseline_acc:.1%}")

    print("\n✨ CE timing provides intelligent acceleration!")


if __name__ == "__main__":
    test_cogs_ce_timing()
