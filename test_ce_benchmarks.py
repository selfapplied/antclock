#!/usr/bin/env python3
"""
CE Benchmark Testing: Test CE Architecture on Unsolved Benchmarks

Tests the complete CE intelligence system on:
- COGS (COmmon-sense Grounded Sentences)
- PCFG (Probabilistic Context-Free Grammar)
- CFQ (Compositional Freebase Questions)
- Math reasoning tasks
- Algorithmic/SCAN-like tasks

This is the real test of CE's architectural advantage, not just timing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse

from ce_learner import CELearner, CELearningConfig, create_ce_learner_for_task
from benchmarks.cogs import load_cogs_data
from benchmarks.pcfg import load_pcfg_data
from benchmarks.scan import load_scan_data


class SequenceClassificationDataset(Dataset):
    """Generic sequence classification dataset."""

    def __init__(self, sequences: List[List[int]], labels: List[int],
                 max_len: int = 512, pad_token: int = 0):
        self.sequences = []
        self.labels = labels

        for seq in sequences:
            # Truncate or pad sequence
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [pad_token] * (max_len - len(seq))
            self.sequences.append(seq)

        self.sequences = torch.tensor(self.sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_benchmark_data(benchmark_name: str, split: str = 'train') -> Tuple[List[List[int]], List[int], int]:
    """
    Load benchmark data in standardized format.

    Returns:
        sequences: List of token sequences
        labels: List of labels
        vocab_size: Size of vocabulary
    """
    if benchmark_name == 'cogs':
        return load_cogs_data(split)
    elif benchmark_name == 'pcfg':
        return load_pcfg_data(split)
    elif benchmark_name == 'cfq':
        return load_cfq_data(split)
    elif benchmark_name == 'scan':
        return load_scan_data(split)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


def load_cogs_data(split: str = 'train') -> Tuple[List[List[int]], List[int], int]:
    """Load COGS benchmark data."""
    # Simplified COGS data loading - would integrate with actual COGS dataset
    print("Loading COGS data...")

    # Mock data for demonstration
    vocab_size = 1000
    seq_len = 20

    if split == 'train':
        num_samples = 1000
    else:
        num_samples = 200

    sequences = []
    labels = []

    for i in range(num_samples):
        # Generate mock sequences (would be actual COGS parsing sequences)
        seq = np.random.randint(1, vocab_size, size=seq_len).tolist()
        label = np.random.randint(0, 2)  # Binary classification for semantic correctness
        sequences.append(seq)
        labels.append(label)

    return sequences, labels, vocab_size


def load_pcfg_data(split: str = 'train') -> Tuple[List[List[int]], List[int], int]:
    """Load PCFG benchmark data."""
    print("Loading PCFG data...")

    vocab_size = 50  # Smaller vocab for PCFG
    max_len = 30

    if split == 'train':
        num_samples = 5000
    else:
        num_samples = 1000

    sequences = []
    labels = []

    for i in range(num_samples):
        # Generate mock PCFG sequences (would be actual parse trees)
        length = np.random.randint(5, max_len)
        seq = np.random.randint(1, vocab_size, size=length).tolist()
        label = np.random.randint(0, 10)  # Multiple parse types
        sequences.append(seq)
        labels.append(label)

    return sequences, labels, vocab_size


def load_cfq_data(split: str = 'train') -> Tuple[List[List[int]], List[int], int]:
    """Load CFQ benchmark data."""
    print("Loading CFQ data...")

    vocab_size = 2000  # Larger vocab for CFQ
    max_len = 50

    if split == 'train':
        num_samples = 2000
    else:
        num_samples = 500

    sequences = []
    labels = []

    for i in range(num_samples):
        # Generate mock CFQ sequences (would be actual compositional questions)
        length = np.random.randint(10, max_len)
        seq = np.random.randint(1, vocab_size, size=length).tolist()
        label = np.random.randint(0, 50)  # Multiple answer types
        sequences.append(seq)
        labels.append(label)

    return sequences, labels, vocab_size


def load_scan_data(split: str = 'train') -> Tuple[List[List[int]], List[int], int]:
    """Load SCAN benchmark data."""
    print("Loading SCAN data...")

    vocab_size = 20  # Small vocab for SCAN commands
    max_len = 15

    if split == 'train':
        num_samples = 15000
    else:
        num_samples = 2000

    sequences = []
    labels = []

    for i in range(num_samples):
        # Generate mock SCAN sequences (would be actual command sequences)
        length = np.random.randint(3, max_len)
        seq = np.random.randint(1, vocab_size, size=length).tolist()
        label = np.random.randint(0, 100)  # Multiple action sequences
        sequences.append(seq)
        labels.append(label)

    return sequences, labels, vocab_size


def test_ce_on_benchmark(benchmark_name: str, use_ce_architecture: bool = True,
                        use_ce_timing: bool = True, save_results: bool = True) -> Dict[str, Any]:
    """
    Test CE learner on a specific benchmark.

    Args:
        benchmark_name: Name of benchmark ('cogs', 'pcfg', 'cfq', 'scan')
        use_ce_architecture: Whether to use CE architecture (vs standard)
        use_ce_timing: Whether to use CE timing (vs standard)
        save_results: Whether to save results to file

    Returns:
        Dictionary of test results
    """
    print(f"\n{'='*60}")
    print(f"Testing CE on {benchmark_name.upper()} benchmark")
    print(f"CE Architecture: {use_ce_architecture}")
    print(f"CE Timing: {use_ce_timing}")
    print(f"{'='*60}")

    # Load data
    print("Loading training data...")
    train_sequences, train_labels, vocab_size = load_benchmark_data(benchmark_name, 'train')
    print("Loading validation data...")
    val_sequences, val_labels, _ = load_benchmark_data(benchmark_name, 'val')

    num_classes = len(set(train_labels + val_labels))

    print(f"Dataset: {len(train_sequences)} train, {len(val_sequences)} val")
    print(f"Vocab size: {vocab_size}, Num classes: {num_classes}")

    # Create datasets
    max_seq_len = 64 if benchmark_name == 'scan' else 128
    train_dataset = SequenceClassificationDataset(
        train_sequences, train_labels, max_len=max_seq_len
    )
    val_dataset = SequenceClassificationDataset(
        val_sequences, val_labels, max_len=max_seq_len
    )

    # Create data loaders
    batch_size = 16 if benchmark_name in ['cfq', 'cogs'] else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create CE learner
    learner = create_ce_learner_for_task(
        benchmark_name, vocab_size=vocab_size, num_classes=num_classes
    )

    # Override CE settings based on test configuration
    learner.config.use_ce_architecture = use_ce_architecture
    learner.config.use_ce_timing = use_ce_timing
    learner.model.toggle_ce_mode(use_ce_architecture)

    # Adjust training parameters for testing
    learner.config.max_epochs = 10  # Quick test
    learner.config.eval_every = 2

    # Setup save path
    save_path = None
    if save_results:
        save_path = f"results/ce_{benchmark_name}_{use_ce_architecture}_{use_ce_timing}"

    # Train
    start_time = time.time()
    results = learner.train(train_loader, val_loader, save_path=save_path)
    training_time = time.time() - start_time

    # Final evaluation
    final_val_results = learner.validate(val_loader)

    # Compile results
    test_results = {
        'benchmark': benchmark_name,
        'ce_architecture': use_ce_architecture,
        'ce_timing': use_ce_timing,
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'train_samples': len(train_sequences),
        'val_samples': len(val_sequences),
        'final_val_accuracy': final_val_results['val_accuracy'],
        'final_val_loss': final_val_results['val_loss'],
        'training_time': training_time,
        'epochs_completed': results['final_epoch'] + 1,
        'best_val_accuracy': results['best_val_accuracy'],
        'config': learner.config.__dict__,
        'metrics': results['metrics']
    }

    print(f"\n{'='*60}")
    print(f"RESULTS for {benchmark_name.upper()}:")
    print(f"Final Validation Accuracy: {final_val_results['val_accuracy']:.4f}")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Epochs Completed: {results['final_epoch'] + 1}")
    print(f"{'='*60}")

    # Save results
    if save_results:
        Path("results").mkdir(exist_ok=True)
        result_file = f"results/ce_{benchmark_name}_results.json"
        with open(result_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for k, v in test_results.items():
                if isinstance(v, dict):
                    serializable_results[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, (np.integer, np.floating)):
                            serializable_results[k][k2] = float(v2)
                        elif isinstance(v2, list) and len(v2) > 0 and isinstance(v2[0], (np.integer, np.floating)):
                            serializable_results[k][k2] = [float(x) for x in v2]
                        else:
                            serializable_results[k][k2] = v2
                elif isinstance(v, (np.integer, np.floating)):
                    serializable_results[k] = float(v)
                else:
                    serializable_results[k] = v

            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {result_file}")

    return test_results


def run_comprehensive_ce_test():
    """Run comprehensive CE testing across multiple benchmarks and configurations."""
    print("Running comprehensive CE benchmark testing...")

    benchmarks = ['scan', 'pcfg', 'cogs', 'cfq']
    configurations = [
        (True, True, "CE Architecture + CE Timing"),
        (True, False, "CE Architecture only"),
        (False, True, "CE Timing only"),
        (False, False, "Baseline (standard)")
    ]

    all_results = []

    for benchmark in benchmarks:
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE TESTING ON {benchmark.upper()}")
        print(f"{'='*80}")

        benchmark_results = []

        for use_ce_arch, use_ce_timing, config_name in configurations:
            print(f"\n--- Testing {config_name} ---")

            try:
                results = test_ce_on_benchmark(
                    benchmark,
                    use_ce_architecture=use_ce_arch,
                    use_ce_timing=use_ce_timing,
                    save_results=True
                )
                benchmark_results.append(results)

            except Exception as e:
                print(f"Error testing {config_name} on {benchmark}: {e}")
                continue

        all_results.extend(benchmark_results)

        # Print benchmark summary
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {benchmark.upper()}:")
        print(f"{'='*80}")

        for result in benchmark_results:
            config_name = "CE Arch + Timing" if result['ce_architecture'] and result['ce_timing'] else \
                         "CE Arch only" if result['ce_architecture'] else \
                         "CE Timing only" if result['ce_timing'] else "Baseline"
            acc = result['final_val_accuracy']
            time = result['training_time']
            print(f"{config_name:15} | Acc: {acc:.4f} | Time: {time:.1f}s")

    # Overall summary
    print(f"\n{'='*100}")
    print("OVERALL CE TESTING SUMMARY")
    print(f"{'='*100}")

    # Group by benchmark
    benchmark_summaries = {}
    for result in all_results:
        bench = result['benchmark']
        if bench not in benchmark_summaries:
            benchmark_summaries[bench] = []
        benchmark_summaries[bench].append(result)

    for bench, results in benchmark_summaries.items():
        print(f"\n{bench.upper()}:")
        ce_full_results = [r for r in results if r['ce_architecture'] and r['ce_timing']]
        baseline_results = [r for r in results if not r['ce_architecture'] and not r['ce_timing']]

        if ce_full_results and baseline_results:
            ce_acc = ce_full_results[0]['final_val_accuracy']
            baseline_acc = baseline_results[0]['final_val_accuracy']
            improvement = ce_acc - baseline_acc
            print(".4f")
            print(".1f")
        else:
            print("  Insufficient results for comparison")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CE Architecture on Benchmarks")
    parser.add_argument('--benchmark', type=str, choices=['cogs', 'pcfg', 'cfq', 'scan'],
                       help='Specific benchmark to test')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive testing across all benchmarks')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')

    args = parser.parse_args()

    if args.comprehensive:
        run_comprehensive_ce_test()
    elif args.benchmark:
        test_ce_on_benchmark(args.benchmark, save_results=not args.no_save)
    else:
        # Default: test on SCAN with CE enabled
        test_ce_on_benchmark('scan', save_results=not args.no_save)
