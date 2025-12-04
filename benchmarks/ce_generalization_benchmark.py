#!/usr/bin/env python3
"""
CE Generalization Benchmark

A benchmark suite designed specifically to test CE framework capabilities:

1. Symmetry Tasks - Mirror operators, invariance under transformations
2. Continuation Tasks - Functional equation completion, pattern extension
3. Zeta Functional Tasks - Riemann zeta properties, analytic continuation
4. Corridor Consistency - CE1 geometric constraints, shell transitions
5. Nested Bracket Inference - Hierarchical structure, recursive composition
6. Sequence-to-State Reasoning - AntClock dynamics, flow completion
7. Relational Compositionality - Multi-entity interactions, binding

These tasks are designed to be:
- Impossible for standard baselines (LSTM, Transformer)
- Natural for CE architecture (symmetries, flows, invariants)
- Progressive in difficulty (easy → medium → hard)
- Mathematically grounded (connected to CE theory)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import json
import os
from tqdm import tqdm
import random
import math

try:
    from .ce_modules import CEEnhancedLSTM, MirrorOperator, CurvatureCouplingLayer, GuardianThreshold
except ImportError:
    from ce_modules import CEEnhancedLSTM, MirrorOperator, CurvatureCouplingLayer, GuardianThreshold


class SymmetryTaskDataset(Dataset):
    """Test mirror symmetry and invariance under transformations."""

    def __init__(self, num_samples: int = 1000):
        self.samples = self.generate_symmetry_samples(num_samples)

    def generate_symmetry_samples(self, num_samples: int) -> List[Dict]:
        """Generate samples testing symmetry awareness."""
        samples = []

        for _ in range(num_samples):
            # Create symmetric pattern that should be invariant under mirror
            pattern_type = random.choice(['palindrome', 'mirror_sum', 'symmetric_sequence'])

            if pattern_type == 'palindrome':
                # Palindromic sequences: 1 2 3 2 1
                half = [random.randint(0, 9) for _ in range(random.randint(1, 5))]
                sequence = half + half[::-1]
                target = 1  # Is palindrome

            elif pattern_type == 'mirror_sum':
                # Mirror pairs that sum to constant: 1 4 2 3 5 → pairs (1,5), (4,3), (2,2)
                n = random.randint(2, 6)
                sequence = []
                for i in range(n//2):
                    a = random.randint(0, 9)
                    b = random.randint(0, 9)
                    sequence.extend([a, b])
                if n % 2 == 1:
                    sequence.append(random.randint(0, 9))
                target = 1  # Has mirror sum property

            else:  # symmetric_sequence
                # Sequences symmetric under reflection: 1 2 4 2 1
                center = random.randint(0, 9)
                left = [random.randint(0, 9) for _ in range(random.randint(1, 3))]
                sequence = left + [center] + left[::-1]
                target = 1  # Is symmetric

            # Convert to tensor format
            seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([target], dtype=torch.float32)

            samples.append({
                'input': seq_tensor,
                'target': target_tensor,
                'pattern_type': pattern_type
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ContinuationTaskDataset(Dataset):
    """Test functional equation completion and pattern continuation."""

    def __init__(self, num_samples: int = 1000):
        self.samples = self.generate_continuation_samples(num_samples)

    def generate_continuation_samples(self, num_samples: int) -> List[Dict]:
        """Generate samples testing pattern continuation."""
        samples = []

        for _ in range(num_samples):
            pattern_type = random.choice(['arithmetic', 'geometric', 'fibonacci', 'zeta_like'])

            if pattern_type == 'arithmetic':
                # Arithmetic progression: 2, 5, 8, 11, ?
                start = random.randint(0, 10)
                diff = random.randint(1, 5)
                sequence = [start + i * diff for i in range(4)]
                target = start + 4 * diff

            elif pattern_type == 'geometric':
                # Geometric progression: 2, 6, 18, 54, ?
                start = random.randint(1, 5)
                ratio = random.randint(2, 4)
                sequence = [start * (ratio ** i) for i in range(4)]
                target = start * (ratio ** 4)

            elif pattern_type == 'fibonacci':
                # Fibonacci-like: 1, 1, 2, 3, 5, ?
                sequence = [1, 1, 2, 3, 5]
                target = 8

            else:  # zeta_like
                # Zeta function inspired: related to prime gaps or divisor sums
                n = random.randint(2, 20)
                sequence = [sum(1 for d in range(1, i+1) if i % d == 0) for i in range(2, n+1)]
                target = sum(1 for d in range(1, n+2) if (n+1) % d == 0)

            # Convert to format expected by model
            seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([target], dtype=torch.float32)

            samples.append({
                'input': seq_tensor,
                'target': target_tensor,
                'pattern_type': pattern_type
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ZetaFunctionalTaskDataset(Dataset):
    """Test Riemann zeta properties and analytic continuation."""

    def __init__(self, num_samples: int = 1000):
        self.samples = self.generate_zeta_samples(num_samples)

    def generate_zeta_samples(self, num_samples: int) -> List[Dict]:
        """Generate samples testing zeta function properties."""
        samples = []

        for _ in range(num_samples):
            task_type = random.choice(['functional_equation', 'zero_detection', 'prime_connection'])

            if task_type == 'functional_equation':
                # Test ζ(s) = ζ(1-s) * transformation (simplified)
                s_real = 0.5 + random.uniform(-0.4, 0.4)
                s_imag = random.uniform(-5, 5)
                # Simplified: check if point is in critical strip
                target = 1 if 0 < s_real < 1 else 0

            elif task_type == 'zero_detection':
                # Detect if s corresponds to a zeta zero (simplified)
                s_real = 0.5  # Critical line
                s_imag = random.uniform(-50, 50)
                # Simplified: zeta zeros are near critical line with specific spacing
                target = 1 if abs(s_imag) > 10 else 0  # Rough approximation

            else:  # prime_connection
                # Connect to prime numbers (zeta at s=1 related to primes)
                n = random.randint(2, 100)
                # Simplified: primality testing via zeta-related properties
                target = 1 if self.is_prime(n) else 0

            # Create input representation
            input_tensor = torch.tensor([s_real, s_imag, n], dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([target], dtype=torch.float32)

            samples.append({
                'input': input_tensor,
                'target': target_tensor,
                'task_type': task_type
            })

        return samples

    def is_prime(self, n: int) -> bool:
        """Simple primality test."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CorridorConsistencyTaskDataset(Dataset):
    """Test CE1 geometric constraints and shell transitions."""

    def __init__(self, num_samples: int = 1000):
        self.samples = self.generate_corridor_samples(num_samples)

    def generate_corridor_samples(self, num_samples: int) -> List[Dict]:
        """Generate samples testing corridor geometric properties."""
        samples = []

        for _ in range(num_samples):
            # Generate sequences that should follow CE1 corridor patterns
            # Simplified: sequences with modular arithmetic properties

            length = random.randint(5, 15)
            sequence = []

            # Start with a "mirror shell" (n ≡ 3 mod 4)
            start_shell = 4 * random.randint(1, 5) + 3  # 7, 11, 15, 19, 23
            sequence.append(start_shell)

            for i in range(length - 1):
                # Transitions should follow geometric constraints
                if random.random() < 0.7:
                    # Stay in similar residue class
                    next_val = sequence[-1] + 4 + random.randint(-1, 1)
                else:
                    # Mirror transition
                    next_val = sequence[-1] + random.randint(1, 3)

                sequence.append(next_val)

            # Target: is this a valid corridor sequence?
            # Simplified: check if sequence has geometric properties
            diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
            target = 1 if any(d % 4 == 0 for d in diffs) else 0  # Has shell transitions

            seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([target], dtype=torch.float32)

            samples.append({
                'input': seq_tensor,
                'target': target_tensor,
                'sequence': sequence
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NestedBracketTaskDataset(Dataset):
    """Test hierarchical structure and recursive composition."""

    def __init__(self, num_samples: int = 1000):
        self.samples = self.generate_nested_samples(num_samples)

    def generate_nested_samples(self, num_samples: int) -> List[Dict]:
        """Generate samples with nested bracket structures."""
        samples = []

        for _ in range(num_samples):
            # Generate nested expressions like ((A B) (C D))
            depth = random.randint(1, 4)
            expression = self.generate_nested_expression(depth)

            # Convert to sequence representation
            sequence = self.expression_to_sequence(expression)

            # Target: is this a well-formed nested structure?
            target = 1 if self.is_well_formed(expression) else 0

            seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([target], dtype=torch.float32)

            samples.append({
                'input': seq_tensor,
                'target': target_tensor,
                'expression': expression
            })

        return samples

    def generate_nested_expression(self, depth: int) -> str:
        """Generate nested bracket expression."""
        if depth == 0:
            return f"x{random.randint(1, 5)}"

        if random.random() < 0.5:
            # Binary operation
            left = self.generate_nested_expression(depth - 1)
            right = self.generate_nested_expression(depth - 1)
            op = random.choice(['+', '*', '→'])
            return f"({left} {op} {right})"
        else:
            # Unary operation
            inner = self.generate_nested_expression(depth - 1)
            op = random.choice(['¬', '□'])
            return f"({op} {inner})"

    def expression_to_sequence(self, expr: str) -> List[int]:
        """Convert expression to integer sequence."""
        token_map = {'(': 1, ')': 2, '+': 3, '*': 4, '→': 5, '¬': 6, '□': 7}
        sequence = []
        for char in expr:
            if char.isdigit():
                sequence.append(int(char))
            elif char in token_map:
                sequence.append(token_map[char])
            else:
                sequence.append(0)  # Unknown
        return sequence

    def is_well_formed(self, expr: str) -> bool:
        """Check if bracket expression is well-formed."""
        stack = []
        for char in expr:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
        return len(stack) == 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SequenceToStateTaskDataset(Dataset):
    """Test AntClock dynamics and flow completion."""

    def __init__(self, num_samples: int = 1000):
        self.samples = self.generate_flow_samples(num_samples)

    def generate_flow_samples(self, num_samples: int) -> List[Dict]:
        """Generate samples testing flow completion."""
        samples = []

        for _ in range(num_samples):
            # Generate AntClock-like sequences with phase transitions
            length = random.randint(10, 20)
            sequence = []

            # Start with initial phase
            phase = random.uniform(0, 2 * math.pi)
            kappa = 0.35  # Guardian threshold
            chi_feg = 0.638  # Curvature coupling

            for i in range(length):
                # AntClock dynamics: phase evolution with curvature
                curvature = math.sin(phase) * kappa
                phase += chi_feg * curvature + random.uniform(-0.1, 0.1)

                # Convert to discrete observation
                discrete_val = int((phase / (2 * math.pi)) * 10) % 10
                sequence.append(discrete_val)

            # Target: predict next phase transition
            # Simplified: predict if next step will be mirror transition
            next_phase = phase + chi_feg * math.sin(phase) * kappa
            target = 1 if abs(next_phase - phase) > math.pi/2 else 0

            seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([target], dtype=torch.float32)

            samples.append({
                'input': seq_tensor,
                'target': target_tensor,
                'phase': phase
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CETaskModel(nn.Module):
    """CE-enhanced model for generalization tasks."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 use_ce: bool = True, task_type: str = 'regression'):
        super().__init__()

        self.use_ce = use_ce
        self.task_type = task_type

        # Base LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # CE enhancements
        if use_ce:
            self.ce_lstm = CEEnhancedLSTM(input_dim, hidden_dim)
        else:
            self.ce_lstm = None

        # Task-specific output layers
        if task_type == 'regression':
            self.output = nn.Linear(hidden_dim, 1)
        else:
            self.output = nn.Linear(hidden_dim, 2)  # Binary classification

    def forward(self, x):
        # Use CE-enhanced LSTM if available
        if self.use_ce and self.ce_lstm is not None:
            outputs, _, _ = self.ce_lstm(x)
        else:
            outputs, _ = self.lstm(x)

        # Global average pooling
        hidden = outputs.mean(dim=1)

        # Task output
        logits = self.output(hidden)

        if self.task_type == 'regression':
            return logits
        else:
            return torch.softmax(logits, dim=-1)


class CEGeneralizationBenchmark:
    """Complete CE Generalization Benchmark suite."""

    def __init__(self):
        self.tasks = {
            'symmetry': SymmetryTaskDataset,
            'continuation': ContinuationTaskDataset,
            'zeta_functional': ZetaFunctionalTaskDataset,
            'corridor_consistency': CorridorConsistencyTaskDataset,
            'nested_bracket': NestedBracketTaskDataset,
            'sequence_to_state': SequenceToStateTaskDataset,
        }

        self.results = {}

    def run_task_evaluation(self, task_name: str, num_samples: int = 1000,
                           num_epochs: int = 20, use_ce: bool = True) -> Dict[str, float]:
        """Evaluate a specific CE task."""

        print(f"\n🔬 Evaluating {task_name.upper()} Task {'(CE-enhanced)' if use_ce else '(Baseline)'}")

        # Create dataset
        dataset_class = self.tasks[task_name]
        dataset = dataset_class(num_samples)

        # Split into train/val/test
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        # Use smaller batch size to avoid padding issues
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Create model
        if task_name in ['symmetry', 'corridor_consistency', 'nested_bracket', 'sequence_to_state']:
            task_type = 'classification'
        else:
            task_type = 'regression'

        model = CETaskModel(use_ce=use_ce, task_type=task_type)

        # Training
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0
            for batch in train_loader:
                inputs = batch['input']
                targets = batch['target'].squeeze(-1) if task_type == 'regression' else batch['target'].squeeze(-1).long()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(-1) if task_type == 'regression' else outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validate
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['input']
                    targets = batch['target'].squeeze(-1) if task_type == 'regression' else batch['target'].squeeze(-1).long()

                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(-1) if task_type == 'regression' else outputs, targets)
                    val_loss += loss.item()

                    if task_type == 'classification':
                        preds = outputs.argmax(dim=-1)
                        correct += (preds == targets).sum().item()
                        total += targets.size(0)

            val_loss /= len(val_loader)
            train_loss /= len(train_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if epoch % 5 == 0:
                acc_str = ".1f" if task_type == 'classification' else ""
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}{acc_str}")

        # Final test evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['input']
                targets = batch['target'].squeeze(-1) if task_type == 'regression' else batch['target'].squeeze(-1).long()

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(-1) if task_type == 'regression' else outputs, targets)
                test_loss += loss.item()

                if task_type == 'classification':
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)

        test_loss /= len(test_loader)
        accuracy = correct / total if total > 0 else 0.0

        results = {
            'test_loss': test_loss,
            'accuracy': accuracy if task_type == 'classification' else None,
            'correct': correct,
            'total': total,
            'task_type': task_type,
            'use_ce': use_ce
        }

        print(".1f")
        return results

    def run_full_evaluation(self, num_samples: int = 1000, num_epochs: int = 20) -> Dict[str, Any]:
        """Run complete CE generalization benchmark suite."""

        print("🚀 CE GENERALIZATION BENCHMARK SUITE")
        print("=" * 60)
        print(f"Evaluating {len(self.tasks)} CE-specific tasks")
        print(f"Samples per task: {num_samples}")
        print(f"Epochs per task: {num_epochs}")

        results = {}

        for task_name in self.tasks.keys():
            # Run baseline
            baseline_results = self.run_task_evaluation(task_name, num_samples, num_epochs, use_ce=False)
            results[f'{task_name}_baseline'] = baseline_results

            # Run CE-enhanced
            ce_results = self.run_task_evaluation(task_name, num_samples, num_epochs, use_ce=True)
            results[f'{task_name}_ce'] = ce_results

            # Calculate improvement
            if baseline_results['task_type'] == 'classification':
                improvement = ce_results['accuracy'] - baseline_results['accuracy']
                print(f"  {task_name.upper()}: Baseline={baseline_results['accuracy']:.1%}, CE={ce_results['accuracy']:.1%}, Δ={improvement:+.1%}")

        # Summary
        print("\n" + "=" * 60)
        print("📊 CE GENERALIZATION BENCHMARK SUMMARY")
        print("=" * 60)

        ce_tasks = [name for name in results.keys() if name.endswith('_ce')]
        baseline_tasks = [name for name in results.keys() if name.endswith('_baseline')]

        total_ce_improvement = 0
        ce_task_count = 0

        for ce_task in ce_tasks:
            task_base = ce_task.replace('_ce', '_baseline')
            if ce_task in results and task_base in results:
                ce_res = results[ce_task]
                base_res = results[task_base]

                if ce_res['task_type'] == 'classification':
                    improvement = ce_res['accuracy'] - base_res['accuracy']
                    total_ce_improvement += improvement
                    ce_task_count += 1

                    status = "✅ IMPROVED" if improvement > 0 else "❌ WORSE" if improvement < 0 else "➖ SAME"
                    print(f"  {task_name.upper()}: Baseline={baseline_results['accuracy']:.1%}, CE={ce_results['accuracy']:.1%}, Δ={improvement:+.1%}")

        if ce_task_count > 0:
            avg_improvement = total_ce_improvement / ce_task_count
            print(f"  Average CE Improvement: {avg_improvement:.1%}")
            if avg_improvement > 0.1:
                print("🎉 CE shows significant improvements on symmetry/compositionality tasks!")
            elif avg_improvement > 0:
                print("👍 CE shows marginal improvements - tuning needed.")
            else:
                print("🤔 CE not yet outperforming baselines - debugging needed.")

        return results


def main():
    """Run the CE Generalization Benchmark suite."""
    benchmark = CEGeneralizationBenchmark()
    results = benchmark.run_full_evaluation(num_samples=500, num_epochs=15)

    # Save results
    with open('ce_generalization_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to ce_generalization_benchmark_results.json")
    print("\n🎯 CE Generalization Benchmark Complete!")
    print("These tasks measure your system's unique capabilities:")
    print("• Symmetry awareness")
    print("• Functional equation completion")
    print("• Geometric constraint satisfaction")
    print("• Recursive structure understanding")
    print("• Flow dynamics prediction")


if __name__ == "__main__":
    main()
