"""
Math Reasoning Benchmark: Systematic Mathematical Reasoning

Tests systematic generalization in mathematical reasoning through
pattern completion and algebraic manipulation tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import os
from tqdm import tqdm
import random


class MathExample:
    """Represents a math reasoning example with pattern and answer."""

    def __init__(self, pattern: str, answer: float, pattern_type: str, split: str = 'train'):
        self.pattern = pattern.strip()
        self.answer = answer
        self.pattern_type = pattern_type
        self.split = split

    def __str__(self):
        return f"{self.pattern} = {self.answer}"


class MathDataset(Dataset):
    """Math reasoning dataset with train/dev/test splits."""

    def __init__(self, examples: List[MathExample], vocab: 'MathVocab', max_len: int = 50):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize pattern
        tokens = self.vocab.tokenize(example.pattern)
        token_ids = [self.vocab.word2idx.get(token, self.vocab.unk_idx) for token in tokens]
        token_ids = token_ids[:self.max_len]
        token_ids += [self.vocab.pad_idx] * (self.max_len - len(token_ids))

        # Create target (regression)
        target = torch.tensor([example.answer], dtype=torch.float32)

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'target': target,
            'pattern_type': example.pattern_type
        }


class MathVocab:
    """Vocabulary for mathematical expressions."""

    def __init__(self, expressions: List[str]):
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

        # Build vocabulary
        all_tokens = set()
        for expr in expressions:
            all_tokens.update(self.tokenize(expr))

        # Add numbers and operators
        all_tokens.update(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        all_tokens.update(['+', '-', '*', '/', '=', '(', ')', '[', ']', '{', '}', ','])

        # Add special tokens
        for token in self.special_tokens:
            all_tokens.add(token)

        # Create mappings
        for idx, token in enumerate(sorted(all_tokens)):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    @property
    def pad_idx(self):
        return self.word2idx['<pad>']

    @property
    def sos_idx(self):
        return self.word2idx['<sos>']

    @property
    def eos_idx(self):
        return self.word2idx['<eos>']

    @property
    def unk_idx(self):
        return self.word2idx['<unk>']

    def tokenize(self, text: str) -> List[str]:
        """Tokenize mathematical expression."""
        tokens = []
        current = ""
        for char in text:
            if char in ['+', '-', '*', '/', '=', '(', ')', '[', ']', '{', '}', ',', ' ']:
                if current:
                    tokens.append(current)
                    current = ""
                if char != ' ':
                    tokens.append(char)
            else:
                current += char
        if current:
            tokens.append(current)
        return tokens

    def __len__(self):
        return len(self.word2idx)


class MathGenerator:
    """Generates mathematical reasoning patterns for systematic generalization."""

    def __init__(self):
        self.pattern_types = [
            'arithmetic_progression',  # Find next term in sequence
            'algebraic_patterns',      # Solve for x in patterns
            'geometric_reasoning',     # Shape/area patterns
            'number_properties',       # Prime, even/odd patterns
        ]

    def generate_examples(self, num_train: int = 200, num_test: int = 50) -> Dict[str, List[MathExample]]:
        """Generate math examples with systematic generalization splits."""

        examples = {'train': [], 'dev': [], 'test': []}

        # Training examples: simple patterns with limited complexity
        for _ in range(num_train):
            pattern_type = random.choice(self.pattern_types[:2])  # Only first 2 types for training
            pattern, answer = self.generate_pattern(pattern_type, complexity='simple')
            examples['train'].append(MathExample(pattern, answer, pattern_type, 'train'))

        # Development examples: mix of simple and complex
        for _ in range(num_test // 2):
            pattern_type = random.choice(self.pattern_types[:2])
            pattern, answer = self.generate_pattern(pattern_type, complexity='simple')
            examples['dev'].append(MathExample(pattern, answer, pattern_type, 'dev'))

        for _ in range(num_test // 2):
            pattern_type = random.choice(self.pattern_types)
            pattern, answer = self.generate_pattern(pattern_type, complexity='complex')
            examples['dev'].append(MathExample(pattern, answer, pattern_type, 'dev'))

        # Test examples: complex patterns from all types (systematic generalization)
        for _ in range(num_test):
            pattern_type = random.choice(self.pattern_types)
            pattern, answer = self.generate_pattern(pattern_type, complexity='complex')
            examples['test'].append(MathExample(pattern, answer, pattern_type, 'test'))

        return examples

    def generate_pattern(self, pattern_type: str, complexity: str = 'simple') -> Tuple[str, float]:
        """Generate a single mathematical pattern."""

        if pattern_type == 'arithmetic_progression':
            return self._generate_arithmetic_progression(complexity)
        elif pattern_type == 'algebraic_patterns':
            return self._generate_algebraic_pattern(complexity)
        elif pattern_type == 'geometric_reasoning':
            return self._generate_geometric_pattern(complexity)
        elif pattern_type == 'number_properties':
            return self._generate_number_property_pattern(complexity)
        else:
            return self._generate_arithmetic_progression(complexity)  # Fallback

    def _generate_arithmetic_progression(self, complexity: str) -> Tuple[str, float]:
        """Generate arithmetic progression pattern."""
        if complexity == 'simple':
            # Simple sequence: 2, 4, 6, 8, ?
            start = random.randint(1, 10)
            diff = random.randint(1, 5)
            sequence = [start + i * diff for i in range(4)]
            answer = start + 4 * diff
            pattern = f"{sequence[0]}, {sequence[1]}, {sequence[2]}, {sequence[3]}, ?"
        else:
            # Complex: nested or multi-step patterns
            start = random.randint(1, 10)
            diff1 = random.randint(1, 3)
            diff2 = random.randint(1, 3)
            sequence = []
            for i in range(4):
                if i % 2 == 0:
                    sequence.append(start + (i // 2) * diff1)
                else:
                    sequence.append(start + (i // 2) * diff1 + diff2)
            answer = start + 2 * diff1 + diff2  # Pattern continues
            pattern = f"{sequence[0]}, {sequence[1]}, {sequence[2]}, {sequence[3]}, ?"

        return pattern, float(answer)

    def _generate_algebraic_pattern(self, complexity: str) -> Tuple[str, float]:
        """Generate algebraic pattern completion."""
        if complexity == 'simple':
            # Simple: solve 2x + 3 = 7, find x
            x = random.randint(1, 10)
            a = random.randint(1, 5)
            b = random.randint(1, 10)
            c = a * x + b
            pattern = f"Solve: {a}x + {b} = {c}, x = ?"
            answer = x
        else:
            # Complex: system of equations or quadratic
            x = random.randint(1, 5)
            y = random.randint(1, 5)
            a = random.randint(1, 3)
            b = random.randint(1, 3)
            c1 = a * x + b * y
            c2 = a * y + b * x  # Symmetric
            pattern = f"Solve system: {a}x + {b}y = {c1}, {a}y + {b}x = {c2}, x + y = ?"
            answer = x + y

        return pattern, float(answer)

    def _generate_geometric_pattern(self, complexity: str) -> Tuple[str, float]:
        """Generate geometric reasoning pattern."""
        if complexity == 'simple':
            # Area of square with side length
            side = random.randint(2, 10)
            pattern = f"Area of square with side {side} = ?"
            answer = side * side
        else:
            # Complex: composite shapes or 3D
            length = random.randint(3, 8)
            width = random.randint(2, 6)
            height = random.randint(2, 5)
            pattern = f"Volume of rectangular prism: {length} × {width} × {height} = ?"
            answer = length * width * height

        return pattern, float(answer)

    def _generate_number_property_pattern(self, complexity: str) -> Tuple[str, float]:
        """Generate number property pattern."""
        if complexity == 'simple':
            # Is number even or odd?
            num = random.randint(1, 100)
            pattern = f"Is {num} even? (1=yes, 0=no)"
            answer = 1.0 if num % 2 == 0 else 0.0
        else:
            # Complex: prime checking or factors
            num = random.randint(2, 50)
            pattern = f"How many factors does {num} have?"
            factors = [i for i in range(1, num + 1) if num % i == 0]
            answer = len(factors)

        return pattern, float(answer)


class MathModel(nn.Module):
    """Neural network for mathematical reasoning."""

    def __init__(self, vocab: MathVocab, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        self.vocab = vocab

        # Encoder
        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Reasoning network
        self.reasoner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Regression output
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        embeds = self.embed(input_ids)
        _, (hidden, _) = self.encoder(embeds)
        hidden = hidden[-1]  # Use last layer

        output = self.reasoner(hidden)
        return output


class MathBenchmark:
    """Complete math reasoning benchmark with training and evaluation."""

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128,
                 learning_rate: float = 1e-3, batch_size: int = 32):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Generate data
        generator = MathGenerator()
        self.data = generator.generate_examples(num_train=300, num_test=50)

        # Create vocabulary
        all_patterns = [ex.pattern for ex in self.data['train'] + self.data['dev'] + self.data['test']]
        self.vocab = MathVocab(all_patterns)

        # Create datasets
        self.train_dataset = MathDataset(self.data['train'], self.vocab)
        self.dev_dataset = MathDataset(self.data['dev'], self.vocab)
        self.test_dataset = MathDataset(self.data['test'], self.vocab)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def create_model(self) -> MathModel:
        """Create a fresh math reasoning model."""
        return MathModel(self.vocab, self.embed_dim, self.hidden_dim)

    def train_epoch(self, model: MathModel, optimizer: optim.Optimizer,
                   criterion: nn.MSELoss, device: str = 'cpu') -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(input_ids)

            # Compute loss
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, model: MathModel, dataset_name: str = 'test',
                device: str = 'cpu') -> Dict[str, float]:
        """Evaluate model on specified dataset."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        loader = getattr(self, f'{dataset_name}_loader')
        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['target'].to(device)

                # Generate predictions
                predictions = model(input_ids)

                # Compute loss
                loss = criterion(predictions, targets)
                total_loss += loss.item()

                # Check if prediction is close to target (within tolerance)
                errors = torch.abs(predictions - targets)
                correct += (errors < 0.5).sum().item()  # Tolerance of 0.5
                total += targets.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    def train_model(self, model: MathModel, num_epochs: int = 100,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """Train model and return training history."""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        dev_accuracies = []

        for epoch in tqdm(range(num_epochs), desc="Training Math"):
            # Train
            train_loss = self.train_epoch(model, optimizer, criterion, device)
            train_losses.append(train_loss)

            # Evaluate on dev set
            if epoch % 10 == 0:
                dev_metrics = self.evaluate(model, 'dev', device)
                dev_accuracies.append(dev_metrics['accuracy'])
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Dev Acc={dev_metrics['accuracy']:.4f}")

        return {
            'train_losses': train_losses,
            'dev_accuracies': dev_accuracies
        }


def run_math_baseline(num_epochs: int = 50, device: str = 'cpu') -> Dict[str, float]:
    """Run math reasoning benchmark with baseline LSTM model."""
    print("🏃 Running Math Reasoning Baseline Benchmark...")

    benchmark = MathBenchmark()

    # Create and train model
    model = benchmark.create_model()
    history = benchmark.train_model(model, num_epochs, device)

    # Final evaluation
    dev_metrics = benchmark.evaluate(model, 'dev', device)
    test_metrics = benchmark.evaluate(model, 'test', device)

    results = {
        'train_loss_final': history['train_losses'][-1],
        'dev_loss': dev_metrics['loss'],
        'dev_accuracy': dev_metrics['accuracy'],
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'dev_correct': dev_metrics['correct'],
        'dev_total': dev_metrics['total'],
        'test_correct': test_metrics['correct'],
        'test_total': test_metrics['total']
    }

    print(f"Baseline Loss: {results['train_loss_final']:.4f}")
    print(f"Dev Accuracy: {results['dev_accuracy']:.1%} ({results['dev_correct']}/{results['dev_total']})")
    print(f"Test Accuracy: {results['test_accuracy']:.1%} ({results['test_correct']}/{results['test_total']})")

    return results


if __name__ == "__main__":
    # Test the benchmark
    results = run_math_baseline(num_epochs=30)
    print(f"\nMath Reasoning Baseline Results: {results}")
