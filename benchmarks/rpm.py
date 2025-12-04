"""
RPM Benchmark: Raven's Progressive Matrices

Tests visual systematic generalization through pattern completion tasks.
Models learn simple visual patterns and must generalize to novel combinations.
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


class RPMExample:
    """Represents an RPM example with 3x3 matrix and missing element."""

    def __init__(self, matrix: np.ndarray, target: np.ndarray, pattern_type: str, split: str = 'train'):
        self.matrix = matrix  # 3x3 grid of patterns
        self.target = target  # Missing bottom-right element
        self.pattern_type = pattern_type
        self.split = split

    def __str__(self):
        return f"RPM {self.pattern_type} pattern"


class RPMDataset(Dataset):
    """RPM dataset with train/dev/test splits."""

    def __init__(self, examples: List[RPMExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Convert to tensors
        matrix_tensor = torch.tensor(example.matrix, dtype=torch.float32).unsqueeze(0)  # [1, 3, 3, feature_dim]
        target_tensor = torch.tensor(example.target, dtype=torch.float32).unsqueeze(0)  # [1, feature_dim]

        return {
            'matrix': matrix_tensor,
            'target': target_tensor,
            'pattern_type': example.pattern_type
        }


class RPMGenerator:
    """Generates Raven's Progressive Matrices patterns."""

    def __init__(self, feature_dim: int = 16):
        self.feature_dim = feature_dim

        # Pattern types for systematic generalization
        self.pattern_types = [
            'constant',      # Same element repeated
            'progression',   # Arithmetic progression
            'distribution',  # Element distribution patterns
            'intersection',  # Row/column intersection rules
        ]

    def generate_examples(self, num_train: int = 200, num_test: int = 50) -> Dict[str, List[RPMExample]]:
        """Generate RPM examples with systematic generalization splits."""

        examples = {'train': [], 'dev': [], 'test': []}

        # Training examples: simple patterns with limited complexity
        for _ in range(num_train):
            pattern_type = random.choice(self.pattern_types[:2])  # Only first 2 types for training
            matrix, target = self.generate_pattern(pattern_type, complexity='simple')
            examples['train'].append(RPMExample(matrix, target, pattern_type, 'train'))

        # Development examples: mix of simple and complex patterns
        for _ in range(num_test // 2):
            pattern_type = random.choice(self.pattern_types[:2])
            matrix, target = self.generate_pattern(pattern_type, complexity='simple')
            examples['dev'].append(RPMExample(matrix, target, pattern_type, 'dev'))

        for _ in range(num_test // 2):
            pattern_type = random.choice(self.pattern_types)
            matrix, target = self.generate_pattern(pattern_type, complexity='complex')
            examples['dev'].append(RPMExample(matrix, target, pattern_type, 'dev'))

        # Test examples: complex patterns from all types (systematic generalization)
        for _ in range(num_test):
            pattern_type = random.choice(self.pattern_types)
            matrix, target = self.generate_pattern(pattern_type, complexity='complex')
            examples['test'].append(RPMExample(matrix, target, pattern_type, 'test'))

        return examples

    def generate_pattern(self, pattern_type: str, complexity: str = 'simple') -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single RPM pattern."""

        if pattern_type == 'constant':
            return self._generate_constant_pattern(complexity)
        elif pattern_type == 'progression':
            return self._generate_progression_pattern(complexity)
        elif pattern_type == 'distribution':
            return self._generate_distribution_pattern(complexity)
        elif pattern_type == 'intersection':
            return self._generate_intersection_pattern(complexity)
        else:
            return self._generate_constant_pattern(complexity)  # Fallback

    def _generate_constant_pattern(self, complexity: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate constant pattern (same element in all positions)."""
        # Create random base element
        base_element = np.random.randn(self.feature_dim)

        # Create 3x3 matrix with same element everywhere
        matrix = np.full((3, 3, self.feature_dim), base_element)

        # Target is the same element
        target = base_element

        return matrix, target

    def _generate_progression_pattern(self, complexity: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate arithmetic progression pattern."""
        # Create base elements
        elements = [np.random.randn(self.feature_dim) for _ in range(3)]

        if complexity == 'simple':
            # Simple row-wise progression
            matrix = np.zeros((3, 3, self.feature_dim))
            for i in range(3):
                for j in range(3):
                    matrix[i, j] = elements[i]  # Same element in each row
        else:
            # Complex progression with row and column variations
            matrix = np.zeros((3, 3, self.feature_dim))
            for i in range(3):
                for j in range(3):
                    # Combine row and column elements
                    matrix[i, j] = (elements[i] + elements[j]) / 2

        # Target combines last row and column elements
        target = (elements[2] + elements[2]) / 2

        return matrix, target

    def _generate_distribution_pattern(self, complexity: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate element distribution patterns."""
        elements = [np.random.randn(self.feature_dim) for _ in range(4)]  # Need 4 elements

        if complexity == 'simple':
            # Simple diagonal pattern
            matrix = np.zeros((3, 3, self.feature_dim))
            matrix[0, 0] = elements[0]
            matrix[0, 1] = elements[1]
            matrix[1, 0] = elements[1]
            matrix[1, 1] = elements[2]  # Missing one element
            matrix[0, 2] = elements[2]
            matrix[1, 2] = elements[3]
            matrix[2, 0] = elements[2]
            matrix[2, 1] = elements[3]
            # matrix[2, 2] is missing - this is our target
        else:
            # More complex distribution
            matrix = np.zeros((3, 3, self.feature_dim))
            # Fill with complex pattern
            positions = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)]
            for idx, (i, j) in enumerate(positions):
                matrix[i, j] = elements[idx % len(elements)]

        # Target based on pattern completion
        target = elements[2]  # Simplified

        return matrix, target

    def _generate_intersection_pattern(self, complexity: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate row/column intersection patterns."""
        elements = [np.random.randn(self.feature_dim) for _ in range(6)]

        if complexity == 'simple':
            # Simple row/column pattern
            matrix = np.zeros((3, 3, self.feature_dim))
            # Row patterns
            matrix[0, :] = elements[0]  # Row 0: element 0
            matrix[1, :] = elements[1]  # Row 1: element 1
            # Column patterns
            matrix[:, 0] = elements[2]  # Col 0: element 2
            matrix[:, 1] = elements[3]  # Col 1: element 3
            # Intersection rules
            matrix[0, 0] = (elements[0] + elements[2]) / 2  # Row+Col intersection
            matrix[0, 1] = (elements[0] + elements[3]) / 2
            matrix[1, 0] = (elements[1] + elements[2]) / 2
            matrix[1, 1] = (elements[1] + elements[3]) / 2
        else:
            # Complex intersection with more rules
            matrix = np.zeros((3, 3, self.feature_dim))
            # Similar but with more complex rules
            for i in range(2):  # Fill first 2 rows and columns
                for j in range(2):
                    matrix[i, j] = (elements[i] + elements[j+2]) / 2 + elements[4+i+j]

        # Target at intersection of last row and column
        target = (elements[1] + elements[3]) / 2

        return matrix, target


class RPMModel(nn.Module):
    """Neural network for RPM pattern completion."""

    def __init__(self, feature_dim: int = 16, hidden_dim: int = 128):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Convolutional encoder for pattern recognition
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),  # [1, 3, 3, feature_dim] -> [32, 3, 3, feature_dim]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1), # [32, 3, 3, feature_dim] -> [64, 3, 3, feature_dim]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling -> [64, 1, 1, feature_dim]
            nn.Flatten()  # -> [64 * feature_dim]
        )

        # Pattern reasoning network
        self.reasoner = nn.Sequential(
            nn.Linear(64 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)  # Output target element
        )

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            matrix: [batch, 1, 3, 3, feature_dim] - Input matrix

        Returns:
            target: [batch, feature_dim] - Predicted target element
        """
        batch_size = matrix.size(0)

        # Reshape to work with conv layers: [batch, 1, 3, 3*feature_dim]
        # Actually, let's reshape differently to preserve spatial structure
        matrix_reshaped = matrix.view(batch_size, 1, 3, 3 * self.feature_dim)  # [batch, 1, 3, 3*feature_dim]

        # Encode pattern
        encoded = self.encoder(matrix_reshaped)  # [batch, 64*feature_dim]

        # Reason about pattern to predict target
        target = self.reasoner(encoded)  # [batch, feature_dim]

        return target


class RPMBenchmark:
    """Complete RPM benchmark with training and evaluation."""

    def __init__(self, feature_dim: int = 16, hidden_dim: int = 128,
                 learning_rate: float = 1e-3, batch_size: int = 32):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Generate data
        generator = RPMGenerator(feature_dim=feature_dim)
        self.data = generator.generate_examples(num_train=300, num_test=50)

        # Create datasets
        self.train_dataset = RPMDataset(self.data['train'])
        self.dev_dataset = RPMDataset(self.data['dev'])
        self.test_dataset = RPMDataset(self.data['test'])

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def create_model(self) -> RPMModel:
        """Create a fresh RPM model."""
        return RPMModel(self.feature_dim, self.hidden_dim)

    def train_epoch(self, model: RPMModel, optimizer: optim.Optimizer,
                   criterion: nn.MSELoss, device: str = 'cpu') -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in self.train_loader:
            matrix = batch['matrix'].to(device)
            target = batch['target'].squeeze(1).to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(matrix)

            # Compute loss
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, model: RPMModel, dataset_name: str = 'test',
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
                matrix = batch['matrix'].to(device)
                target = batch['target'].squeeze(1).to(device)

                # Generate predictions
                predictions = model(matrix)

                # Compute loss
                loss = criterion(predictions, target)
                total_loss += loss.item()

                # Check if prediction is close to target (within tolerance)
                distances = torch.norm(predictions - target, dim=-1)
                correct += (distances < 0.5).sum().item()  # Tolerance of 0.5
                total += target.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

    def train_model(self, model: RPMModel, num_epochs: int = 100,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """Train model and return training history."""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        dev_accuracies = []

        for epoch in tqdm(range(num_epochs), desc="Training RPM"):
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


def run_rpm_baseline(num_epochs: int = 50, device: str = 'cpu') -> Dict[str, float]:
    """Run RPM benchmark with baseline CNN model."""
    print("🏃 Running RPM Baseline Benchmark...")

    benchmark = RPMBenchmark()

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
    results = run_rpm_baseline(num_epochs=30)
    print(f"\nRPM Baseline Results: {results}")
