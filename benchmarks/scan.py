"""
SCAN Benchmark: Simple Compositionality with Addition of Numbers

Tests systematic generalization in sequence-to-sequence models.
Models learn simple commands and must generalize to novel compositions.
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


class SCANCommand:
    """Represents a SCAN command with input and output sequences."""

    def __init__(self, command: str, actions: str):
        self.command = command.strip()
        self.actions = actions.strip()

    def __str__(self):
        return f"{self.command} -> {self.actions}"


class SCANDataset(Dataset):
    """SCAN dataset with train/test splits."""

    def __init__(self, commands: List[SCANCommand], vocab_command: 'SCANVocab',
                 vocab_action: 'SCANVocab', max_len: int = 50):
        self.commands = commands
        self.vocab_command = vocab_command
        self.vocab_action = vocab_action
        self.max_len = max_len

    def __len__(self):
        return len(self.commands)

    def __getitem__(self, idx):
        cmd = self.commands[idx]

        # Tokenize command
        cmd_tokens = self.vocab_command.tokenize(cmd.command)
        cmd_ids = [self.vocab_command.word2idx[token] for token in cmd_tokens]
        cmd_ids = cmd_ids[:self.max_len]  # Truncate
        cmd_ids += [self.vocab_command.pad_idx] * (self.max_len - len(cmd_ids))

        # Tokenize actions
        action_tokens = self.vocab_action.tokenize(cmd.actions)
        action_ids = [self.vocab_action.word2idx[token] for token in action_tokens]
        action_ids = action_ids[:self.max_len]  # Truncate
        action_ids += [self.vocab_action.pad_idx] * (self.max_len - len(action_ids))

        return {
            'command_ids': torch.tensor(cmd_ids, dtype=torch.long),
            'action_ids': torch.tensor(action_ids, dtype=torch.long),
            'command_len': len(cmd_tokens),
            'action_len': len(action_tokens)
        }


class SCANVocab:
    """Vocabulary for SCAN commands and actions."""

    def __init__(self, commands: List[str]):
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

        # Build vocabulary
        all_words = set()
        for cmd in commands:
            all_words.update(self.tokenize(cmd))

        # Add special tokens
        for token in self.special_tokens:
            all_words.add(token)

        # Create mappings
        for idx, word in enumerate(sorted(all_words)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

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
        """Simple tokenization by splitting on spaces."""
        return text.split()

    def __len__(self):
        return len(self.word2idx)


class SCANDataLoader:
    """Loads SCAN dataset from predefined splits."""

    @staticmethod
    def load_scan_data() -> Dict[str, List[SCANCommand]]:
        """Load the SCAN dataset with predefined train/test splits."""

        # Try to load real SCAN dataset first
        try:
            train_commands = []
            test_commands = []

            # Load the main simple split
            train_file = 'benchmarks/real_data/scan/simple_split/tasks_train_simple.txt'
            test_file = 'benchmarks/real_data/scan/simple_split/tasks_test_simple.txt'

            # Parse training data
            with open(train_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('IN: ') and ' OUT: ' in line:
                        parts = line.split(' OUT: ')
                        command = parts[0].replace('IN: ', '')
                        actions = parts[1]
                        train_commands.append(SCANCommand(command, actions))

            # Parse test data
            with open(test_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('IN: ') and ' OUT: ' in line:
                        parts = line.split(' OUT: ')
                        command = parts[0].replace('IN: ', '')
                        actions = parts[1]
                        test_commands.append(SCANCommand(command, actions))

            print(f"Loaded REAL SCAN dataset: {len(train_commands)} train, {len(test_commands)} test")
            return {'train': train_commands, 'test': test_commands}

        except FileNotFoundError:
            # Fallback to synthetic datasets
            try:
                import json
                with open('benchmarks/comprehensive_scan_dataset.json', 'r') as f:
                    data = json.load(f)

                train_commands = [SCANCommand(ex['command'], ex['actions']) for ex in data['train']]
                test_commands = [SCANCommand(ex['command'], ex['actions']) for ex in data['test']]

                print(f"Loaded comprehensive SCAN dataset: {len(train_commands)} train, {len(test_commands)} test")
                return {'train': train_commands, 'test': test_commands}

            except FileNotFoundError:
                print("No datasets found, using original small dataset")

        except FileNotFoundError:
            # Fallback to original small dataset
            print("Enhanced dataset not found, using original small dataset")

            # Simple SCAN commands (subset for demonstration)
            train_commands = [
                # Basic commands
                SCANCommand("jump", "JUMP"),
                SCANCommand("turn left", "TURN_LEFT"),
                SCANCommand("turn right", "TURN_RIGHT"),
                SCANCommand("walk", "WALK"),
                SCANCommand("look", "LOOK"),
                SCANCommand("run", "RUN"),

                # Jump variations
                SCANCommand("jump twice", "JUMP JUMP"),
                SCANCommand("jump thrice", "JUMP JUMP JUMP"),

                # Turn variations
                SCANCommand("turn left twice", "TURN_LEFT TURN_LEFT"),
                SCANCommand("turn left thrice", "TURN_LEFT TURN_LEFT TURN_LEFT"),
                SCANCommand("turn right twice", "TURN_RIGHT TURN_RIGHT"),
                SCANCommand("turn right thrice", "TURN_RIGHT TURN_RIGHT TURN_RIGHT"),

                # Opposite commands
                SCANCommand("turn opposite left", "TURN_RIGHT"),
                SCANCommand("turn opposite right", "TURN_LEFT"),
                SCANCommand("turn opposite left twice", "TURN_RIGHT TURN_RIGHT"),
                SCANCommand("turn opposite right twice", "TURN_LEFT TURN_LEFT"),

                # Around commands
                SCANCommand("turn around left", "TURN_LEFT TURN_LEFT"),
                SCANCommand("turn around right", "TURN_RIGHT TURN_RIGHT"),

                # Combined commands
                SCANCommand("jump and turn left", "JUMP TURN_LEFT"),
                SCANCommand("jump and turn right", "JUMP TURN_RIGHT"),
                SCANCommand("turn left and jump", "TURN_LEFT JUMP"),
                SCANCommand("turn right and jump", "TURN_RIGHT JUMP"),

                # Complex combinations
                SCANCommand("jump twice and turn left", "JUMP JUMP TURN_LEFT"),
                SCANCommand("turn left and jump twice", "TURN_LEFT JUMP JUMP"),
                SCANCommand("jump and turn opposite left", "JUMP TURN_RIGHT"),
                SCANCommand("turn opposite right and jump twice", "TURN_LEFT JUMP JUMP"),
            ]

            # Test commands (novel compositions for generalization)
            test_commands = [
                # Length generalization
                SCANCommand("jump jump", "JUMP JUMP"),  # Primitive recursion
                SCANCommand("turn left turn left", "TURN_LEFT TURN_LEFT"),
                SCANCommand("turn right turn right", "TURN_RIGHT TURN_RIGHT"),

                # Novel turn-around combinations
                SCANCommand("jump and turn around left", "JUMP TURN_LEFT TURN_LEFT"),
                SCANCommand("turn around right and jump", "TURN_RIGHT TURN_RIGHT JUMP"),
                SCANCommand("jump twice and turn around left", "JUMP JUMP TURN_LEFT TURN_LEFT"),

                # Complex opposite combinations
                SCANCommand("turn opposite left thrice", "TURN_RIGHT TURN_RIGHT TURN_RIGHT"),
                SCANCommand("turn opposite right and turn left", "TURN_LEFT TURN_LEFT"),
                SCANCommand("jump and turn opposite right twice", "JUMP TURN_LEFT TURN_LEFT"),

                # Higher-order compositions
                SCANCommand("run and jump twice", "RUN JUMP JUMP"),
                SCANCommand("walk twice and turn right", "WALK WALK TURN_RIGHT"),
                SCANCommand("look and turn opposite left twice", "LOOK TURN_RIGHT TURN_RIGHT"),
            ]

            return {
                'train': train_commands,
                'test': test_commands
            }


class SCANModel(nn.Module):
    """Sequence-to-sequence model for SCAN benchmark."""

    def __init__(self, vocab_command: SCANVocab, vocab_action: SCANVocab,
                 embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()

        self.vocab_command = vocab_command
        self.vocab_action = vocab_action

        # Encoder
        self.encoder_embed = nn.Embedding(len(vocab_command), embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

        # Decoder
        self.decoder_embed = nn.Embedding(len(vocab_action), embed_dim)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, len(vocab_action))

    def encode(self, command_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Encode command sequence."""
        embeds = self.encoder_embed(command_ids)
        outputs, (hidden, cell) = self.encoder(embeds)
        return outputs, (hidden, cell)

    def decode_step(self, input_token: torch.Tensor, hidden: Tuple) -> Tuple[torch.Tensor, Tuple]:
        """Single decoder step."""
        embed = self.decoder_embed(input_token.unsqueeze(1))
        output, hidden = self.decoder(embed, hidden)
        logits = self.output_proj(output.squeeze(1))
        return logits, hidden

    def forward(self, command_ids: torch.Tensor, action_ids: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """Forward pass for training."""
        batch_size = command_ids.size(0)
        max_len = action_ids.size(1) if action_ids is not None else 50

        # Encode
        _, (hidden, cell) = self.encode(command_ids)

        # Decode
        outputs = []
        input_token = torch.full((batch_size,), self.vocab_action.sos_idx,
                               device=command_ids.device)

        for t in range(max_len):
            logits, (hidden, cell) = self.decode_step(input_token, (hidden, cell))
            outputs.append(logits.unsqueeze(1))

            # Teacher forcing
            if action_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = action_ids[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)


class SCANBenchmark:
    """Complete SCAN benchmark with training and evaluation."""

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2,
                 learning_rate: float = 1e-3, batch_size: int = 32):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Load data
        self.data = SCANDataLoader.load_scan_data()

        # Create vocabularies
        all_commands = [cmd.command for cmd in self.data['train'] + self.data['test']]
        all_actions = [cmd.actions for cmd in self.data['train'] + self.data['test']]

        self.vocab_command = SCANVocab(all_commands)
        self.vocab_action = SCANVocab(all_actions)

        # Create datasets
        self.train_dataset = SCANDataset(self.data['train'], self.vocab_command, self.vocab_action)
        self.test_dataset = SCANDataset(self.data['test'], self.vocab_command, self.vocab_action)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def create_model(self) -> SCANModel:
        """Create a fresh SCAN model."""
        return SCANModel(self.vocab_command, self.vocab_action,
                        self.embed_dim, self.hidden_dim, self.num_layers)

    def train_epoch(self, model: SCANModel, optimizer: optim.Optimizer,
                   criterion: nn.CrossEntropyLoss, device: str = 'cpu') -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in self.train_loader:
            command_ids = batch['command_ids'].to(device)
            action_ids = batch['action_ids'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(command_ids, action_ids, teacher_forcing_ratio=0.5)

            # Compute loss (ignore padding)
            loss = criterion(outputs.view(-1, len(self.vocab_action)),
                           action_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, model: SCANModel, device: str = 'cpu') -> Dict[str, float]:
        """Evaluate model on test set."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.test_loader:
                command_ids = batch['command_ids'].to(device)
                action_ids = batch['action_ids'].to(device)

                # Generate predictions
                outputs = model(command_ids, teacher_forcing_ratio=0.0)
                predictions = outputs.argmax(dim=-1)

                # Compare with targets (ignoring padding)
                for pred, target, pred_len in zip(predictions, action_ids, batch['action_len']):
                    pred_seq = pred[:pred_len].cpu().numpy()
                    target_seq = target[:pred_len].cpu().numpy()

                    if np.array_equal(pred_seq, target_seq):
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}

    def train_model(self, model: SCANModel, num_epochs: int = 100,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """Train model and return training history."""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_action.pad_idx)

        train_losses = []
        test_accuracies = []

        for epoch in tqdm(range(num_epochs), desc="Training SCAN"):
            # Train
            train_loss = self.train_epoch(model, optimizer, criterion, device)
            train_losses.append(train_loss)

            # Evaluate
            if epoch % 10 == 0:
                metrics = self.evaluate(model, device)
                test_accuracies.append(metrics['accuracy'])
                print(f"Epoch {epoch}: Loss={train_loss:.4f}, Test Acc={metrics['accuracy']:.4f}")

        return {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }


def run_scan_baseline(num_epochs: int = 100, device: str = 'cpu') -> Dict[str, float]:
    """Run SCAN benchmark with baseline LSTM model."""
    print("🏃 Running SCAN Baseline Benchmark...")

    benchmark = SCANBenchmark()

    # Create and train model
    model = benchmark.create_model()
    history = benchmark.train_model(model, num_epochs, device)

    # Final evaluation
    final_metrics = benchmark.evaluate(model, device)

    results = {
        'train_loss_final': history['train_losses'][-1],
        'test_accuracy': final_metrics['accuracy'],
        'test_correct': final_metrics['correct'],
        'test_total': final_metrics['total']
    }

    print(f"Baseline Loss: {results['train_loss_final']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.1%} ({results['test_correct']}/{results['test_total']})")

    return results


if __name__ == "__main__":
    # Test the benchmark
    results = run_scan_baseline(num_epochs=50)
    print(f"\nSCAN Baseline Results: {results}")
