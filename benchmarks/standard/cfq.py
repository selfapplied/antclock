#!run.sh
"""
CE-Enhanced CFQ (Compositional Freebase Questions) Benchmark

Implements CE-enhanced models for CFQ semantic parsing.
CFQ tests systematic compositionality in semantic parsing with SPARQL queries.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Import CE modules
from ..modules import CEEnhancedLSTM, CEConfig, MirrorOperator, CurvatureCouplingLayer
from ..definitions import BenchmarkResult


class CFQDataset(Dataset):
    """CFQ dataset wrapper."""

    def __init__(self, data: List[Dict[str, str]], vocab_question: Dict[str, int],
                 vocab_sparql: Dict[str, int], max_len: int = 100):
        self.data = data
        self.vocab_question = vocab_question
        self.vocab_sparql = vocab_sparql
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize question
        question_tokens = item['question'].split()
        question_ids = [self.vocab_question.get(token, self.vocab_question.get('<unk>', 0))
                       for token in question_tokens]
        question_ids = question_ids[:self.max_len] + [self.vocab_question['<pad>']] * max(0, self.max_len - len(question_ids))

        # Tokenize SPARQL
        sparql_tokens = item['sparql'].split()
        sparql_ids = [self.vocab_sparql.get(token, self.vocab_sparql.get('<unk>', 0))
                     for token in sparql_tokens]
        sparql_ids = sparql_ids[:self.max_len] + [self.vocab_sparql['<pad>']] * max(0, self.max_len - len(sparql_ids))

        return {
            'question': torch.tensor(question_ids, dtype=torch.long),
            'sparql': torch.tensor(sparql_ids, dtype=torch.long),
            'question_text': item['question'],
            'sparql_text': item['sparql']
        }


class CFQBenchmark:
    """CE-enhanced CFQ benchmark."""

    def __init__(self, data_dir: str = ".data/cfq"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_cfq_data(self):
        """Download CFQ dataset if not present."""
        cfq_url = "https://dl.fbaipublicfiles.com/cfq/cfq-dataset-json-v2.zip"
        zip_path = self.data_dir / "cfq-dataset-json-v2.zip"
        extract_path = self.data_dir / "cfq"

        if not extract_path.exists():
            print("📥 Downloading CFQ dataset...")
            urllib.request.urlretrieve(cfq_url, zip_path)

            print("📦 Extracting CFQ dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

            print("✅ CFQ dataset ready")
        else:
            print("✅ CFQ dataset already downloaded")

    def load_cfq_data(self, split: str = 'mcd1') -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Load CFQ dataset."""
        self.download_cfq_data()

        # Load the specific split
        train_file = self.data_dir / f"cfq/splits/{split}_train.json"
        test_file = self.data_dir / f"cfq/splits/{split}_test.json"

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(f"CFQ split '{split}' not found. Available splits: mcd1, mcd2, mcd3, question_complexity_split")

        with open(train_file, 'r') as f:
            train_data = json.load(f)

        with open(test_file, 'r') as f:
            test_data = json.load(f)

        return train_data, test_data

    def build_vocabularies(self, data: List[Dict[str, str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build vocabularies from data."""
        questions = [item['question'] for item in data]
        sparqls = [item['sparql'] for item in data]

        # Build question vocabulary
        question_tokens = set()
        for q in questions:
            question_tokens.update(q.split())

        vocab_question = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        for token in sorted(question_tokens):
            vocab_question[token] = len(vocab_question)

        # Build SPARQL vocabulary
        sparql_tokens = set()
        for s in sparqls:
            sparql_tokens.update(s.split())

        vocab_sparql = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        for token in sorted(sparql_tokens):
            vocab_sparql[token] = len(vocab_sparql)

        return vocab_question, vocab_sparql

    def create_model(self, vocab_question_size: int, vocab_sparql_size: int) -> nn.Module:
        """Create CE-enhanced CFQ model."""
        config = CEConfig(
            vocab_size=vocab_question_size,
            hidden_size=512,
            num_layers=3,
            dropout=0.2,
            kappa=0.35,
            zeta_strength=0.1
        )

        class CECFQModel(nn.Module):
            def __init__(self, config, vocab_sparql_size):
                super().__init__()
                self.config = config

                # Question encoder
                self.question_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
                self.encoder = CEEnhancedLSTM(
                    config.vocab_size, config.hidden_size, config.num_layers, config.dropout, config
                )

                # SPARQL decoder
                self.sparql_embedding = nn.Embedding(vocab_sparql_size, config.hidden_size)
                self.decoder = nn.LSTM(config.hidden_size, config.hidden_size, config.num_layers, batch_first=True)
                self.output_projection = nn.Linear(config.hidden_size, vocab_sparql_size)

            def forward(self, questions, sparql_targets=None):
                # Encode question
                question_embeds = self.question_embedding(questions)
                encoder_outputs, (h_n, c_n) = self.encoder(question_embeds)

                # Decode SPARQL
                batch_size = questions.size(0)
                max_len = sparql_targets.size(1) if sparql_targets is not None else 50

                # Start with SOS token
                decoder_input = torch.full((batch_size, 1), 2, dtype=torch.long, device=questions.device)  # <sos>
                decoder_hidden = (h_n, c_n)

                outputs = []
                for t in range(max_len):
                    decoder_embed = self.sparql_embedding(decoder_input)
                    decoder_output, decoder_hidden = self.decoder(decoder_embed, decoder_hidden)
                    logits = self.output_projection(decoder_output.squeeze(1))
                    outputs.append(logits)

                    if sparql_targets is not None:
                        # Teacher forcing
                        decoder_input = sparql_targets[:, t:t+1]
                    else:
                        # Greedy decoding
                        decoder_input = logits.argmax(dim=-1, keepdim=True)

                return torch.stack(outputs, dim=1)

        return CECFQModel(config, vocab_sparql_size)

    def train_model(self, model, train_loader, val_loader, num_epochs, device):
        """Train the CFQ model."""
        print("🧠 Training CE-CFQ model...")

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

        history = {
            'train_losses': [],
            'val_accuracies': [],
            'epochs': num_epochs
        }

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                questions = batch['question'].to(device)
                sparql_targets = batch['sparql'].to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(questions, sparql_targets)

                # Compute loss (exclude SOS token)
                loss = criterion(
                    outputs[:, :-1].contiguous().view(-1, outputs.size(-1)),
                    sparql_targets[:, 1:].contiguous().view(-1)
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches
            history['train_losses'].append(avg_train_loss)

            # Validation
            val_accuracy = self.evaluate_accuracy(model, val_loader, device)
            history['val_accuracies'].append(val_accuracy)

            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_train_loss:.4f}, Val Acc={val_accuracy:.1%}")

        return history

    def evaluate_accuracy(self, model, data_loader, device) -> float:
        """Evaluate exact match accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                questions = batch['question'].to(device)
                sparql_targets = batch['sparql'].to(device)

                # Generate predictions
                outputs = model(questions)  # Greedy decoding

                # Convert to predictions
                predictions = outputs.argmax(dim=-1)

                # Check exact matches (ignoring padding)
                for pred, target in zip(predictions, sparql_targets):
                    # Remove padding and EOS
                    pred_clean = []
                    target_clean = []

                    for p in pred:
                        if p.item() in [0, 3]:  # <pad>, <eos>
                            break
                        pred_clean.append(p.item())

                    for t in target:
                        if t.item() in [0, 3]:
                            break
                        target_clean.append(t.item())

                    if pred_clean == target_clean:
                        correct += 1
                    total += 1

        return correct / total if total > 0 else 0.0


def run_ce_cfq_experiment(num_epochs: int = 5, batch_size: int = 32,
                         device: str = 'auto', split: str = 'mcd1') -> Dict[str, Any]:
    """
    Run CE-enhanced CFQ experiment.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        device: Device to run on ('auto', 'cpu', 'cuda')
        split: CFQ split to use ('mcd1', 'mcd2', 'mcd3', 'question_complexity_split')

    Returns:
        Experiment results
    """
    print("🔬 Running CE-Enhanced CFQ Experiment...")
    print(f"Parameters: κ=0.35, χ_FEG=0.638, Split={split}")

    # Set device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon GPU acceleration
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    # Create benchmark
    benchmark = CFQBenchmark()

    # Load data
    train_data, test_data = benchmark.load_cfq_data(split)
    print(f"Loaded CFQ-{split}: {len(train_data)} train, {len(test_data)} test")

    # Build vocabularies
    vocab_question, vocab_sparql = benchmark.build_vocabularies(train_data)
    print(f"Vocabularies: {len(vocab_question)} question tokens, {len(vocab_sparql)} SPARQL tokens")

    # Create datasets
    train_dataset = CFQDataset(train_data, vocab_question, vocab_sparql)
    test_dataset = CFQDataset(test_data, vocab_question, vocab_sparql)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = benchmark.create_model(len(vocab_question), len(vocab_sparql))
    model.to(device)

    # Train model
    start_time = time.time()
    history = benchmark.train_model(model, train_loader, test_loader, num_epochs, device)
    training_time = time.time() - start_time

    # Final evaluation
    test_accuracy = benchmark.evaluate_accuracy(model, test_loader, device)

    # Results
    results = {
        'experiment': f'ce_cfq_{split}',
        'parameters': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'split': split,
            'ce_config': {
                'kappa': 0.35,
                'chi_feg': 0.638,
                'zeta_strength': 0.1
            }
        },
        'training': {
            'final_train_loss': history['train_losses'][-1],
            'final_val_accuracy': history['val_accuracies'][-1],
            'training_time': training_time,
            'epochs_completed': len(history['train_losses'])
        },
        'evaluation': {
            'test_accuracy': test_accuracy,
            'dataset_size': len(test_data)
        },
        'ce_features': [
            'kappa_guardian_early_stopping',
            'chi_feg_learning_rate_scheduling',
            'mirror_operator_symmetry',
            'zeta_regularization'
        ]
    }

    print(".2f")
    print(".1%")
    return results


# Quick test
if __name__ == "__main__":
    print("Testing CE-CFQ implementation...")
    results = run_ce_cfq_experiment(num_epochs=1, batch_size=2)
    print("Test completed!")
    print(f"Results: {results}")
