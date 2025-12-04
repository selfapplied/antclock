"""
COGS Benchmark: Compositional Generalization in Semantic Parsing

Tests semantic parsing with systematic generalization to novel combinations.
Models learn natural language -> logical form mapping and must generalize.
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


class COGSExample:
    """Represents a COGS example with sentence and logical form."""

    def __init__(self, sentence: str, logical_form: str, split: str = 'train'):
        self.sentence = sentence.strip()
        self.logical_form = logical_form.strip()
        self.split = split

    def __str__(self):
        return f"{self.sentence} -> {self.logical_form}"


class COGSDataset(Dataset):
    """COGS dataset with train/dev/test splits."""

    def __init__(self, examples: List[COGSExample], vocab_sentence: 'COGSVocab',
                 vocab_lf: 'COGSVocab', max_len: int = 50):
        self.examples = examples
        self.vocab_sentence = vocab_sentence
        self.vocab_lf = vocab_lf
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize sentence
        sent_tokens = self.vocab_sentence.tokenize(example.sentence)
        sent_ids = [self.vocab_sentence.word2idx.get(token, self.vocab_sentence.unk_idx)
                   for token in sent_tokens]
        sent_ids = sent_ids[:self.max_len]
        sent_ids += [self.vocab_sentence.pad_idx] * (self.max_len - len(sent_ids))

        # Tokenize logical form
        lf_tokens = self.vocab_lf.tokenize(example.logical_form)
        lf_ids = [self.vocab_lf.word2idx.get(token, self.vocab_lf.unk_idx)
                 for token in lf_tokens]
        lf_ids = lf_ids[:self.max_len]
        lf_ids += [self.vocab_lf.pad_idx] * (self.max_len - len(lf_ids))

        return {
            'sentence_ids': torch.tensor(sent_ids, dtype=torch.long),
            'lf_ids': torch.tensor(lf_ids, dtype=torch.long),
            'sentence_len': len(sent_tokens),
            'lf_len': len(lf_tokens)
        }


class COGSVocab:
    """Vocabulary for COGS sentences and logical forms."""

    def __init__(self, texts: List[str]):
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

        # Build vocabulary
        all_words = set()
        for text in texts:
            all_words.update(self.tokenize(text))

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
        """Simple tokenization preserving logical form structure."""
        # Split on spaces and parentheses
        tokens = []
        current = ""
        for char in text:
            if char in ['(', ')', ' ']:
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


class COGSDataLoader:
    """Loads COGS dataset with different generalization splits."""

    @staticmethod
    def load_cogs_data() -> Dict[str, List[COGSExample]]:
        """Load COGS dataset with train/dev/test splits for different generalization types."""

        # Basic training examples
        train_examples = [
            # Simple transitive verbs
            COGSExample("Alex ate the cake", "(ate alex cake)", 'train'),
            COGSExample("Sam read the book", "(read sam book)", 'train'),
            COGSExample("Lee saw the dog", "(saw lee dog)", 'train'),
            COGSExample("Pat liked the movie", "(liked pat movie)", 'train'),

            # Intransitive verbs
            COGSExample("Alex smiled", "(smiled alex)", 'train'),
            COGSExample("Sam ran", "(ran sam)", 'train'),
            COGSExample("Lee danced", "(danced lee)", 'train'),

            # Adjectives
            COGSExample("Alex ate the red cake", "(ate alex (red cake))", 'train'),
            COGSExample("Sam read the big book", "(read sam (big book))", 'train'),
            COGSExample("Lee saw the small dog", "(saw lee (small dog))", 'train'),

            # Plurals
            COGSExample("Alex ate the cakes", "(ate alex cakes)", 'train'),
            COGSExample("Sam read the books", "(read sam books)", 'train'),

            # Passives
            COGSExample("The cake was eaten by Alex", "(ate alex cake)", 'train'),
            COGSExample("The book was read by Sam", "(read sam book)", 'train'),

            # Relative clauses
            COGSExample("Alex ate the cake that Sam baked", "(ate alex (that cake (baked sam)))", 'train'),
            COGSExample("Lee saw the dog that chased the cat", "(saw lee (that dog (chased dog cat)))", 'train'),
        ]

        # Development examples (in-distribution)
        dev_examples = [
            COGSExample("Pat watched the movie", "(watched pat movie)", 'dev'),
            COGSExample("Kim sang", "(sang kim)", 'dev'),
            COGSExample("Jordan ate the blue cake", "(ate jordan (blue cake))", 'dev'),
            COGSExample("The song was sung by Kim", "(sang kim song)", 'dev'),
        ]

        # Test examples for different generalization types
        test_examples = [
            # Primitive generalization (new verbs)
            COGSExample("Alex admired the painting", "(admired alex painting)", 'test'),
            COGSExample("Sam composed the music", "(composed sam music)", 'test'),

            # Lexical generalization (new nouns in familiar structures)
            COGSExample("Lee visited the museum", "(visited lee museum)", 'test'),
            COGSExample("Pat bought the sculpture", "(bought pat sculpture)", 'test'),

            # Structural generalization (new syntactic structures)
            COGSExample("Alex gave the cake to Sam", "(gave alex cake sam)", 'test'),
            COGSExample("Lee showed the book to Pat", "(showed lee book pat)", 'test'),

            # Template generalization (new combinations)
            COGSExample("The painting was admired by Alex", "(admired alex painting)", 'test'),
            COGSExample("Alex ate the cake that Pat baked", "(ate alex (that cake (baked pat)))", 'test'),
            COGSExample("Sam composed the music that Lee liked", "(composed sam (that music (liked lee)))", 'test'),

            # Complex nested structures
            COGSExample("Alex gave the cake that Sam baked to Lee", "(gave alex (that cake (baked sam)) lee)", 'test'),
            COGSExample("Pat saw the dog that chased the cat that Kim owned", "(saw pat (that dog (chased dog (that cat (owned kim)))))", 'test'),
        ]

        # Try to load real COGS dataset first
        try:
            train_examples = []
            dev_examples = []
            test_examples = []

            # Load the real COGS data
            data_dir = 'benchmarks/real_data/COGS-main/data'

            # Parse training data (use train.tsv)
            with open(f'{data_dir}/train.tsv', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            sentence = parts[0]
                            logical_form = parts[1]
                            train_examples.append(COGSExample(sentence, logical_form, 'train'))

            # Parse development data
            with open(f'{data_dir}/dev.tsv', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            sentence = parts[0]
                            logical_form = parts[1]
                            dev_examples.append(COGSExample(sentence, logical_form, 'dev'))

            # Parse test data
            with open(f'{data_dir}/test.tsv', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            sentence = parts[0]
                            logical_form = parts[1]
                            test_examples.append(COGSExample(sentence, logical_form, 'test'))

            print(f"Loaded REAL COGS dataset: {len(train_examples)} train, {len(dev_examples)} dev, {len(test_examples)} test")
            return {
                'train': train_examples,
                'dev': dev_examples,
                'test': test_examples
            }

        except FileNotFoundError:
            # Fallback to synthetic dataset
            print("Real COGS dataset not found, using synthetic dataset")
            return {
                'train': train_examples,
                'dev': dev_examples,
                'test': test_examples
            }


class COGSModel(nn.Module):
    """Sequence-to-sequence model for COGS semantic parsing."""

    def __init__(self, vocab_sentence: COGSVocab, vocab_lf: COGSVocab,
                 embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()

        self.vocab_sentence = vocab_sentence
        self.vocab_lf = vocab_lf

        # Encoder (bidirectional LSTM for better semantic understanding)
        self.encoder_embed = nn.Embedding(len(vocab_sentence), embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=True)

        # Decoder (matches bidirectional encoder output size)
        self.decoder_embed = nn.Embedding(len(vocab_lf), embed_dim)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, len(vocab_lf))

        # Context projection (from encoder hidden_dim*2 to decoder hidden_dim)
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def encode(self, sentence_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Encode sentence with bidirectional LSTM."""
        embeds = self.encoder_embed(sentence_ids)
        outputs, (hidden, cell) = self.encoder(embeds)

        # Reshape bidirectional hidden states for decoder
        # hidden/cell shape: [num_layers * 2, batch, hidden_dim]
        # We need: [num_layers, batch, hidden_dim] for unidirectional decoder
        num_layers = self.encoder.num_layers
        hidden_fwd = hidden[::2]  # Forward directions
        hidden_bwd = hidden[1::2]  # Backward directions
        hidden = (hidden_fwd + hidden_bwd) / 2  # Average forward and backward

        cell_fwd = cell[::2]
        cell_bwd = cell[1::2]
        cell = (cell_fwd + cell_bwd) / 2

        return outputs, (hidden, cell)

    def decode_step(self, input_token: torch.Tensor, hidden: Tuple,
                   encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Single decoder step with simple encoder context."""
        embed = self.decoder_embed(input_token.unsqueeze(1))
        output, hidden = self.decoder(embed, hidden)

        # Simple context: average encoder outputs
        context = encoder_outputs.mean(dim=1)  # [batch, hidden_dim*2]

        # Project context to decoder dimension
        context_proj = self.context_proj(context)  # [batch, hidden_dim]
        combined = output.squeeze(1) + context_proj  # Residual connection

        logits = self.output_proj(combined)

        return logits, hidden

    def forward(self, sentence_ids: torch.Tensor, lf_ids: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """Forward pass for training."""
        batch_size = sentence_ids.size(0)
        max_len = lf_ids.size(1) if lf_ids is not None else 50

        # Encode
        encoder_outputs, (hidden, cell) = self.encode(sentence_ids)

        # Decode
        outputs = []
        input_token = torch.full((batch_size,), self.vocab_lf.sos_idx,
                               device=sentence_ids.device)

        for t in range(max_len):
            logits, (hidden, cell) = self.decode_step(input_token, (hidden, cell), encoder_outputs)
            outputs.append(logits.unsqueeze(1))

            # Teacher forcing
            if lf_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = lf_ids[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)


class COGSBenchmark:
    """Complete COGS benchmark with training and evaluation."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2,
                 learning_rate: float = 1e-3, batch_size: int = 32):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Load data
        self.data = COGSDataLoader.load_cogs_data()

        # Create vocabularies
        all_sentences = [ex.sentence for ex in self.data['train'] + self.data['dev'] + self.data['test']]
        all_lfs = [ex.logical_form for ex in self.data['train'] + self.data['dev'] + self.data['test']]

        self.vocab_sentence = COGSVocab(all_sentences)
        self.vocab_lf = COGSVocab(all_lfs)

        # Create datasets
        self.train_dataset = COGSDataset(self.data['train'], self.vocab_sentence, self.vocab_lf)
        self.dev_dataset = COGSDataset(self.data['dev'], self.vocab_sentence, self.vocab_lf)
        self.test_dataset = COGSDataset(self.data['test'], self.vocab_sentence, self.vocab_lf)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def create_model(self) -> COGSModel:
        """Create a fresh COGS model."""
        return COGSModel(self.vocab_sentence, self.vocab_lf,
                        self.embed_dim, self.hidden_dim, self.num_layers)

    def train_epoch(self, model: COGSModel, optimizer: optim.Optimizer,
                   criterion: nn.CrossEntropyLoss, device: str = 'cpu') -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in self.train_loader:
            sentence_ids = batch['sentence_ids'].to(device)
            lf_ids = batch['lf_ids'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(sentence_ids, lf_ids, teacher_forcing_ratio=0.5)

            # Compute loss (ignore padding)
            loss = criterion(outputs.view(-1, len(self.vocab_lf)),
                           lf_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, model: COGSModel, dataset_name: str = 'test',
                device: str = 'cpu') -> Dict[str, float]:
        """Evaluate model on specified dataset."""
        model.eval()
        correct = 0
        total = 0

        loader = getattr(self, f'{dataset_name}_loader')

        with torch.no_grad():
            for batch in loader:
                sentence_ids = batch['sentence_ids'].to(device)
                lf_ids = batch['lf_ids'].to(device)

                # Generate predictions
                outputs = model(sentence_ids, teacher_forcing_ratio=0.0)
                predictions = outputs.argmax(dim=-1)

                # Compare with targets (ignoring padding and EOS)
                for pred, target, pred_len in zip(predictions, lf_ids, batch['lf_len']):
                    pred_seq = pred[:pred_len].cpu().numpy()
                    target_seq = target[:pred_len].cpu().numpy()

                    if np.array_equal(pred_seq, target_seq):
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}

    def train_model(self, model: COGSModel, num_epochs: int = 100,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """Train model and return training history."""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_lf.pad_idx)

        train_losses = []
        dev_accuracies = []

        for epoch in tqdm(range(num_epochs), desc="Training COGS"):
            # Train
            train_loss = self.train_epoch(model, optimizer, criterion, device)
            train_losses.append(train_loss)

            # Evaluate on dev set
            if epoch % 10 == 0:
                dev_metrics = self.evaluate(model, 'dev', device)
                dev_accuracies.append(dev_metrics['accuracy'])
                print(f"Epoch {epoch}: Loss={train_loss:.4f}, Dev Acc={dev_metrics['accuracy']:.4f}")

        return {
            'train_losses': train_losses,
            'dev_accuracies': dev_accuracies
        }


def run_cogs_baseline(num_epochs: int = 100, device: str = 'cpu') -> Dict[str, float]:
    """Run COGS benchmark with baseline LSTM model."""
    print("🏃 Running COGS Baseline Benchmark...")

    benchmark = COGSBenchmark()

    # Create and train model
    model = benchmark.create_model()
    history = benchmark.train_model(model, num_epochs, device)

    # Final evaluation
    dev_metrics = benchmark.evaluate(model, 'dev', device)
    test_metrics = benchmark.evaluate(model, 'test', device)

    results = {
        'train_loss_final': history['train_losses'][-1],
        'dev_accuracy': dev_metrics['accuracy'],
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
    results = run_cogs_baseline(num_epochs=50)
    print(f"\nCOGS Baseline Results: {results}")
