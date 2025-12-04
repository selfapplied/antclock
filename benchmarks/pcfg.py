"""
PCFG Benchmark: Probabilistic Context-Free Grammar

Tests compositional generalization in language understanding through
probabilistic context-free grammars with systematic structural changes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import json
import os
from tqdm import tqdm
import random


class PCFGExample:
    """Represents a PCFG example with sentence and parse tree."""

    def __init__(self, sentence: str, parse_tree: str, split: str = 'train'):
        self.sentence = sentence.strip()
        self.parse_tree = parse_tree.strip()
        self.split = split

    def __str__(self):
        return f"{self.sentence} -> {self.parse_tree}"


class PCFGDataset(Dataset):
    """PCFG dataset with train/dev/test splits."""

    def __init__(self, examples: List[PCFGExample], vocab_sentence: 'PCFGVocab',
                 vocab_tree: 'PCFGVocab', max_len: int = 50):
        self.examples = examples
        self.vocab_sentence = vocab_sentence
        self.vocab_tree = vocab_tree
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

        # Tokenize parse tree
        tree_tokens = self.vocab_tree.tokenize(example.parse_tree)
        tree_ids = [self.vocab_tree.word2idx.get(token, self.vocab_tree.unk_idx)
                   for token in tree_tokens]
        tree_ids = tree_ids[:self.max_len]
        tree_ids += [self.vocab_tree.pad_idx] * (self.max_len - len(tree_ids))

        return {
            'sentence_ids': torch.tensor(sent_ids, dtype=torch.long),
            'tree_ids': torch.tensor(tree_ids, dtype=torch.long),
            'sentence_len': len(sent_tokens),
            'tree_len': len(tree_tokens)
        }


class PCFGVocab:
    """Vocabulary for PCFG sentences and parse trees."""

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
        """Tokenize preserving tree structure."""
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


class PCFGGrammar:
    """Simple PCFG grammar for generating systematic generalization tasks."""

    def __init__(self):
        # Basic grammar rules
        self.terminals = ['the', 'cat', 'dog', 'runs', 'jumps', 'quickly', 'slowly', 'big', 'small']
        self.non_terminals = ['S', 'NP', 'VP', 'PP', 'ADJ', 'ADV']

        # Primitive rules (always trained)
        self.primitive_rules = {
            'NP': ['the cat', 'the dog'],
            'VP': ['runs', 'jumps'],
            'ADJ': ['big', 'small'],
            'ADV': ['quickly', 'slowly']
        }

        # Compound rules (for generalization)
        self.compound_rules = {
            'NP': ['the big cat', 'the small dog', 'the quick cat', 'the lazy dog'],
            'VP': ['runs quickly', 'jumps slowly', 'runs fast', 'jumps high'],
            'ADJ': ['big red', 'small blue', 'quick brown', 'lazy black'],
            'ADV': ['quickly and', 'slowly but', 'fast then', 'high when']
        }

    def generate_examples(self, num_train: int = 100, num_test: int = 50) -> Dict[str, List[PCFGExample]]:
        """Generate PCFG examples with systematic generalization splits."""

        examples = {'train': [], 'dev': [], 'test': []}

        # Training examples: only primitive rules
        for _ in range(num_train):
            sentence, tree = self.generate_primitive_sentence()
            examples['train'].append(PCFGExample(sentence, tree, 'train'))

        # Development examples: mix of primitive and compound
        for _ in range(num_test // 2):
            sentence, tree = self.generate_primitive_sentence()
            examples['dev'].append(PCFGExample(sentence, tree, 'dev'))

        for _ in range(num_test // 2):
            sentence, tree = self.generate_compound_sentence()
            examples['dev'].append(PCFGExample(sentence, tree, 'dev'))

        # Test examples: only compound rules (systematic generalization)
        for _ in range(num_test):
            sentence, tree = self.generate_compound_sentence()
            examples['test'].append(PCFGExample(sentence, tree, 'test'))

        return examples

    def generate_primitive_sentence(self) -> Tuple[str, str]:
        """Generate sentence using only primitive rules."""
        # Simple S -> NP VP structure
        np = random.choice(self.primitive_rules['NP'])
        vp = random.choice(self.primitive_rules['VP'])

        sentence = f"{np} {vp}"
        tree = f"(S (NP {np}) (VP {vp}))"

        return sentence, tree

    def generate_compound_sentence(self) -> Tuple[str, str]:
        """Generate sentence using compound rules (for generalization)."""
        # More complex structures
        structures = [
            # NP with adjectives
            lambda: self._gen_np_with_adj(),
            # VP with adverbs
            lambda: self._gen_vp_with_adv(),
            # Complex NP + VP
            lambda: self._gen_complex_sentence(),
        ]

        sentence, tree = random.choice(structures)()
        return sentence, tree

    def _gen_np_with_adj(self) -> Tuple[str, str]:
        """Generate NP with adjectives."""
        adj = random.choice(self.compound_rules['ADJ'])
        noun = random.choice(['cat', 'dog'])
        sentence = f"the {adj} {noun}"
        tree = f"(NP (ADJ {adj}) {noun})"
        return sentence, tree

    def _gen_vp_with_adv(self) -> Tuple[str, str]:
        """Generate VP with adverbs."""
        verb = random.choice(['runs', 'jumps'])
        adv = random.choice(self.compound_rules['ADV'])
        sentence = f"{verb} {adv} fast"
        tree = f"(VP {verb} (ADV {adv} fast))"
        return sentence, tree

    def _gen_complex_sentence(self) -> Tuple[str, str]:
        """Generate more complex sentence structure."""
        adj = random.choice(self.compound_rules['ADJ'])
        noun = random.choice(['cat', 'dog'])
        verb = random.choice(['runs', 'jumps'])
        adv = random.choice(['quickly', 'slowly'])

        sentence = f"the {adj} {noun} {verb} {adv}"
        tree = f"(S (NP (ADJ {adj}) {noun}) (VP {verb} (ADV {adv})))"

        return sentence, tree


class PCFGModel(nn.Module):
    """Simple sequence-to-sequence model for PCFG parsing."""

    def __init__(self, vocab_sentence: PCFGVocab, vocab_tree: PCFGVocab,
                 embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        self.vocab_sentence = vocab_sentence
        self.vocab_tree = vocab_tree

        # Simple encoder-decoder
        self.encoder_embed = nn.Embedding(len(vocab_sentence), embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.decoder_embed = nn.Embedding(len(vocab_tree), embed_dim)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, len(vocab_tree))

    def encode(self, sentence_ids: torch.Tensor) -> torch.Tensor:
        """Encode sentence."""
        embeds = self.encoder_embed(sentence_ids)
        _, (hidden, _) = self.encoder(embeds)
        return hidden[-1]  # Use last layer hidden state

    def forward(self, sentence_ids: torch.Tensor, tree_ids: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """Forward pass."""
        batch_size = sentence_ids.size(0)
        max_len = tree_ids.size(1) if tree_ids is not None else 50

        # Encode
        hidden = self.encode(sentence_ids)

        # Decode
        outputs = []
        input_token = torch.full((batch_size, 1), self.vocab_tree.sos_idx,
                               device=sentence_ids.device)

        current_hidden = hidden.unsqueeze(0)  # Add layer dimension

        for t in range(max_len):
            embed = self.decoder_embed(input_token)
            output, current_hidden = self.decoder(embed, current_hidden)
            logits = self.output_proj(output.squeeze(1))
            outputs.append(logits.unsqueeze(1))

            # Teacher forcing
            if tree_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tree_ids[:, t:t+1]
            else:
                input_token = logits.argmax(dim=-1, keepdim=True)

        return torch.cat(outputs, dim=1)


class PCFGBenchmark:
    """Complete PCFG benchmark with training and evaluation."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2,
                 learning_rate: float = 1e-3, batch_size: int = 32):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Generate data
        grammar = PCFGGrammar()
        self.data = grammar.generate_examples(num_train=200, num_test=50)

        # Create vocabularies
        all_sentences = [ex.sentence for ex in self.data['train'] + self.data['dev'] + self.data['test']]
        all_trees = [ex.parse_tree for ex in self.data['train'] + self.data['dev'] + self.data['test']]

        self.vocab_sentence = PCFGVocab(all_sentences)
        self.vocab_tree = PCFGVocab(all_trees)

        # Create datasets
        self.train_dataset = PCFGDataset(self.data['train'], self.vocab_sentence, self.vocab_tree)
        self.dev_dataset = PCFGDataset(self.data['dev'], self.vocab_sentence, self.vocab_tree)
        self.test_dataset = PCFGDataset(self.data['test'], self.vocab_sentence, self.vocab_tree)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def create_model(self) -> PCFGModel:
        """Create a fresh PCFG model."""
        return PCFGModel(self.vocab_sentence, self.vocab_tree,
                        self.embed_dim, self.hidden_dim)

    def train_epoch(self, model: PCFGModel, optimizer: optim.Optimizer,
                   criterion: nn.CrossEntropyLoss, device: str = 'cpu') -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in self.train_loader:
            sentence_ids = batch['sentence_ids'].to(device)
            tree_ids = batch['tree_ids'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(sentence_ids, tree_ids, teacher_forcing_ratio=0.5)

            # Compute loss (ignore padding)
            loss = criterion(outputs.view(-1, len(self.vocab_tree)),
                           tree_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, model: PCFGModel, dataset_name: str = 'test',
                device: str = 'cpu') -> Dict[str, float]:
        """Evaluate model on specified dataset."""
        model.eval()
        correct = 0
        total = 0

        loader = getattr(self, f'{dataset_name}_loader')

        with torch.no_grad():
            for batch in loader:
                sentence_ids = batch['sentence_ids'].to(device)
                tree_ids = batch['tree_ids'].to(device)

                # Generate predictions
                outputs = model(sentence_ids, teacher_forcing_ratio=0.0)
                predictions = outputs.argmax(dim=-1)

                # Compare with targets (ignoring padding)
                for pred, target, pred_len in zip(predictions, tree_ids, batch['tree_len']):
                    pred_seq = pred[:pred_len].cpu().numpy()
                    target_seq = target[:pred_len].cpu().numpy()

                    if np.array_equal(pred_seq, target_seq):
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}

    def train_model(self, model: PCFGBenchmark, num_epochs: int = 100,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """Train model and return training history."""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_tree.pad_idx)

        train_losses = []
        dev_accuracies = []

        for epoch in tqdm(range(num_epochs), desc="Training PCFG"):
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


def run_pcfg_baseline(num_epochs: int = 50, device: str = 'cpu') -> Dict[str, float]:
    """Run PCFG benchmark with baseline LSTM model."""
    print("🏃 Running PCFG Baseline Benchmark...")

    benchmark = PCFGBenchmark()

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
    results = run_pcfg_baseline(num_epochs=30)
    print(f"\nPCFG Baseline Results: {results}")
