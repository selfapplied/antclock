"""
CFQ Benchmark: Compositional Freebase Questions

Tests semantic parsing compositionality by training on questions with
specific structural patterns and testing on novel combinations.
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


class CFQExample:
    """Represents a CFQ example with question and SPARQL query."""

    def __init__(self, question: str, sparql: str, split: str = 'train'):
        self.question = question.strip()
        self.sparql = sparql.strip()
        self.split = split

    def __str__(self):
        return f"{self.question} -> {self.sparql}"


class CFQDataset(Dataset):
    """CFQ dataset with train/dev/test splits."""

    def __init__(self, examples: List[CFQExample], vocab_question: 'CFQVocab',
                 vocab_sparql: 'CFQVocab', max_len: int = 100):
        self.examples = examples
        self.vocab_question = vocab_question
        self.vocab_sparql = vocab_sparql
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize question
        q_tokens = self.vocab_question.tokenize(example.question)
        q_ids = [self.vocab_question.word2idx.get(token, self.vocab_question.unk_idx)
                for token in q_tokens]
        q_ids = q_ids[:self.max_len]
        q_ids += [self.vocab_question.pad_idx] * (self.max_len - len(q_ids))

        # Tokenize SPARQL
        sparql_tokens = self.vocab_sparql.tokenize(example.sparql)
        sparql_ids = [self.vocab_sparql.word2idx.get(token, self.vocab_sparql.unk_idx)
                     for token in sparql_tokens]
        sparql_ids = sparql_ids[:self.max_len]
        sparql_ids += [self.vocab_sparql.pad_idx] * (self.max_len - len(sparql_ids))

        return {
            'question_ids': torch.tensor(q_ids, dtype=torch.long),
            'sparql_ids': torch.tensor(sparql_ids, dtype=torch.long),
            'question_len': len(q_tokens),
            'sparql_len': len(sparql_tokens)
        }


class CFQVocab:
    """Vocabulary for CFQ questions and SPARQL queries."""

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
        """Tokenize preserving SPARQL structure."""
        # Split on whitespace and punctuation
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def __len__(self):
        return len(self.word2idx)


class CFQGenerator:
    """Generates CFQ-style compositional questions and SPARQL queries."""

    def __init__(self):
        # Entities and relations for the knowledge base
        self.entities = {
            'person': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'city': ['Paris', 'London', 'Berlin', 'Tokyo', 'New York'],
            'country': ['France', 'England', 'Germany', 'Japan', 'USA'],
            'company': ['Google', 'Apple', 'Microsoft', 'Amazon', 'Tesla']
        }

        self.relations = {
            'born_in': 'birthPlace',
            'works_for': 'employer',
            'lives_in': 'residence',
            'located_in': 'location',
            'founded_by': 'founder',
            'headquartered_in': 'headquarters'
        }

        # Primitive query templates (always in training)
        self.primitive_templates = [
            {
                'question': "Where was {person} born?",
                'sparql': "SELECT ?x WHERE {{ {person_uri} <http://www.wikidata.org/prop/direct/P19> ?x }}"
            },
            {
                'question': "Where does {person} live?",
                'sparql': "SELECT ?x WHERE {{ {person_uri} <http://www.wikidata.org/prop/direct/P551> ?x }}"
            },
            {
                'question': "What company does {person} work for?",
                'sparql': "SELECT ?x WHERE {{ {person_uri} <http://www.wikidata.org/prop/direct/P108> ?x }}"
            }
        ]

        # Compound query templates (for generalization)
        self.compound_templates = [
            {
                'question': "Where was the person born who works for {company}?",
                'sparql': "SELECT ?x WHERE {{ ?p <http://www.wikidata.org/prop/direct/P108> {company_uri} . ?p <http://www.wikidata.org/prop/direct/P19> ?x }}"
            },
            {
                'question': "What company does the person work for who lives in {city}?",
                'sparql': "SELECT ?x WHERE {{ ?p <http://www.wikidata.org/prop/direct/P551> {city_uri} . ?p <http://www.wikidata.org/prop/direct/P108> ?x }}"
            },
            {
                'question': "Where does the person live who was born in {city}?",
                'sparql': "SELECT ?x WHERE {{ ?p <http://www.wikidata.org/prop/direct/P19> {city_uri} . ?p <http://www.wikidata.org/prop/direct/P551> ?x }}"
            },
            {
                'question': "What is the birthplace of the founder of {company}?",
                'sparql': "SELECT ?x WHERE {{ {company_uri} <http://www.wikidata.org/prop/direct/P112> ?p . ?p <http://www.wikidata.org/prop/direct/P19> ?x }}"
            }
        ]

    def generate_examples(self, num_train: int = 200, num_test: int = 50) -> Dict[str, List[CFQExample]]:
        """Generate CFQ examples with systematic generalization splits."""

        examples = {'train': [], 'dev': [], 'test': []}

        # Training examples: only primitive templates with limited entities
        train_entities = {k: v[:3] for k, v in self.entities.items()}  # First 3 of each type

        for _ in range(num_train):
            template = random.choice(self.primitive_templates)
            example = self.fill_template(template, train_entities)
            examples['train'].append(CFQExample(example['question'], example['sparql'], 'train'))

        # Development examples: mix of primitive and compound with train entities
        for _ in range(num_test // 2):
            template = random.choice(self.primitive_templates)
            example = self.fill_template(template, train_entities)
            examples['dev'].append(CFQExample(example['question'], example['sparql'], 'dev'))

        for _ in range(num_test // 2):
            template = random.choice(self.compound_templates)
            example = self.fill_template(template, train_entities)
            examples['dev'].append(CFQExample(example['question'], example['sparql'], 'dev'))

        # Test examples: compound templates with novel entity combinations
        test_entities = self.entities  # All entities including novel ones

        for _ in range(num_test):
            template = random.choice(self.compound_templates)
            example = self.fill_template(template, test_entities)
            examples['test'].append(CFQExample(example['question'], example['sparql'], 'test'))

        return examples

    def fill_template(self, template: Dict[str, str], entities: Dict[str, List[str]]) -> Dict[str, str]:
        """Fill a template with random entities."""

        # Select random entities
        person = random.choice(entities['person'])
        city = random.choice(entities['city'])
        company = random.choice(entities['company'])

        # Create URIs (simplified)
        person_uri = f"wd:Q{random.randint(1000, 9999)}"  # Mock Wikidata QID
        city_uri = f"wd:Q{random.randint(1000, 9999)}"
        company_uri = f"wd:Q{random.randint(1000, 9999)}"

        # Fill template
        question = template['question'].format(
            person=person, city=city, company=company
        )

        sparql = template['sparql'].format(
            person_uri=person_uri,
            city_uri=city_uri,
            company_uri=company_uri
        )

        return {'question': question, 'sparql': sparql}


class CFQModel(nn.Module):
    """Sequence-to-sequence model for CFQ semantic parsing."""

    def __init__(self, vocab_question: CFQVocab, vocab_sparql: CFQVocab,
                 embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()

        self.vocab_question = vocab_question
        self.vocab_sparql = vocab_sparql

        # Encoder
        self.encoder_embed = nn.Embedding(len(vocab_question), embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

        # Decoder
        self.decoder_embed = nn.Embedding(len(vocab_sparql), embed_dim)
        self.decoder = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, len(vocab_sparql))

        # Attention
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)

    def encode(self, question_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode question."""
        embeds = self.encoder_embed(question_ids)
        outputs, (hidden, cell) = self.encoder(embeds)
        return outputs, hidden[-1]  # Use last layer hidden state

    def decode_step(self, input_token: torch.Tensor, hidden: torch.Tensor,
                   encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single decoder step with attention."""
        embed = self.decoder_embed(input_token.unsqueeze(1))

        # Attention mechanism
        attn_weights = torch.softmax(
            torch.bmm(hidden.unsqueeze(1), encoder_outputs.transpose(1, 2)),
            dim=-1
        )
        context = torch.bmm(attn_weights, encoder_outputs)

        # Combine embedding and context
        decoder_input = torch.cat([embed, context], dim=-1)

        output, hidden = self.decoder(decoder_input, hidden.unsqueeze(0))
        logits = self.output_proj(output.squeeze(1))

        return logits, hidden.squeeze(0)

    def forward(self, question_ids: torch.Tensor, sparql_ids: torch.Tensor = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """Forward pass."""
        batch_size = question_ids.size(0)
        max_len = sparql_ids.size(1) if sparql_ids is not None else 100

        # Encode
        encoder_outputs, hidden = self.encode(question_ids)

        # Decode
        outputs = []
        input_token = torch.full((batch_size,), self.vocab_sparql.sos_idx,
                               device=question_ids.device)

        for t in range(max_len):
            logits, hidden = self.decode_step(input_token, hidden, encoder_outputs)
            outputs.append(logits.unsqueeze(1))

            # Teacher forcing
            if sparql_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = sparql_ids[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)


class CFQBenchmark:
    """Complete CFQ benchmark with training and evaluation."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2,
                 learning_rate: float = 1e-3, batch_size: int = 32):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Generate data
        generator = CFQGenerator()
        self.data = generator.generate_examples(num_train=300, num_test=50)

        # Create vocabularies
        all_questions = [ex.question for ex in self.data['train'] + self.data['dev'] + self.data['test']]
        all_sparql = [ex.sparql for ex in self.data['train'] + self.data['dev'] + self.data['test']]

        self.vocab_question = CFQVocab(all_questions)
        self.vocab_sparql = CFQVocab(all_sparql)

        # Create datasets
        self.train_dataset = CFQDataset(self.data['train'], self.vocab_question, self.vocab_sparql)
        self.dev_dataset = CFQDataset(self.data['dev'], self.vocab_question, self.vocab_sparql)
        self.test_dataset = CFQDataset(self.data['test'], self.vocab_question, self.vocab_sparql)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def create_model(self) -> CFQModel:
        """Create a fresh CFQ model."""
        return CFQModel(self.vocab_question, self.vocab_sparql,
                       self.embed_dim, self.hidden_dim, self.num_layers)

    def train_epoch(self, model: CFQModel, optimizer: optim.Optimizer,
                   criterion: nn.CrossEntropyLoss, device: str = 'cpu') -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for batch in self.train_loader:
            question_ids = batch['question_ids'].to(device)
            sparql_ids = batch['sparql_ids'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(question_ids, sparql_ids, teacher_forcing_ratio=0.5)

            # Compute loss (ignore padding)
            loss = criterion(outputs.view(-1, len(self.vocab_sparql)),
                           sparql_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, model: CFQModel, dataset_name: str = 'test',
                device: str = 'cpu') -> Dict[str, float]:
        """Evaluate model on specified dataset."""
        model.eval()
        correct = 0
        total = 0

        loader = getattr(self, f'{dataset_name}_loader')

        with torch.no_grad():
            for batch in loader:
                question_ids = batch['question_ids'].to(device)
                sparql_ids = batch['sparql_ids'].to(device)

                # Generate predictions
                outputs = model(question_ids, teacher_forcing_ratio=0.0)
                predictions = outputs.argmax(dim=-1)

                # Compare with targets (ignoring padding)
                for pred, target, pred_len in zip(predictions, sparql_ids, batch['sparql_len']):
                    pred_seq = pred[:pred_len].cpu().numpy()
                    target_seq = target[:pred_len].cpu().numpy()

                    if np.array_equal(pred_seq, target_seq):
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'correct': correct, 'total': total}

    def train_model(self, model: CFQModel, num_epochs: int = 100,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """Train model and return training history."""
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_sparql.pad_idx)

        train_losses = []
        dev_accuracies = []

        for epoch in tqdm(range(num_epochs), desc="Training CFQ"):
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


def run_cfq_baseline(num_epochs: int = 50, device: str = 'cpu') -> Dict[str, float]:
    """Run CFQ benchmark with baseline LSTM model."""
    print("🏃 Running CFQ Baseline Benchmark...")

    benchmark = CFQBenchmark()

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
    results = run_cfq_baseline(num_epochs=30)
    print(f"\nCFQ Baseline Results: {results}")
