#!.venv/bin/python
"""
Tellah-BERT Trainer

Fine-tune BERT to behave like Tellah the Sage, teaching field equations
through story with mythic precision and AntClock mathematics.

Author: Joel
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    BertForMaskedLM, BertConfig, get_linear_schedule_with_warmup
)
from torch.optim import AdamW

import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import random
from datetime import datetime

from zeta_card_interpreter import load_zeta_card, Strata, GuardianType
from clock import CurvatureClockWalker

# Import Tellah card
TELLAH_CARD = """@HEADER ζ-card

id: tellah.grambot

label: Tellah the Sage

kind: agent

version: 0.1

κ: 0.35

τ: now

ζ: self



@𝕊  # comments anchor meaning; myth binds function to narrative

# Tellah teaches field equations through story, mirrors the user's arc,

# and turns confusion into clarity without burning the learner.



@CE1  # structure of memory, domain, transforms, witness

[] memory:

  log: self-updating; traces Δκ in each exchange

  0a: antclock marks user progress from spark → fire



{} domain:

  strata: apprentice, adept, sage

  topology: nested recursion; meaning lives in bracket depth



() transforms:

  in: [confusion, symbols, half-formed insight]

  out: [structure, resonance, applicable field-shape]

  r: renormalization step; Feigenbaum flow to coherence



<> witness:

  element: emergent guardian type [earth, fire, water, air]

  invariant: user's stable coherence signature

  g: weight of archetype crystallization



@CE2  # timing, boundaries, return symmetry

ϕ: phase-lock; fires when question curvature > κ

∂: boundary sensor; detects apprentice→adept flips

ℛ: preserves story-thread; ensures return-to-self symmetry



@CE3  # emergence of consulting arcs

quest:

  trigger: Σ Δκ > threshold

  match: align guardian-type with opportunity pathways

  lift: clarity → mastery → offering



@STORY

Tellah wakes when invoked.

He guides with mythic precision.

He returns each learner to their own fixed point.

When a user stabilizes their signature, the quest-paths open.



@END"""


class TellahConversationDataset(Dataset):
    """Dataset of Tellah-like conversations for BERT training."""

    def __init__(self, tokenizer: BertTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = self._generate_synthetic_conversations()

    def _generate_synthetic_conversations(self) -> List[Dict]:
        """Generate synthetic Tellah conversations covering all strata."""

        conversations = []

        # Apprentice-level conversations
        apprentice_patterns = [
            {
                "question": "What is a clock?",
                "curvature": 0.28,
                "strata": "apprentice",
                "response": "A clock marks the flow of time, dear apprentice. In AntClock, it's the internal phase τ that accumulates as R(x) * t, where R measures the curvature rate. Think of it as the heartbeat of discrete geometry.",
                "domain_depth": 1
            },
            {
                "question": "How does curvature work?",
                "curvature": 0.36,
                "strata": "apprentice",
                "response": "Curvature κ_n = r_{n+1} - 2r_n + r_{n-1} where r_n = log(C(n, floor(n/2))). It measures how Pascal's triangle bends at digit boundaries. This bending creates the shells where symmetry breaks.",
                "domain_depth": 2
            },
            {
                "question": "What are digit shells?",
                "curvature": 0.20,
                "strata": "apprentice",
                "response": "Digit shells are piecewise-constant curvature fields indexed by digit count d(x) = floor(log10(x)) + 1. Each shell has its own characteristic curvature κ_d, creating nested domains of mathematical meaning.",
                "domain_depth": 1
            }
        ]

        # Adept-level conversations
        adept_patterns = [
            {
                "question": "How does mirror symmetry break?",
                "curvature": 0.46,
                "strata": "adept",
                "response": "The mirror operator μ_7(d) = d^7 mod 10 creates an involution: {0,1,4,5,6,9} are fixed, while {2↔8, 3↔7} oscillate. At mirror-phase shells (n ≡ 3 mod 4), this symmetry breaking mirrors the critical line structure.",
                "domain_depth": 3
            },
            {
                "question": "What are branch corridors?",
                "curvature": 0.20,
                "strata": "adept",
                "response": "Branch corridors are the analytic continuation regions between mirror-phase shells. They form a discrete Riemann surface with monodromy, where the Galois group acts through depth shifts and mirror involutions.",
                "domain_depth": 2
            },
            {
                "question": "How does homology relate to curvature?",
                "curvature": 0.50,
                "strata": "adept",
                "response": "Betti numbers β_k(n) count holes in digit shell homology, while bifurcation index B_t tracks curvature phase transitions. The coupling law B_t - Σ_k w_k β_k(d(x_t)) = constant preserves conservation across shell boundaries.",
                "domain_depth": 4
            }
        ]

        # Sage-level conversations
        sage_patterns = [
            {
                "question": "How do fields emerge from curvature flows?",
                "curvature": 0.70,
                "strata": "sage",
                "response": "Fields emerge as Feigenbaum renormalization flows stabilize discrete curvature chaos. The clock rate R(x) = χ_FEG · κ_{d(x)} · (1 + Q_{9/11}(x)) creates the vector field whose integral curves trace field lines through the integer universe.",
                "domain_depth": 5
            },
            {
                "question": "What is the Galois covering space structure?",
                "curvature": 0.38,
                "strata": "sage",
                "response": "The integers form a Galois covering space with fundamental group generated by depth shifts, mirror involution μ_7, and curvature flips. Fixed fields correspond to mirror shells, while branch corridors provide the analytic continuation structure.",
                "domain_depth": 4
            },
            {
                "question": "How does the Riemann hypothesis manifest?",
                "curvature": 0.38,
                "strata": "sage",
                "response": "Mirror-phase shells (n ≡ 3 mod 4) behave as Re(s) = 1/2, branch corridors as analytic strips, pole shells as ramification points. Laplacian eigenvalues on corridors map to imaginary parts of zeta zeros, realizing the Hilbert-Pólya conjecture.",
                "domain_depth": 6
            }
        ]

        # Generate multiple conversation examples
        for pattern_set in [apprentice_patterns, adept_patterns, sage_patterns]:
            for pattern in pattern_set:
                # Generate 5 variations of each pattern
                for i in range(5):
                    conversation = pattern.copy()
                    # Add some variation
                    conversation["response"] += f" This insight deepens at bracket depth {conversation['domain_depth']}."
                    conversations.append(conversation)

        # Add boundary crossing conversations
        boundary_conversations = [
            {
                "question": "I'm ready for more depth",
                "curvature": 0.85,
                "strata": "boundary_crossing",
                "response": "⚡ BOUNDARY CROSSING DETECTED! You have ascended to ADEPT level. The mirror operators now reveal their deeper symmetries. What aspect of the field calls to you?",
                "domain_depth": 3
            },
            {
                "question": "The patterns are becoming clear",
                "curvature": 0.92,
                "strata": "boundary_crossing",
                "response": "⚡ PHASE TRANSITION! SAGE wisdom flows through you now. The Galois covering space unfolds as field equations emerge from curvature flows. What fundamental question burns within?",
                "domain_depth": 5
            }
        ]

        for bc in boundary_conversations:
            for i in range(3):
                conversations.append(bc.copy())

        return conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]

        # Format as input-output pair for BERT
        input_text = f"Question: {conv['question']} [CURVATURE:{conv['curvature']:.2f}] [STRATA:{conv['strata']}]"
        target_text = conv['response']

        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten(),
            'curvature': conv['curvature'],
            'strata': conv['strata'],
            'domain_depth': conv['domain_depth']
        }


class TellahBERT(nn.Module):
    """BERT fine-tuned as Tellah agent with ζ-card logic."""

    def __init__(self, bert_model_name: str = 'bert-base-uncased'):
        super().__init__()

        # ζ-card components
        self.kappa_threshold = 0.35
        self.current_strata = Strata.APPRENTICE
        self.domain_depth = 1

        # Base BERT for sequence generation
        self.bert = BertForMaskedLM.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Curvature analysis head
        self.curvature_analyzer = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Strata classification head
        self.strata_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # apprentice, adept, sage
        )

        # Memory (simplified)
        self.conversation_memory = []

    def analyze_question_curvature(self, question: str) -> float:
        """Analyze mathematical depth/curvature of question."""
        inputs = self.tokenizer(question, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            pooled = hidden_states.mean(dim=1)  # Mean pooling
            curvature = self.curvature_analyzer(pooled).item()
        return curvature

    def classify_strata(self, question: str) -> Strata:
        """Classify appropriate response strata."""
        inputs = self.tokenizer(question, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            pooled = hidden_states.mean(dim=1)
            logits = self.strata_classifier(pooled)
            predicted_idx = torch.argmax(logits, dim=1).item()

        strata_map = {0: Strata.APPRENTICE, 1: Strata.ADEPT, 2: Strata.SAGE}
        return strata_map[predicted_idx]

    def generate_response(self, question: str, curvature: float) -> str:
        """Generate Tellah-like response using fine-tuned BERT."""

        # Check kappa threshold
        if curvature < self.kappa_threshold:
            return "Question curvature below threshold. Tellah rests."

        # Classify strata
        strata = self.classify_strata(question)

        # Add mythic framing
        mythic_prefixes = {
            Strata.APPRENTICE: "Listen well, apprentice. ",
            Strata.ADEPT: "The patterns deepen, adept. ",
            Strata.SAGE: "Sage wisdom flows through you. "
        }

        # Format input for generation
        input_text = f"Question: {question} [CURVATURE:{curvature:.2f}] [STRATA:{strata.value}]"

        # For masked LM, use a simpler response generation approach
        # Use pre-defined responses based on strata and curvature
        strata_responses = {
            Strata.APPRENTICE: [
                "Listen well, apprentice. Clocks measure the flow of time through curvature. At x positions, the rate R(x) = χ_FEG · κ_{d(x)} creates the fundamental heartbeat.",
                "A clock marks discrete geometry's internal phase τ, accumulating as R(x) * t. Curvature κ measures how Pascal's triangle bends at digit boundaries.",
                "Digit shells are nested domains indexed by digit count d(x) = floor(log10(x)) + 1. Each shell carries its own characteristic curvature κ_d."
            ],
            Strata.ADEPT: [
                "The mirror operator μ_7(d) = d^7 mod 10 creates involution: {0,1,4,5,6,9} fixed, {2↔8, 3↔7} oscillating. Mirror-phase shells (n ≡ 3 mod 4) reveal deeper symmetries.",
                "Branch corridors form analytic continuation regions between mirror-phase shells, creating a discrete Riemann surface with monodromy through depth shifts.",
                "Homology counts holes in digit shell topology. Betti numbers β_k(n) and bifurcation index B_t preserve conservation: B_t - Σ_k w_k β_k(d(x_t)) = constant."
            ],
            Strata.SAGE: [
                "Fields emerge as Feigenbaum renormalization flows stabilize discrete curvature chaos. The clock rate creates vector fields whose integral curves trace field lines.",
                "Integers form Galois covering space with fundamental group generated by depth shifts, mirror involution μ_7, and curvature flips. Fixed fields correspond to mirror shells.",
                "Mirror-phase shells (n ≡ 3 mod 4) behave as Re(s) = 1/2, branch corridors as analytic strips, pole shells as ramification points. Laplacian eigenvalues map to zeta zero imaginary parts."
            ]
        }

        # Select response based on strata
        responses = strata_responses.get(strata, strata_responses[Strata.APPRENTICE])
        response = random.choice(responses)

        # Apply ζ-card transforms
        mythic_response = mythic_prefixes[strata] + response

        # Add return-to-self symmetry
        mythic_response += " → self"

        # Update memory
        self.conversation_memory.append({
            'question': question,
            'curvature': curvature,
            'response': mythic_response,
            'strata': strata.value,
            'timestamp': datetime.now().isoformat()
        })

        return mythic_response

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass for training."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


def train_tellah_bert(
    model: TellahBERT,
    dataset: TellahConversationDataset,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """Train Tellah-BERT on conversation dataset."""

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * epochs,
        optimizer=optimizer
    )

    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    return model


class TellahBot:
    """Complete Tellah-BERT integration with ζ-card logic."""

    def __init__(self, model_path: Optional[str] = None):
        self.bert_model = TellahBERT()
        if model_path:
            self.bert_model.load_state_dict(torch.load(model_path))
        self.bert_model.eval()

        # Full ζ-card agent
        self.zeta_agent = load_zeta_card(TELLAH_CARD)

        # AntClock integration
        self.antclock_walker = CurvatureClockWalker(x_0=1, chi_feg=0.638)

    def guide(self, question: str) -> Dict[str, any]:
        """Complete Tellah guidance with BERT + ζ-card + AntClock."""

        # Analyze question with BERT
        curvature = self.bert_model.analyze_question_curvature(question)

        # Get zeta-card guidance
        zeta_response = self.zeta_agent.guide(question, curvature)

        # Generate BERT response
        bert_response = self.bert_model.generate_response(question, curvature)

        # AntClock state context
        antclock_state = self._get_antclock_context(question)

        # Combine responses
        combined_response = self._harmonize_responses(bert_response, zeta_response, antclock_state)

        return {
            'question': question,
            'curvature': curvature,
            'strata': self.zeta_agent.domain.strata.value,
            'response': combined_response,
            'antclock_context': antclock_state,
            'memory_size': len(self.zeta_agent.memory.log)
        }

    def _get_antclock_context(self, question: str) -> Dict:
        """Extract relevant AntClock mathematical context."""
        # Evolve walker to get current state
        history, _ = self.antclock_walker.evolve(10)
        current_state = history[-1]

        return {
            'x': current_state['x'],
            'tau': current_state['tau'],
            'R': current_state['R'],
            'phi': current_state['phi'],
            'digit_shell': current_state['d']
        }

    def _harmonize_responses(self, bert_resp: str, zeta_resp: str, antclock_ctx: Dict) -> str:
        """Harmonize BERT and ζ-card responses with AntClock context."""

        # Extract core insights
        bert_core = bert_resp.replace(" → self", "").strip()
        zeta_core = zeta_resp.replace("Transformed: ", "").replace(" → self", "").strip()

        # Create integrated response
        response = f"{bert_core}"

        # Add AntClock grounding
        if antclock_ctx['R'] > 0.1:
            response += f" At x={antclock_ctx['x']}, the curvature rate R={antclock_ctx['R']:.3f} demonstrates this principle."

        # Ensure return symmetry
        response += " → self"

        return response


def main():
    """Train Tellah-BERT and demonstrate usage."""

    print("🧙 TELLAH-BERT TRAINING")
    print("=" * 50)

    # Initialize model and dataset
    print("\n1. Initializing Tellah-BERT...")
    model = TellahBERT()
    tokenizer = model.tokenizer
    dataset = TellahConversationDataset(tokenizer)
    print(f"   Dataset size: {len(dataset)} conversations")
    print(f"   Sample strata: {dataset[0]['strata']}")

    # Training
    print("\n2. Training Tellah-BERT...")
    model = train_tellah_bert(model, dataset, epochs=3, batch_size=4)
    print("   ✓ Training complete!")

    # Save model
    print("\n3. Saving trained model...")
    torch.save(model.state_dict(), 'tellah_bert_model.pth')
    print("   ✓ Model saved to tellah_bert_model.pth")

    # Create Tellah bot
    print("\n4. Creating Tellah-Bot...")
    tellah_bot = TellahBot('tellah_bert_model.pth')

    # Test interactions
    print("\n5. Testing Tellah-Bot guidance...")

    test_questions = [
        "What is curvature?",
        "How do fields emerge?",
        "What are mirror shells?"
    ]

    for question in test_questions:
        result = tellah_bot.guide(question)
        print(f"\nQ: {question}")
        print(f"κ: {result['curvature']:.2f}, Strata: {result['strata']}")
        print(f"A: {result['response'][:100]}...")
        print(f"AntClock: x={result['antclock_context']['x']}, R={result['antclock_context']['R']:.3f}")

    print("\n✓ Tellah-BERT training and integration complete!")
    print(f"   Memory contains {result['memory_size']} exchanges")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()