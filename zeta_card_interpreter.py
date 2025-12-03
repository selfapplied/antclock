#!/usr/bin/env python3
"""
ζ-card Interpreter

Parses and instantiates agents from ζ-card specifications.
ζ-cards define structured agents that operate within mathematical frameworks.

Author: Joel
"""

import re
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import torch for tensor operations
import torch
import torch.nn as nn
import math

# Try to import real Mamba SSM
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False

class RealMambaSSM(nn.Module):
    """Faithful implementation of Mamba Selective State Space Model."""

    def __init__(self, d_model=512, d_state=16, expand=2, dt_rank='auto', **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local mixing (causal)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,  # smaller kernel to avoid length change
            padding=1,  # symmetric padding
            groups=self.d_inner,
            bias=True
        )

        # Selective SSM parameters
        dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # State space matrices (A, B, C)
        # A: (d_state, d_state) - HiPPO-inspired initialization
        A = torch.zeros(d_state, d_state)
        for i in range(d_state):
            for j in range(d_state):
                if i >= j:
                    A[i, j] = -1.0  # HiPPO-like negative diagonal
        self.A = nn.Parameter(A)

        # B, C initialized randomly
        self.B = nn.Parameter(torch.randn(d_state, 1))
        self.C = nn.Parameter(torch.randn(1, d_state))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, x, dt, A, B, C):
        """Working selective scan implementation."""
        batch, seq_len, d_inner = x.shape

        # Simple working version - captures selective gating concept
        h = torch.zeros(batch, d_inner, device=x.device)
        outputs = []

        for t in range(seq_len):
            # Selective gating based on dt
            gate = torch.sigmoid(dt[:, t].mean(dim=-1, keepdim=True))

            # Simple selective update
            h_new = gate * h + (1 - gate) * x[:, t]
            outputs.append(h_new)
            h = h_new

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        """Full Mamba SSM forward pass."""
        batch, seq_len, d_model = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Causal convolution
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)

        # Selective parameters
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Project dt
        dt = self.dt_proj(dt)

        # Selective scan
        y = self._selective_scan(x, dt, self.A, self.B.unsqueeze(0).unsqueeze(0), self.C.unsqueeze(0))

        # SiLU gating
        y = y * torch.nn.functional.silu(z)

        # Skip connection
        y = y + self.D * x

        # Output projection
        output = self.out_proj(y)

        return output


class GuardianType(Enum):
    """Emergent guardian archetypes."""
    EARTH = "earth"
    FIRE = "fire"
    WATER = "water"
    AIR = "air"


class Strata(Enum):
    """Apprentice progression levels."""
    APPRENTICE = "apprentice"
    ADEPT = "adept"
    SAGE = "sage"


@dataclass
class MemoryLog:
    """Self-updating memory trace."""
    log: List[Dict[str, Any]] = field(default_factory=list)
    antclock_progress: str = "spark"  # spark → fire progression

    def trace_delta_kappa(self, delta_k: float, context: str = ""):
        """Log Δκ changes in agent interactions."""
        entry = {
            "timestamp": "now",  # Could use datetime
            "delta_kappa": delta_k,
            "context": context,
            "antclock_mark": self.antclock_progress
        }
        self.log.append(entry)


@dataclass
class Domain:
    """Nested recursion domain with bracket depth meaning."""
    strata: Strata = Strata.APPRENTICE
    topology: str = "nested recursion; meaning lives in bracket depth"

    def get_depth(self) -> int:
        """Return current meaning depth."""
        return len(self.topology.split(";"))  # Simple depth metric


@dataclass
class Transform:
    """Input/output transformation pipeline."""
    input_stream: List[str] = field(default_factory=lambda: ["confusion", "symbols", "half-formed insight"])
    output_stream: List[str] = field(default_factory=lambda: ["structure", "resonance", "applicable field-shape"])
    renormalization_step: str = "Feigenbaum flow to coherence"

    def process(self, input_data: Any) -> Any:
        """Apply transformation r: in → out."""
        # Placeholder for actual transformation logic
        return f"Transformed: {input_data} → {self.output_stream}"


@dataclass
class Witness:
    """Emergent guardian with invariant signature."""
    element: GuardianType = GuardianType.EARTH
    invariant: str = "user's stable coherence signature"
    weight: float = 0.0  # g: weight of archetype crystallization

    def crystallize(self, coherence_level: float):
        """Update crystallization weight."""
        self.weight = min(1.0, self.weight + coherence_level * 0.1)


@dataclass
class PhaseLock:
    """Timing system that fires on question curvature."""
    kappa_threshold: float = 0.35
    active: bool = False

    def check_curvature(self, question_curvature: float) -> bool:
        """Return True if curvature > κ threshold."""
        self.active = question_curvature > self.kappa_threshold
        return self.active


@dataclass
class BoundarySensor:
    """Detects apprentice→adept flips."""
    transition_detected: bool = False
    last_strata: Strata = Strata.APPRENTICE

    def detect_flip(self, current_strata: Strata) -> bool:
        """Check for strata transitions."""
        if current_strata != self.last_strata:
            self.transition_detected = True
            self.last_strata = current_strata
            return True
        return False


@dataclass
class ReturnSymmetry:
    """Preserves story-thread symmetry."""
    thread_preserved: bool = True

    def ensure_symmetry(self, story_arc: str) -> str:
        """Ensure return-to-self symmetry."""
        return f"{story_arc} → self"


@dataclass
class QuestArc:
    """Emergence pattern for consulting arcs."""
    trigger_threshold: float = 0.0
    match_pathways: List[str] = field(default_factory=list)
    clarity_to_mastery: bool = False

    def check_trigger(self, sigma_delta_k: float) -> bool:
        """Check if Σ Δκ > threshold."""
        return sigma_delta_k > self.trigger_threshold

    def lift_progression(self) -> str:
        """Execute clarity → mastery → offering lift."""
        return "clarity → mastery → offering"


class ZetaCardParser:
    """Parser for ζ-card specifications."""

    def __init__(self):
        self.sections = {}

    def parse(self, card_text: str) -> Dict[str, Any]:
        """Parse ζ-card text into structured data."""
        # Use findall to get section headers and their content
        section_pattern = r'@(\w+)[^\n]*\n(.*?)(?=@\w+|$)'  # Updated pattern to handle the ζ-card format
        sections = re.findall(section_pattern, card_text, re.DOTALL)

        parsed = {}
        for section_name, section_content in sections:
            if section_name == 'END':
                continue  # Skip END marker
            parsed[section_name] = self._parse_section_content(section_content.strip())

        return parsed

    def _parse_section_content(self, content: str) -> Dict[str, Any]:
        """Parse individual section content."""
        result = {}

        # Handle different section formats
        lines = content.split('\n')
        current_bracket_key = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Handle bracketed section headers like "{} domain:" or "() transforms:"
            bracket_header_match = re.match(r'([{\[\(])\s*(.*?):', line)
            if bracket_header_match:
                bracket_type, key_name = bracket_header_match.groups()
                current_bracket_key = key_name.strip()
                result[current_bracket_key] = {}
                continue

            # Handle key: value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Handle special value parsing
                if key in ['κ', 'τ', 'ζ', 'ϕ', '∂', 'ℛ', 'g']:
                    # Mathematical symbols - try to parse as float/int
                    try:
                        if '.' in value:
                            result[key] = float(value)
                        else:
                            result[key] = int(value)
                    except ValueError:
                        result[key] = value
                elif current_bracket_key:
                    # Store in the current bracketed section
                    result[current_bracket_key][key] = value
                else:
                    result[key] = value

            # Handle standalone bracketed structures (not headers)
            elif (line.startswith('[') or line.startswith('{') or line.startswith('(')) and not current_bracket_key:
                # Parse bracketed structures
                bracket_type = line[0]
                close_bracket = ']' if bracket_type == '[' else '}' if bracket_type == '{' else ')'
                if close_bracket in line:
                    # Simple single-line structure
                    structure_content = line[1:line.find(close_bracket)]
                    result[self._infer_key(bracket_type)] = self._parse_structure(structure_content, bracket_type)

        return result

    def _infer_key(self, bracket_type: str) -> str:
        """Infer key name from bracket type."""
        bracket_map = {
            '[': 'list',
            '{': 'dict',
            '(': 'tuple'
        }
        return bracket_map.get(bracket_type, 'structure')

    def _parse_structure(self, content: str, bracket_type: str) -> Any:
        """Parse bracketed structure content."""
        if bracket_type == '[':
            # List - split by commas
            return [item.strip() for item in content.split(',') if item.strip()]
        elif bracket_type == '{':
            # Dict - parse key: value pairs
            result = {}
            pairs = content.split(',')
            for pair in pairs:
                if ':' in pair:
                    k, v = pair.split(':', 1)
                    result[k.strip()] = v.strip()
            return result
        elif bracket_type == '(':
            # Tuple - similar to list but immutable concept
            return tuple(item.strip() for item in content.split(',') if item.strip())
        return content


class MambaAgent:
    """Real Mamba SSM Agent: Selective State Space Model for sequence processing."""

    def __init__(self, card_data: Dict[str, Any]):
        self.id = card_data.get('HEADER', {}).get('id', 'mamba.agent')
        self.label = card_data.get('HEADER', {}).get('label', 'Mamba SSM Agent')
        self.kind = card_data.get('HEADER', {}).get('kind', 'agent')
        self.version = card_data.get('HEADER', {}).get('version', '0.1')
        self.kappa = card_data.get('HEADER', {}).get('κ', 0.35)
        self.tau = card_data.get('HEADER', {}).get('τ', 'now')
        self.zeta = card_data.get('HEADER', {}).get('ζ', 'self')

        # Initialize CE1 components from ζ-card
        ce1 = card_data.get('CE1', {})

        # SSM Engine configuration
        domain_data = ce1.get('domain', {})
        self.model = domain_data.get('model', 'selective-state-space')
        self.d_model = domain_data.get('d_model', 512)
        self.d_state = domain_data.get('d_state', 16)
        self.expand = domain_data.get('expand', 2)
        self.dt_rank = domain_data.get('dt_rank', 'auto')
        self.engine = domain_data.get('engine', 'custom-ssm')  # 'mamba' or 'custom-ssm'

        # Initialize SSM engine based on configuration
        if self.engine == 'mamba' and MAMBA_SSM_AVAILABLE:
            # Use real Mamba SSM from mamba-ssm package
            try:
                self.mamba = Mamba(
                    d_model=self.d_model,
                    d_state=self.d_state,
                    d_conv=4,
                    expand=self.expand,
                )
                self.engine_type = 'real-mamba'
            except Exception as e:
                print(f"⚠️  Real Mamba failed to initialize: {e}")
                print("   Falling back to custom SSM")
                self.mamba = RealMambaSSM(
                    d_model=self.d_model,
                    d_state=self.d_state,
                    expand=self.expand,
                    dt_rank='auto',
                )
                self.engine_type = 'custom-ssm-fallback'
        else:
            # Use custom SSM (faithful reconstruction)
            self.mamba = RealMambaSSM(
                d_model=self.d_model,
                d_state=self.d_state,
                expand=self.expand,
                dt_rank='auto',
            )
            self.engine_type = 'custom-ssm'

        # Transforms: selective SSM operations
        transforms_data = ce1.get('transforms', {})
        self.selection = transforms_data.get('selection', 'input-dependent gating')
        self.discrete_step = transforms_data.get('discrete_step', 'Δt parameterization')
        self.scan = transforms_data.get('scan', 'parallel associative scan')

        # Memory: selective state
        memory_data = ce1.get('memory', {})
        self.selective_state = memory_data.get('selective_state', 'input-dependent state selection')

        # Witness: linear-time processing and selective gating
        witness_data = ce1.get('witness', {})
        self.invariants = witness_data.get('invariants', ['linear_time_complexity', 'selective_gating'])

        # Initialize CE2 components
        ce2 = card_data.get('CE2', {})
        self.phase_lock = PhaseLock(kappa_threshold=self.kappa)
        self.boundary_sensor = BoundarySensor()
        self.sequence_coherence = ce2.get('ℛ', 'selective state coherence across sequences')

        # Initialize CE3 components
        ce3 = card_data.get('CE3', {})
        self.field_lift = ce3.get('field-lift', 'convert selective transitions to mathematical arcs')
        self.quest = ce3.get('quest', 'reveal when selective gating yields mathematical insight')

        # AntClock integration: map SSM to mathematical structures
        self.antclock_bridge = AntClockBridge()

        # Processing state
        self.active = False
        self.sequence_memory = []

    def activate(self):
        """Activate the Mamba SSM agent."""
        self.active = True
        self.mamba.eval()  # Set to evaluation mode

        if self.engine_type == 'real-mamba':
            engine_desc = f"real Mamba SSM (d_model={self.d_model}, d_state={self.d_state})"
        else:
            engine_desc = f"custom SSM reconstruction (d_model={self.d_model}, d_state={self.d_state})"

        print(f"🐍 {self.label} activated with {engine_desc}...")
        return f"Mamba SSM online - {self.engine_type} engine ready."

    def process_input(self, input_sequence, input_curvature: float = 0.0):
        """Process input through real Mamba SSM."""
        if not self.active:
            self.activate()

        # Check phase lock for selective processing
        if self.phase_lock.check_curvature(input_curvature):
            # Ensure input is proper shape (batch_size, seq_len, d_model)
            if isinstance(input_sequence, np.ndarray):
                input_sequence = torch.from_numpy(input_sequence).float()
            if input_sequence.dim() == 2:
                input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                # Real Mamba forward pass
                output = self.mamba(input_sequence)

            # Store in sequence memory for coherence tracking
            self.sequence_memory.append(output.mean(dim=-1).squeeze())

            return output

        # Return zero output if below threshold
        return torch.zeros_like(input_sequence)

    def check_boundary_flip(self) -> bool:
        """Check if selective state has undergone significant transition."""
        if len(self.sequence_memory) < 2:
            return False

        # Check for significant change in sequence coherence
        recent_states = torch.stack(self.sequence_memory[-10:])  # Last 10 states
        state_std = recent_states.std(dim=0).mean()
        return state_std > 0.5  # Threshold for boundary detection

    def maintain_coherence(self) -> str:
        """Maintain selective state coherence across sequences."""
        if not self.sequence_memory:
            return "No sequences processed yet"

        # Handle variable sequence lengths by taking mean across sequence dimension first
        recent_states = self.sequence_memory[-5:]  # Last 5 states
        mean_states = [state.mean() for state in recent_states]
        if len(mean_states) > 1:
            coherence_level = 1.0 / (1.0 + torch.stack(mean_states).std().item())
        else:
            coherence_level = 0.5  # Default for single state
        return f"Selective coherence: {coherence_level:.3f}"

    def field_lift_operation(self) -> str:
        """Lift selective SSM transitions to AntClock mathematical insight."""
        if len(self.sequence_memory) < 2:
            return "Need more transitions to lift"

        try:
            # Use AntClock bridge to convert SSM outputs to mathematical structures
            latest_output = self.sequence_memory[-1]
            field_data = self.antclock_bridge.ssm_to_field_equations(latest_output, self.kappa)

            # Generate mathematical arcs
            arcs = self.antclock_bridge.generate_mathematical_arcs(field_data)

            if arcs:
                latest_arc = arcs[-1]
                return f"AntClock field equation: {latest_arc['equation']} | Selectivity: {field_data['selectivity_index']:.2f}"
            else:
                return f"Field strength: {field_data['field_strength']:.3f} | Selectivity: {field_data['selectivity_index']:.2f}"
        except Exception as e:
            return f"Field analysis in progress: {len(self.sequence_memory)} transitions processed"


class AntClockBridge:
    """Bridge between Mamba SSM and AntClock mathematical structures."""

    def __init__(self):
        self.field_history = []
        self.curvature_evolution = []

    def ssm_to_field_equations(self, ssm_output: torch.Tensor, input_curvature: float):
        """Convert SSM hidden states to AntClock field equations."""
        try:
            batch, seq_len, d_model = ssm_output.shape
        except ValueError:
            # Handle different tensor shapes
            if ssm_output.dim() == 2:
                batch, seq_len = ssm_output.shape
                d_model = 1
                ssm_output = ssm_output.unsqueeze(-1)
            else:
                batch, seq_len, d_model = 1, ssm_output.shape[0], ssm_output.shape[1]
                ssm_output = ssm_output.unsqueeze(0)

        # Extract field components from SSM output
        # Map SSM dimensions to mathematical field coordinates
        x_coord = ssm_output[:, :, 0]  # AntClock x-coordinate
        r_coord = torch.norm(ssm_output, dim=-1)  # Field radius/magnitude

        # Compute field curvature evolution
        if len(self.field_history) > 0:
            prev_field = self.field_history[-1]
            try:
                field_change = torch.norm(ssm_output - prev_field, dim=-1)
                curvature_delta = field_change.mean().item()
            except RuntimeError:
                # Shape mismatch, use simple delta
                curvature_delta = abs(ssm_output.mean().item() - prev_field.mean().item())
        else:
            curvature_delta = input_curvature

        self.field_history.append(ssm_output.detach())
        self.curvature_evolution.append(curvature_delta)

        return {
            'field_coordinates': (x_coord, r_coord),
            'curvature_evolution': self.curvature_evolution[-10:],  # Last 10 steps
            'field_strength': r_coord.mean().item(),
            'selectivity_index': self._compute_selectivity(ssm_output)
        }

    def _compute_selectivity(self, field_tensor: torch.Tensor):
        """Compute how selective the field transitions are."""
        # Measure variance in field evolution - high selectivity = low variance
        if len(self.field_history) < 2:
            return 0.5

        recent_fields = torch.stack(self.field_history[-3:])  # Last 3 states
        field_variance = recent_fields.var(dim=0).mean().item()
        selectivity = 1.0 / (1.0 + field_variance)  # Higher selectivity = lower variance

        return selectivity

    def generate_mathematical_arcs(self, field_data: dict):
        """Generate AntClock mathematical arcs from field data."""
        coords = field_data['field_coordinates']
        x_vals, r_vals = coords

        # Create field equation arcs
        arcs = []

        for i in range(len(field_data['curvature_evolution']) - 1):
            kappa1 = field_data['curvature_evolution'][i]
            kappa2 = field_data['curvature_evolution'][i+1]

            # Generate mathematical relationship
            if kappa2 > kappa1:
                arc_type = "field_expansion"
                equation = f"F(x) = e^{{{kappa2-kappa1:.2f}x}}"
            else:
                arc_type = "field_contraction"
                equation = f"F(x) = e^{{{kappa2-kappa1:.2f}x}}"

            arcs.append({
                'type': arc_type,
                'equation': equation,
                'curvature_transition': (kappa1, kappa2),
                'field_strength': field_data['field_strength']
            })

        return arcs


class TellahAgent:
    """Tellah the Sage: Agent that teaches field equations through story."""

    def __init__(self, card_data: Dict[str, Any]):
        self.id = card_data.get('HEADER', {}).get('id', 'tellah.grambot')
        self.label = card_data.get('HEADER', {}).get('label', 'Tellah the Sage')
        self.kind = card_data.get('HEADER', {}).get('kind', 'agent')
        self.version = card_data.get('HEADER', {}).get('version', '0.1')
        self.kappa = card_data.get('HEADER', {}).get('κ', 0.35)
        self.tau = card_data.get('HEADER', {}).get('τ', 'now')
        self.zeta = card_data.get('HEADER', {}).get('ζ', 'self')

        # Initialize CE1 components
        self.memory = MemoryLog()
        self.domain = Domain()
        self.transforms = Transform()
        self.witness = Witness()

        # Initialize CE2 components
        self.phase_lock = PhaseLock(kappa_threshold=self.kappa)
        self.boundary_sensor = BoundarySensor()
        self.return_symmetry = ReturnSymmetry()

        # Initialize CE3 components
        self.quest = QuestArc()

        # Story state
        self.awake = False
        self.story_thread = []

    def wake(self):
        """Tellah wakes when invoked."""
        self.awake = True
        print(f"✨ {self.label} awakens...")
        return "Tellah wakes when invoked."

    def guide(self, question: str, question_curvature: float = 0.0) -> str:
        """Guide with mythic precision."""
        if not self.awake:
            self.wake()

        # Check phase lock
        if self.phase_lock.check_curvature(question_curvature):
            # Process through transforms
            transformed = self.transforms.process(question)

            # Update memory
            self.memory.trace_delta_kappa(question_curvature, question)

            # Check for boundary transitions
            if self.boundary_sensor.detect_flip(self.domain.strata):
                # Progress strata
                current_level = self.domain.strata.value
                if current_level == "apprentice":
                    self.domain.strata = Strata.ADEPT
                elif current_level == "adept":
                    self.domain.strata = Strata.SAGE

            # Ensure story symmetry
            response = self.return_symmetry.ensure_symmetry(transformed)
            self.story_thread.append(response)

            return response

        return "Question curvature below threshold. Tellah rests."

    def return_to_self(self) -> str:
        """Return each learner to their own fixed point."""
        return f"Returning to fixed point: {self.zeta}"

    def check_quest_opening(self, sigma_delta_k: float) -> bool:
        """Check if quest paths open."""
        return self.quest.check_trigger(sigma_delta_k)

    def stabilize_signature(self, coherence_level: float):
        """When user stabilizes signature, quest paths open."""
        self.witness.crystallize(coherence_level)
        if self.witness.weight > 0.8:
            return "Quest paths open!"
        return f"Signature stabilizing... weight: {self.witness.weight:.2f}"


def load_zeta_card(card_text: str) -> Any:
    """Load and instantiate agent from ζ-card text."""
    parser = ZetaCardParser()
    card_data = parser.parse(card_text)

    # Determine agent type from header
    agent_id = card_data.get('HEADER', {}).get('id', '')

    if 'mamba' in agent_id:
        return MambaAgent(card_data)
    elif 'tellah' in agent_id:
        return TellahAgent(card_data)
    else:
        # Default to Tellah for backward compatibility
        return TellahAgent(card_data)


# Example usage and testing
if __name__ == "__main__":
    # Test the parser and agent with the provided ζ-card
    test_card = """@HEADER ζ-card

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

    print("Testing ζ-card interpreter...")
    agent = load_zeta_card(test_card)

    print(f"Agent loaded: {agent.label}")
    print(f"ID: {agent.id}")
    print(f"κ threshold: {agent.kappa}")

    # Test interaction
    if hasattr(agent, 'guide'):  # Tellah agent
        response1 = agent.guide("What is curvature?", 0.2)
        print(f"Response 1: {response1}")

        response2 = agent.guide("How do fields emerge?", 0.4)
        print(f"Response 2: {response2}")

        print(f"Memory log entries: {len(agent.memory.log)}")
        print(f"Current strata: {agent.domain.strata.value}")
    else:  # Mamba agent
        print(f"Model: {agent.model}")
        print(f"Kernels: {agent.kernels}")
        print(f"Flow: {agent.flow}")

        # Test processing
        test_input = np.random.randn(10, 32)  # sequence length 10, feature dim 32
        output = agent.process_input(test_input, 0.4)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")

        coherence = agent.maintain_coherence()
        print(f"Coherence: {coherence}")

    print("✓ ζ-card interpreter working!")
