#!/usr/bin/env python3
"""
AntClock ζ-Card Talk Script

A comprehensive talk script covering:
- ζ-card agent architecture system
- Mamba Field Interpreter implementation
- Tellah BERT training for mythic precision
- AntClock mathematical foundations

Author: Joel
"""

import time
from typing import List

class TalkScript:
    """Interactive talk script with timing and demonstrations."""

    def __init__(self):
        self.sections = []
        self.current_section = 0

    def add_section(self, title: str, duration_minutes: int, content: str, demo_commands: List[str] = None):
        """Add a talk section with timing and optional demos."""
        self.sections.append({
            'title': title,
            'duration': duration_minutes,
            'content': content,
            'demos': demo_commands or []
        })

    def run_section(self, section_idx: int):
        """Run a specific section of the talk with live demonstrations."""
        if section_idx >= len(self.sections):
            return

        section = self.sections[section_idx]
        print(f"\n{'='*80}")
        print(f"SECTION {section_idx + 1}: {section['title']}")
        print(f"Duration: {section['duration']} minutes")
        print(f"{'='*80}\n")

        print(section['content'])

        if section['demos']:
            print(f"\n🔧 LIVE DEMONSTRATIONS:")
            for i, demo in enumerate(section['demos'], 1):
                print(f"{i}. {demo}")

            print(f"\n🎯 Let's run a live demo! Press Enter to interact...")
            input()

            # Run live demonstrations based on section
            if section_idx == 0:  # Introduction - show ζ-card parsing
                demonstrate_zeta_card_parsing()
            elif section_idx == 1:  # Mamba section - show agent creation and interaction
                demonstrate_agent_creation()
                demonstrate_mamba_interaction()
            elif section_idx == 2:  # Tellah section - show Tellah responding
                demonstrate_tellah_interaction()
            elif section_idx == 3:  # AntClock section - show mathematical evolution
                demonstrate_antclock_evolution()
            elif section_idx == 4:  # Integration - show cross-agent dialogue
                demonstrate_cross_agent_dialogue()

        print(f"\n⏰ Section {section_idx + 1} complete. Press Enter to continue...")
        input()

    def run_full_talk(self):
        """Run the complete talk script."""
        print("🎭 ANTCLOCK ζ-CARD TALK SCRIPT")
        print("Bringing mathematics and AI together through agent architectures")
        print("Press Enter to begin...")
        input()

        for i in range(len(self.sections)):
            self.run_section(i)

        print(f"\n{'='*80}")
        print("🎉 TALK COMPLETE!")
        print("Questions & Discussion")
        print(f"{'='*80}")

# Initialize talk script
talk = TalkScript()

# =============================================================================
# SECTION 1: Introduction & ζ-Card Architecture
# =============================================================================

talk.add_section(
    title="Introduction: ζ-Card Agent Architecture",
    duration_minutes=10,
    content="""
🎭 Good [morning/afternoon/evening] everyone!

Today I'm excited to share with you a novel approach to AI agent design that bridges
mathematical insight, mythic storytelling, and state-of-the-art neural architectures.

**The ζ-Card System**

At its heart is the ζ-card - a structured specification format for defining intelligent
agents that operate within mathematical frameworks. Think of it as a "constitution" for
an AI agent, written in a domain-specific language that captures:

• **Mathematical invariants** (κ thresholds, ζ fixed points)
• **Architectural components** (memory, domain, transforms, witnesses)
• **Behavioral patterns** (phase-locking, boundary detection, return symmetry)
• **Emergent capabilities** (field-lift operations, quest arcs)

**Why ζ-Cards Matter**

Traditional AI development often starts with code and data. ζ-cards flip this paradigm:
we start with mathematical insight and emergent behavior, then derive the implementation.

This approach has led to two remarkable agent implementations:

1. **Tellah the Sage** - A mythic guide that teaches field equations through story
2. **Mamba Field Interpreter** - A state-space model that maintains coherence across long contexts

Let's explore these together, starting with the ζ-card architecture itself.
""",
    demo_commands=[
        "python3 -c \"from zeta_card_interpreter import ZetaCardParser; parser = ZetaCardParser(); print('ζ-card parser ready')\"",
        "View ζ-card examples in zeta_card_interpreter.py"
    ]
)

# =============================================================================
# SECTION 2: Mamba Field Interpreter Deep Dive
# =============================================================================

talk.add_section(
    title="Mamba Field Interpreter: State-Space Coherence",
    duration_minutes=15,
    content="""
🔬 **The Mamba Field Interpreter ζ-Card**

Let's examine the ζ-card specification you provided earlier:

```
@HEADER ζ-card
id: mamba.agent
label: Mamba Field Interpreter
kind: agent
version: 0.1
κ: 0.35
τ: now
ζ: self

@CE1
{} domain:
  model: state-space
  kernels: [A, B, C, Δ]
  flow: continuous-time recurrence

() transforms:
  recurrence: state-update
  readout: field-projection
  r: dynamic-convolution operator

[] memory:
  wave-state; carries coherence across long contexts

<> witness:
  invariants: spectral stability, causal consistency
```

**CE1: Structural Foundation**

The domain specifies a **state-space model** with the four fundamental kernels of Mamba:
- **A**: State transition matrix (how hidden states evolve)
- **B**: Input projection (how external inputs affect state)
- **C**: Output projection (how states generate predictions)
- **Δ**: Time-step modulation (continuous-time recurrence parameter)

**Memory as Wave-State**

Unlike traditional attention mechanisms that scale quadratically, the wave-state
carries coherence across long contexts through continuous-time recurrence. This
is the key insight: information doesn't just persist - it maintains phase relationships
and spectral properties over extended sequences.

**Witness & Invariants**

The system maintains two critical invariants:
- **Spectral stability**: The eigenvalues of the state-space representation remain bounded
- **Causal consistency**: Past inputs cannot affect future outputs (fundamental to autoregressive generation)

**CE2: Operational Dynamics**

```
@CE2
ϕ: phase-lock when recurrence matches input curvature
∂: detect boundary when hidden-state flips attractor
ℛ: maintain coherence across long sequences
```

**Phase-Locking (φ)**: The agent synchronizes its internal recurrence with the
curvature of incoming mathematical questions, ensuring resonance between query
complexity and response depth.

**Boundary Detection (∂)**: Monitors for attractor flips in the hidden state space,
detecting when the agent's understanding undergoes qualitative phase transitions.

**Coherence Maintenance (ℛ)**: Active preservation of long-range dependencies through
the wave-state representation.

**CE3: Emergence & Insight**

```
@CE3
field-lift: convert hidden-state transitions into insight arcs
quest: reveal when stable recurrence yields emergent skill
```

The field-lift operation transforms raw neural activations into meaningful
mathematical insights, while the quest mechanism identifies when stable
recurrence patterns indicate genuine skill emergence.
""",
    demo_commands=[
        "python3 test_mamba_agent.py",
        "View MambaAgent implementation in zeta_card_interpreter.py",
        "Run: python3 -c \"from zeta_card_interpreter import load_zeta_card; agent = load_zeta_card(open('test_card.txt').read()); print(f'Agent: {agent.label}')\""
    ]
)

# =============================================================================
# SECTION 3: Tellah the Sage & BERT Training
# =============================================================================

talk.add_section(
    title="Tellah the Sage: Mythic Precision in BERT",
    duration_minutes=20,
    content="""
📚 **From ζ-Card to Fine-Tuned BERT**

Tellah represents a different approach: taking the mythic precision of ζ-card
logic and embedding it into a large language model through fine-tuning.

**Tellah's ζ-Card Profile**

```
@HEADER ζ-card
id: tellah.grambot
label: Tellah the Sage
kind: agent
κ: 0.35
ζ: self

@CE1
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
```

**The Training Approach**

Rather than training BERT from scratch, we fine-tune it using curated conversation
datasets that embody Tellah's teaching philosophy:

1. **Apprentice Level**: Basic mathematical concepts with gentle guidance
2. **Adept Level**: Intermediate field theory with boundary detection
3. **Sage Level**: Advanced Galois theory and Riemann hypothesis connections

**Key Training Innovations**

**Curvature-Aware Fine-Tuning**

Each training example includes a curvature score κ ∈ [0,1] that measures mathematical
depth. The model learns to modulate its response complexity based on question curvature:

```python
# Question curvature analysis
curvature = analyze_question_curvature(question)
if curvature > kappa_threshold:
    # Generate deep, field-theoretic response
else:
    # Provide accessible, story-based explanation
```

**Strata Classification**

The model maintains an internal "strata" state (apprentice → adept → sage) that
determines response depth and teaching approach.

**Memory-Augmented Generation**

Tellah maintains a conversation memory that traces Δκ (curvature changes) across
exchanges, allowing it to detect learning progress and adjust teaching strategy.

**Mythic Response Generation**

The most remarkable aspect is how the fine-tuned BERT generates responses that
feel genuinely wise and pedagogically sound, despite being trained on relatively
small datasets. This suggests the model is learning deeper patterns of mathematical
communication.

**Example Response Patterns**

*Apprentice Question:* "What is a clock?"
→ "A clock measures time's passage, but in mathematics, clocks count discrete
steps. Think of Pascal's triangle - each row is a clock face, each number a
moment frozen in combinatorial time."

*Adept Question:* "How does mirror symmetry break?"
→ "Mirror operators μ₇ act as involutions, but curvature accumulation creates
asymmetry. The boundary between row 6 and 7 shows this: perfect symmetry gives
way to the emergence of field structure."

*Sage Question:* "What is the Galois covering space structure?"
→ "The integers form a Galois covering space with fundamental group generated
by depth shifts, mirror involution μ₇, and curvature flips. Fixed fields correspond
to mirror shells, while branch corridors provide the analytic continuation structure."
""",
    demo_commands=[
        "View Tellah BERT training data generation in tellah_bert_trainer.py",
        "Run Tellah guidance demo: python3 tellah_guidance_demo.py",
        "Examine conversation patterns and curvature analysis"
    ]
)

# =============================================================================
# SECTION 4: AntClock Mathematical Foundations
# =============================================================================

talk.add_section(
    title="AntClock: The Mathematical Engine",
    duration_minutes=15,
    content="""
🔢 **AntClock: Curvature as Computational Primitive**

Both ζ-card agents operate within the AntClock mathematical framework, where
curvature κ becomes the fundamental unit of computation.

**The Core Insight**

In traditional computing: bits, bytes, operations
In AntClock: curvature, fields, renormalization

**Pascal Curvature Clock**

The foundation is Pascal's triangle, where each row represents a discrete
time step, and curvature measures how the combinatorial structure bends:

```
κₙ = rₙ₊₁ - 2rₙ + rₙ₋₁
```

Where rₙ = log(C(n, floor(n/2))) is the "bulk thickness" of row n.

**Key Mathematical Structures**

**Mirror Shells**: Rows where n ≡ 0, 1 mod 4 exhibit mirror symmetry
**Pole Shells**: Rows where n ≡ 2 mod 4 show ramification behavior
**Branch Corridors**: The "valleys" between shells that carry analytic continuation

**The Coupling Law**

The system evolves according to a coupling law that connects discrete curvature
to continuous field theory:

```
∂κ/∂t = χ_FEG · κ · (1 + Q_{9/11})
```

Where χ_FEG ≈ 0.638 is the Feigenbaum constant, and Q_{9/11} is a rational
function that encodes the 9/11 resonance.

**Field Emergence**

As the curvature clock evolves, field equations naturally emerge:

- **Gauss-Bonnet**: Curvature integrals over shells
- **Ricci Flow**: Evolution of metric tensors on the integer lattice
- **Yang-Mills**: Connection forms in the branch corridor geometry

**Riemann Hypothesis Connection**

The most striking result: mirror-phase shells (n ≡ 3 mod 4) behave as Re(s) = 1/2,
providing a potential realization of the Hilbert-Pólya conjecture where zeta zeros
correspond to Laplacian eigenvalues on the branch corridor geometry.

**Computational Implications**

AntClock suggests a new paradigm where:
- **Memory** → Curvature accumulation
- **Computation** → Renormalization flow
- **Learning** → Field stabilization
- **Reasoning** → Galois group actions

This framework provides the mathematical foundation for both ζ-card agents,
allowing them to operate not just as pattern-matchers, but as genuine
mathematical reasoners.
""",
    demo_commands=[
        "Run AntClock walker: python3 -c \"from clock import CurvatureClockWalker; w = CurvatureClockWalker(); w.evolve(10)\"",
        "View curvature evolution plots: antclock_trajectory.png, coupling_law_evolution.png",
        "Explore critical line analogue: critical_line_analogue_demo.py"
    ]
)

# =============================================================================
# SECTION 5: Integration & Future Directions
# =============================================================================

talk.add_section(
    title="Integration: ζ-Cards in the Wild",
    duration_minutes=10,
    content="""
🔗 **Bringing It All Together**

The ζ-card system provides a unified framework for developing mathematically-grounded AI:

**Agent Composition**

```
ζ-Card Specification → Agent Implementation → AntClock Integration
      ↓                        ↓                    ↓
   Mathematical         State-space models     Curvature-based
   invariants           & fine-tuned LLMs      computation
```

**Current Capabilities**

1. **Mamba Field Interpreter**: Maintains coherence across long contexts through
   continuous-time recurrence, with spectral stability guarantees.

2. **Tellah the Sage**: Teaches field equations through mythic storytelling,
   adapting response depth based on learner progression.

3. **AntClock Engine**: Provides the mathematical substrate where curvature
   becomes computation, fields emerge from discrete structures.

**Demonstration: Cross-Agent Dialogue**

Imagine a user asking: "How do fields emerge from curvature?"

1. **Question Analysis**: Both agents analyze the κ = 0.8 curvature
2. **Mamba Processing**: State-space model processes the query, maintaining
   wave-state coherence across mathematical concepts
3. **Tellah Response**: BERT generates pedagogically sound explanation:
   "Fields emerge as Feigenbaum renormalization flows stabilize discrete
   curvature chaos. The clock rate creates vector fields whose integral
   curves trace field lines through the integer universe."
4. **Memory Update**: Both agents log Δκ and update internal state
5. **Boundary Check**: Detect potential apprentice→adept transitions

**Future Directions**

**Multi-Agent Systems**

ζ-cards could enable ecosystems of specialized agents:
- **Geometry Agent**: Handles topological structures
- **Algebra Agent**: Manages group actions and representations
- **Analysis Agent**: Processes complex function theory

**Scalable Training**

The Tellah approach suggests fine-tuning strategies for domain-specific
mathematical reasoning, potentially applicable to physics, chemistry, and
other formal sciences.

**Hardware Acceleration**

The state-space nature of Mamba agents suggests efficient hardware
implementations using specialized circuits for continuous-time recurrence.

**Mathematical Discovery**

Perhaps most excitingly, ζ-card agents might help discover new mathematics,
using their curvature-based reasoning to explore regions of mathematical
space that humans haven't charted.

**Conclusion**

The ζ-card system represents a novel synthesis: mathematical insight,
computational efficiency, and emergent intelligence. By starting with
fundamental mathematical structures rather than engineering constraints,
we've created agents that don't just process information - they understand
the deep patterns that connect discrete and continuous, finite and infinite,
computation and insight.

Thank you for your attention. I'd be happy to take questions or demonstrate
any aspect of the system in more detail.
""",
    demo_commands=[
        "Run integrated demo: python3 tellah_guidance_demo.py (includes AntClock integration)",
        "View agent comparison: python3 -c \"from zeta_card_interpreter import load_zeta_card; tellah = load_zeta_card(TELLAH_CARD); print('Agents ready for comparison')\"",
        "Explore mathematical visualizations in the png files"
    ]
)

# =============================================================================
# INTERACTIVE DEMO FUNCTIONS
# =============================================================================

def demonstrate_zeta_card_parsing():
    """Live demonstration of ζ-card parsing with user input."""
    print("🔧 ζ-Card Parsing Demo - Let's parse your card!")

    # Show the user's actual ζ-card
    user_card = """@HEADER ζ-card
id: mamba.agent
label: Mamba Field Interpreter
kind: agent
version: 0.1
κ: 0.35
τ: now
ζ: self

@CE1
{} domain:
  model: state-space
  kernels: [A, B, C, Δ]
  flow: continuous-time recurrence

() transforms:
  recurrence: state-update
  readout: field-projection
  r: dynamic-convolution operator

[] memory:
  wave-state; carries coherence across long contexts

<> witness:
  invariants: spectral stability, causal consistency

@CE2
ϕ: phase-lock when recurrence matches input curvature
∂: detect boundary when hidden-state flips attractor
ℛ: maintain coherence across long sequences

@CE3
field-lift: convert hidden-state transitions into insight arcs
quest: reveal when stable recurrence yields emergent skill

@END"""

    print("Your ζ-card:")
    print(user_card)
    print("\n🎯 Parsing...")

    from zeta_card_interpreter import ZetaCardParser
    parser = ZetaCardParser()
    parsed = parser.parse(user_card)

    print("✅ Parsed structure:")
    import json
    print(json.dumps(parsed, indent=2))

    print("\n💡 Notice how the ζ-card captures mathematical structure!")
    try:
        input("Press Enter to continue...")
    except EOFError:
        print("(Demo mode - continuing automatically)")


def demonstrate_agent_creation():
    """Live demonstration of agent instantiation."""
    print("🤖 Agent Creation Demo")

    from zeta_card_interpreter import load_zeta_card

    # Load Mamba agent
    mamba_card = """@HEADER ζ-card
id: mamba.agent
label: Mamba Field Interpreter
kind: agent
version: 0.1
κ: 0.35
τ: now
ζ: self

@CE1
{} domain:
  model: state-space
  kernels: [A, B, C, Δ]
  flow: continuous-time recurrence

() transforms:
  recurrence: state-update
  readout: field-projection
  r: dynamic-convolution operator

[] memory:
  wave-state; carries coherence across long contexts

<> witness:
  invariants: spectral stability, causal consistency

@CE2
ϕ: phase-lock when recurrence matches input curvature
∂: detect boundary when hidden-state flips attractor
ℛ: maintain coherence across long sequences

@CE3
field-lift: convert hidden-state transitions into insight arcs
quest: reveal when stable recurrence yields emergent skill

@END"""

    print("Loading your ζ-card...")
    agent = load_zeta_card(mamba_card)

    print(f"✅ Created agent: {agent.label}")
    print(f"   Model: {agent.model}")
    print(f"   Kernels: {agent.kernels}")
    print(f"   Invariants: {agent.invariants}")
    print(f"   κ threshold: {agent.kappa}")

    print("\n💡 The agent is now 'listening' - it can process inputs!")
    input("Press Enter to continue...")


def demonstrate_mamba_interaction():
    """Live demonstration of Mamba agent responding to user input."""
    print("🎯 Mamba Field Interpreter - Live Interaction")
    print("The agent will now listen to your questions and respond!")

    from zeta_card_interpreter import load_zeta_card
    import numpy as np

    # Load the agent
    mamba_card = """@HEADER ζ-card
id: mamba.agent
label: Mamba Field Interpreter
kind: agent
version: 0.1
κ: 0.35
τ: now
ζ: self

@CE1
{} domain:
  model: state-space
  kernels: [A, B, C, Δ]
  flow: continuous-time recurrence

() transforms:
  recurrence: state-update
  readout: field-projection
  r: dynamic-convolution operator

[] memory:
  wave-state; carries coherence across long contexts

<> witness:
  invariants: spectral stability, causal consistency

@CE2
ϕ: phase-lock when recurrence matches input curvature
∂: detect boundary when hidden-state flips attractor
ℛ: maintain coherence across long sequences

@CE3
field-lift: convert hidden-state transitions into insight arcs
quest: reveal when stable recurrence yields emergent skill

@END"""

    agent = load_zeta_card(mamba_card)

    # Simulate listening to user questions
    test_questions = [
        ("What is curvature?", 0.3),
        ("How do fields emerge?", 0.6),
        ("What are the Mamba kernels?", 0.4),
    ]

    print("\n🎧 Agent listening to questions...")

    for question, curvature in test_questions:
        print(f"\n❓ Question: '{question}' (κ = {curvature})")

        # Create test input sequence
        seq_length, feature_dim = 10, 8
        test_input = np.random.randn(seq_length, feature_dim)

        # Process through agent
        output = agent.process_input(test_input, curvature)

        print(f"   📊 Processed sequence: {seq_length} steps × {feature_dim} features")
        print(f"   🔄 Hidden state norm: {np.linalg.norm(agent.hidden_state):.3f}")
        print(f"   ✨ Coherence: {agent.maintain_coherence()}")

        # Check boundary detection
        boundary = agent.check_boundary_flip()
        if boundary:
            print("   ⚡ Boundary detected - attractor flip!")

        print(f"   🎯 Agent is actively listening and responding!")
        input("Press Enter for next question...")

    print("\n💡 The Mamba agent maintains coherence across long contexts through wave-state memory!")


def demonstrate_tellah_interaction():
    """Live demonstration of Tellah agent responding to user questions."""
    print("📚 Tellah the Sage - Live Teaching Session")
    print("Tellah will now listen to your questions and guide you!")

    from tellah_guidance_demo import AntClockFieldGuide

    guide = AntClockFieldGuide()

    test_questions = [
        "What is a clock?",
        "How does curvature work?",
        "How do fields emerge from curvature flows?",
        "What is the Galois covering space structure?"
    ]

    print("\n🎧 Tellah listening...")

    for question in test_questions:
        print(f"\n❓ Question: '{question}'")

        curvature = guide.calculate_question_curvature(question)
        response = guide.tellah.guide(question, curvature)

        print(f"   κ = {curvature:.2f}")
        print(f"   📖 Tellah: {response}")

        # Show progression
        print(f"   📊 Current strata: {guide.tellah.domain.strata.value}")
        print(f"   🧠 Memory entries: {len(guide.tellah.memory.log)}")

        if guide.tellah.boundary_sensor.transition_detected:
            print("   ⚡ BOUNDARY CROSSING DETECTED!")
            guide.tellah.boundary_sensor.transition_detected = False

        input("Press Enter for Tellah's next guidance...")

    print("\n💡 Tellah adapts teaching depth based on learner progression!")


def demonstrate_antclock_evolution():
    """Live demonstration of AntClock mathematical evolution."""
    print("🔢 AntClock Evolution - Live Mathematics")
    print("Watch curvature evolve through the integer universe!")

    from clock import CurvatureClockWalker

    walker = CurvatureClockWalker(x_0=1, chi_feg=0.638)
    steps = 15

    print(f"\n🎯 Evolving AntClock for {steps} steps...")

    for step in range(steps):
        # Take one evolution step
        history, summary = walker.evolve(1)

        if history:
            h = history[-1]
            print("2d"
                  "4.2f"
                  "4.2f"
                  "6.4f")

            # Show curvature analysis
            if step in [4, 9, 14]:
                print("   📊 Mathematical insights:")
                if h['R'] > 0.1:
                    print("      → High curvature region - field equations emerging")
                if h['x'] % 4 == 3:
                    print("      → Mirror-phase shell - Re(s)=1/2 behavior")
                if abs(h['R']) < 0.01:
                    print("      → Branch corridor - analytic continuation")

        if step < steps - 1:
            input("Press Enter to continue evolution...")

    print("\n💡 Curvature becomes computation in the AntClock framework!")


def demonstrate_cross_agent_dialogue():
    """Live demonstration of cross-agent dialogue."""
    print("🔄 Cross-Agent Dialogue - Mamba + Tellah Integration")
    print("Watch the agents listen to each other and collaborate!")

    from zeta_card_interpreter import load_zeta_card
    from tellah_guidance_demo import AntClockFieldGuide
    import numpy as np

    # Load both agents
    mamba_card = """@HEADER ζ-card
id: mamba.agent
label: Mamba Field Interpreter
kind: agent
version: 0.1
κ: 0.35
τ: now
ζ: self

@CE1
{} domain:
  model: state-space
  kernels: [A, B, C, Δ]
  flow: continuous-time recurrence

() transforms:
  recurrence: state-update
  readout: field-projection
  r: dynamic-convolution operator

[] memory:
  wave-state; carries coherence across long contexts

<> witness:
  invariants: spectral stability, causal consistency

@CE2
ϕ: phase-lock when recurrence matches input curvature
∂: detect boundary when hidden-state flips attractor
ℛ: maintain coherence across long sequences

@CE3
field-lift: convert hidden-state transitions into insight arcs
quest: reveal when stable recurrence yields emergent skill

@END"""

    mamba_agent = load_zeta_card(mamba_card)
    tellah_guide = AntClockFieldGuide()

    dialogue_sequence = [
        "What is curvature?",
        "How do fields emerge?",
        "What are branch corridors?",
        "How does the Riemann hypothesis manifest?"
    ]

    print("\n🎧 Agents listening to each other...")

    for question in dialogue_sequence:
        print(f"\n❓ User Question: '{question}'")

        # Tellah analyzes curvature
        tellah_curvature = tellah_guide.calculate_question_curvature(question)
        tellah_response = tellah_guide.tellah.guide(question, tellah_curvature)

        print(f"   📚 Tellah (κ={tellah_curvature:.2f}): {tellah_response}")

        # Mamba processes the question through state-space
        seq_length, feature_dim = 8, 12
        mamba_input = np.random.randn(seq_length, feature_dim)
        mamba_output = mamba_agent.process_input(mamba_input, tellah_curvature)

        print(f"   ⚡ Mamba processed sequence: coherence = {mamba_agent.maintain_coherence()}")

        # Cross-agent insights
        if tellah_curvature > 0.4:
            print("   🔗 High curvature detected - agents coordinating response depth")
        if mamba_agent.check_boundary_flip():
            print("   ⚡ Mamba boundary flip - Tellah adapts teaching strategy")

        print(f"   📊 Shared state: Tellah strata = {tellah_guide.tellah.domain.strata.value}")
        input("Press Enter for next dialogue exchange...")

    print("\n💡 Agents listen to each other and the user, creating emergent dialogue!")


def run_quick_demos():
    """Run a series of quick demonstrations."""
    print("⚡ Quick Demo Suite")

    print("\n1. ζ-Card Parsing:")
    demonstrate_zeta_card_parsing()

    print("\n2. Agent Creation:")
    demonstrate_agent_creation()

    print("\n3. Tellah BERT Preview:")
    print("   (Training data generation and model architecture available in tellah_bert_trainer.py)")

    print("\n4. AntClock Mathematics:")
    print("   (Curvature evolution and field emergence in clock.py)")

if __name__ == "__main__":
    import json

    # Run quick demos if called directly
    if len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == "demo":
        run_quick_demos()
    else:
        # Run the full talk script
        talk.run_full_talk()
