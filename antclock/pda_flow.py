#!run.sh
"""
CE1::PDA-FLOW — Opcode Set (v1)

PDA-FLOW opcodes as CE1 morphism-generators with bracket-topology,
morphism generators, and clean operational semantics.

Each opcode is written as a CE1 morphism-generator:
  OP_<name>(inputs) → <witness> [effects] {scope modifications}
and can be chained with >> (sequential) or ⊕ (resonant blend).
"""

from typing import Dict, Any, List, Optional, Callable, Protocol, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math


# ============================================================================
# PDA-FLOW State and Context
# ============================================================================

@dataclass
class PDAFlowState:
    """
    State maintained across PDA-FLOW opcode execution.
    
    Encodes the task cycle state, autonomy metrics, and scope modifications.
    """
    # Task ordering
    task_queue: List[Any] = field(default_factory=list)
    hard_task: Optional[Any] = None
    current_position: int = 0
    
    # Reward envelope
    reward_before: Optional[Any] = None
    reward_after: Optional[Any] = None
    reward_envelope_intact: bool = True
    
    # Timing constraints
    interval_window: Optional[Tuple[float, float]] = None  # (start, end) in minutes
    slack_window: Optional[Tuple[float, float]] = None
    
    # Autonomy metrics
    autonomy_baseline: float = 1.0
    autonomy_resistance: float = 0.0
    blockback_detected: bool = False
    blockback_threshold: float = 0.5
    
    # Task scope
    task_scope: List[Any] = field(default_factory=list)
    task_velocity: float = 1.0
    timing_band_width: float = 1.0
    
    # Recovery state
    activation_level: float = 1.0
    residual_tension: float = 0.0
    
    # Witness tracking (CE1 style)
    witness: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PDAFlowWitness:
    """
    CE1 witness object for PDA-FLOW operations.
    
    Captures invariants and resonant behaviors across the opcode chain.
    """
    opcode_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    effects: List[str]
    scope_modifications: Dict[str, Any]
    resonant_frequency: Optional[float] = None  # For resonant blend operations


# ============================================================================
# Base Opcode Protocol
# ============================================================================

class PDAFlowOpcode(Protocol):
    """
    Protocol for PDA-FLOW opcodes as CE1 morphism-generators.
    
    Each opcode transforms state and produces a witness.
    """
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        """
        Execute opcode as morphism: state → (new_state, witness)
        
        Returns:
            Tuple of (updated_state, witness_object)
        """
        ...


# ============================================================================
# Opcode Implementations
# ============================================================================

class OP_FRONTLOAD:
    """
    OP_FRONTLOAD()
    
    <frontload> 
    [requires: HARD_TASK]
    [effect: push(HARD_TASK) into first-position]
    {
      reorder.scope := bias(aversion↑, reward↓)
    }
    
    Meaning: Move the hard task to the front of the cycle. Raises the
    "aversion-stress amplitude" temporarily but reduces long-tail anxiety.
    """
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        if state.hard_task is None:
            # No hard task to frontload
            witness = PDAFlowWitness(
                opcode_name="OP_FRONTLOAD",
                inputs={"hard_task": None},
                outputs={"reordered": False},
                effects=[],
                scope_modifications={}
            )
            return state, witness
        
        # Reorder: move hard task to front
        new_queue = [state.hard_task]
        for task in state.task_queue:
            if task != state.hard_task:
                new_queue.append(task)
        
        # Update state
        new_state = PDAFlowState(
            task_queue=new_queue,
            hard_task=state.hard_task,
            current_position=0,
            # Bias: aversion↑, reward↓
            autonomy_resistance=min(1.0, state.autonomy_resistance + 0.1),
            # Copy other fields
            reward_before=state.reward_before,
            reward_after=state.reward_after,
            reward_envelope_intact=state.reward_envelope_intact,
            interval_window=state.interval_window,
            slack_window=state.slack_window,
            autonomy_baseline=state.autonomy_baseline,
            blockback_detected=state.blockback_detected,
            blockback_threshold=state.blockback_threshold,
            task_scope=state.task_scope,
            task_velocity=state.task_velocity,
            timing_band_width=state.timing_band_width,
            activation_level=state.activation_level,
            residual_tension=state.residual_tension,
            witness=state.witness.copy()
        )
        
        witness = PDAFlowWitness(
            opcode_name="OP_FRONTLOAD",
            inputs={"hard_task": state.hard_task, "queue_length": len(state.task_queue)},
            outputs={"reordered": True, "new_queue_length": len(new_queue)},
            effects=["push(HARD_TASK) into first-position", "aversion↑", "reward↓"],
            scope_modifications={"reorder.scope": "bias(aversion↑, reward↓)"}
        )
        
        return new_state, witness


class OP_SANDWICH:
    """
    OP_SANDWICH(reward)
    
    <buffer>
    [effect: insert(reward.before), insert(reward.after)]
    {
      morphism: create reward-envelope around HARD_TASK
    }
    
    Meaning: Wrap the main task in a reward bracket. Creates a dopamine buffer.
    """
    
    def __init__(self, reward: Any):
        self.reward = reward
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        # Create reward envelope around hard task
        new_state = PDAFlowState(
            task_queue=state.task_queue,
            hard_task=state.hard_task,
            current_position=state.current_position,
            reward_before=self.reward,
            reward_after=self.reward,
            reward_envelope_intact=True,
            # Copy other fields
            interval_window=state.interval_window,
            slack_window=state.slack_window,
            autonomy_baseline=state.autonomy_baseline,
            autonomy_resistance=state.autonomy_resistance,
            blockback_detected=state.blockback_detected,
            blockback_threshold=state.blockback_threshold,
            task_scope=state.task_scope,
            task_velocity=state.task_velocity,
            timing_band_width=state.timing_band_width,
            activation_level=state.activation_level,
            residual_tension=state.residual_tension,
            witness=state.witness.copy()
        )
        
        witness = PDAFlowWitness(
            opcode_name="OP_SANDWICH",
            inputs={"reward": self.reward},
            outputs={"reward_envelope_created": True},
            effects=["insert(reward.before)", "insert(reward.after)"],
            scope_modifications={"morphism": "create reward-envelope around HARD_TASK"}
        )
        
        return new_state, witness


class OP_INTERVAL:
    """
    OP_INTERVAL(timer)
    
    <timed>
    [effect: bind(timer.window)]
    {
      attenuate(anticipation), compress(anxiety-spectrum)
    }
    
    Meaning: Bind execution to an interval window (10–25m, elastic). 
    Reduces anticipatory threat.
    """
    
    def __init__(self, timer: Tuple[float, float]):
        """
        Args:
            timer: (start_minutes, end_minutes) window
        """
        self.timer = timer
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        # Bind to interval window
        new_state = PDAFlowState(
            task_queue=state.task_queue,
            hard_task=state.hard_task,
            current_position=state.current_position,
            reward_before=state.reward_before,
            reward_after=state.reward_after,
            reward_envelope_intact=state.reward_envelope_intact,
            interval_window=self.timer,
            # Attenuate anticipation, compress anxiety
            autonomy_resistance=max(0.0, state.autonomy_resistance - 0.1),
            # Copy other fields
            slack_window=state.slack_window,
            autonomy_baseline=state.autonomy_baseline,
            blockback_detected=state.blockback_detected,
            blockback_threshold=state.blockback_threshold,
            task_scope=state.task_scope,
            task_velocity=state.task_velocity,
            timing_band_width=state.timing_band_width,
            activation_level=state.activation_level,
            residual_tension=max(0.0, state.residual_tension - 0.1),
            witness=state.witness.copy()
        )
        
        witness = PDAFlowWitness(
            opcode_name="OP_INTERVAL",
            inputs={"timer": self.timer},
            outputs={"interval_bound": True, "window": self.timer},
            effects=["bind(timer.window)"],
            scope_modifications={
                "attenuate": "anticipation",
                "compress": "anxiety-spectrum"
            }
        )
        
        return new_state, witness


class OP_FLEX:
    """
    OP_FLEX(slack)
    
    <flex>
    [effect: allocate(slack.window)]
    {
      allow autonomy-rebound; pre-authorize deflection without penalty
    }
    
    Meaning: Pre-allocate wiggle space for PDA blowback.
    """
    
    def __init__(self, slack: Tuple[float, float]):
        """
        Args:
            slack: (start_minutes, end_minutes) slack window
        """
        self.slack = slack
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        # Allocate slack window
        new_state = PDAFlowState(
            task_queue=state.task_queue,
            hard_task=state.hard_task,
            current_position=state.current_position,
            reward_before=state.reward_before,
            reward_after=state.reward_after,
            reward_envelope_intact=state.reward_envelope_intact,
            interval_window=state.interval_window,
            slack_window=self.slack,
            # Allow autonomy rebound
            autonomy_baseline=max(state.autonomy_baseline, 0.8),
            # Copy other fields
            autonomy_resistance=state.autonomy_resistance,
            blockback_detected=state.blockback_detected,
            blockback_threshold=state.blockback_threshold,
            task_scope=state.task_scope,
            task_velocity=state.task_velocity,
            timing_band_width=state.timing_band_width,
            activation_level=state.activation_level,
            residual_tension=state.residual_tension,
            witness=state.witness.copy()
        )
        
        witness = PDAFlowWitness(
            opcode_name="OP_FLEX",
            inputs={"slack": self.slack},
            outputs={"slack_allocated": True, "window": self.slack},
            effects=["allocate(slack.window)"],
            scope_modifications={
                "allow": "autonomy-rebound",
                "pre-authorize": "deflection without penalty"
            }
        )
        
        return new_state, witness


class OP_BLOCKBACK_CHECK:
    """
    OP_BLOCKBACK_CHECK()
    
    <blockback?>
    [effect: detect(autonomy-resistance)]
    {
      emit SIGNAL_BLOCKBACK if threshold crossed
    }
    """
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        # Detect autonomy resistance
        blockback_detected = state.autonomy_resistance >= state.blockback_threshold
        
        new_state = PDAFlowState(
            task_queue=state.task_queue,
            hard_task=state.hard_task,
            current_position=state.current_position,
            reward_before=state.reward_before,
            reward_after=state.reward_after,
            reward_envelope_intact=state.reward_envelope_intact,
            interval_window=state.interval_window,
            slack_window=state.slack_window,
            autonomy_baseline=state.autonomy_baseline,
            autonomy_resistance=state.autonomy_resistance,
            blockback_detected=blockback_detected,
            blockback_threshold=state.blockback_threshold,
            # Copy other fields
            task_scope=state.task_scope,
            task_velocity=state.task_velocity,
            timing_band_width=state.timing_band_width,
            activation_level=state.activation_level,
            residual_tension=state.residual_tension,
            witness=state.witness.copy()
        )
        
        effects = ["detect(autonomy-resistance)"]
        if blockback_detected:
            effects.append("emit SIGNAL_BLOCKBACK")
        
        witness = PDAFlowWitness(
            opcode_name="OP_BLOCKBACK_CHECK",
            inputs={"autonomy_resistance": state.autonomy_resistance, 
                   "threshold": state.blockback_threshold},
            outputs={"blockback_detected": blockback_detected},
            effects=effects,
            scope_modifications={}
        )
        
        return new_state, witness


class OP_SLOW:
    """
    OP_SLOW()
    
    <modulate>
    [requires: SIGNAL_BLOCKBACK]
    {
      reduce(task-velocity), widen(timing-band)
    }
    
    Meaning: Slow the task-push when resistance is detected.
    """
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        if not state.blockback_detected:
            # No blockback signal, no-op
            witness = PDAFlowWitness(
                opcode_name="OP_SLOW",
                inputs={"blockback_detected": False},
                outputs={"modulated": False},
                effects=[],
                scope_modifications={}
            )
            return state, witness
        
        # Reduce velocity, widen timing band
        new_state = PDAFlowState(
            task_queue=state.task_queue,
            hard_task=state.hard_task,
            current_position=state.current_position,
            reward_before=state.reward_before,
            reward_after=state.reward_after,
            reward_envelope_intact=state.reward_envelope_intact,
            interval_window=state.interval_window,
            slack_window=state.slack_window,
            autonomy_baseline=state.autonomy_baseline,
            autonomy_resistance=state.autonomy_resistance,
            blockback_detected=state.blockback_detected,
            blockback_threshold=state.blockback_threshold,
            task_scope=state.task_scope,
            task_velocity=max(0.1, state.task_velocity * 0.7),  # Reduce velocity
            timing_band_width=state.timing_band_width * 1.5,  # Widen timing band
            activation_level=state.activation_level,
            residual_tension=state.residual_tension,
            witness=state.witness.copy()
        )
        
        witness = PDAFlowWitness(
            opcode_name="OP_SLOW",
            inputs={"blockback_detected": True, 
                   "current_velocity": state.task_velocity,
                   "current_timing_band": state.timing_band_width},
            outputs={"velocity_reduced": True, 
                    "timing_band_widened": True,
                    "new_velocity": new_state.task_velocity,
                    "new_timing_band": new_state.timing_band_width},
            effects=["reduce(task-velocity)", "widen(timing-band)"],
            scope_modifications={}
        )
        
        return new_state, witness


class OP_DROP_WEIGHT:
    """
    OP_DROP_WEIGHT()
    
    <unload>
    [requires: SIGNAL_BLOCKBACK]
    {
      shrink.task-scope := drop non-essential subtasks
    }
    
    Meaning: Lighten scope. Offload complexity to reduce the autonomy threat.
    """
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        if not state.blockback_detected:
            # No blockback signal, no-op
            witness = PDAFlowWitness(
                opcode_name="OP_DROP_WEIGHT",
                inputs={"blockback_detected": False},
                outputs={"unloaded": False},
                effects=[],
                scope_modifications={}
            )
            return state, witness
        
        # Shrink task scope: drop non-essential subtasks
        # Keep only essential tasks (first 50% or minimum 1)
        essential_count = max(1, len(state.task_scope) // 2)
        new_scope = state.task_scope[:essential_count]
        
        new_state = PDAFlowState(
            task_queue=state.task_queue,
            hard_task=state.hard_task,
            current_position=state.current_position,
            reward_before=state.reward_before,
            reward_after=state.reward_after,
            reward_envelope_intact=state.reward_envelope_intact,  # INV3: envelope intact
            interval_window=state.interval_window,
            slack_window=state.slack_window,
            autonomy_baseline=state.autonomy_baseline,
            autonomy_resistance=max(0.0, state.autonomy_resistance - 0.2),  # Relief
            blockback_detected=state.blockback_detected,
            blockback_threshold=state.blockback_threshold,
            task_scope=new_scope,
            task_velocity=state.task_velocity,
            timing_band_width=state.timing_band_width,
            activation_level=state.activation_level,
            residual_tension=max(0.0, state.residual_tension - 0.15),
            witness=state.witness.copy()
        )
        
        witness = PDAFlowWitness(
            opcode_name="OP_DROP_WEIGHT",
            inputs={"blockback_detected": True, 
                   "original_scope_size": len(state.task_scope)},
            outputs={"scope_shrunk": True, 
                    "new_scope_size": len(new_scope),
                    "dropped_tasks": len(state.task_scope) - len(new_scope)},
            effects=["shrink.task-scope", "drop non-essential subtasks"],
            scope_modifications={"shrink.task-scope": "drop non-essential subtasks"}
        )
        
        return new_state, witness


class OP_RECOVERY:
    """
    OP_RECOVERY()
    
    <rebind>
    [effect: normalize(activation), restore(autonomy-baseline)]
    {
      prep for next cycle; resolve reward-envelope
    }
    
    Meaning: Smooth the end of the cycle and discharge residual tension.
    """
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        # Normalize activation, restore autonomy baseline
        new_state = PDAFlowState(
            task_queue=state.task_queue,
            hard_task=state.hard_task,
            current_position=state.current_position,
            reward_before=state.reward_before,
            reward_after=state.reward_after,
            reward_envelope_intact=state.reward_envelope_intact,
            interval_window=state.interval_window,
            slack_window=state.slack_window,
            autonomy_baseline=1.0,  # Restore baseline
            autonomy_resistance=max(0.0, state.autonomy_resistance - 0.3),  # Discharge
            blockback_detected=False,  # Reset blockback
            blockback_threshold=state.blockback_threshold,
            task_scope=state.task_scope,
            task_velocity=1.0,  # Reset velocity
            timing_band_width=1.0,  # Reset timing band
            activation_level=1.0,  # Normalize activation
            residual_tension=0.0,  # Discharge tension
            witness=state.witness.copy()
        )
        
        witness = PDAFlowWitness(
            opcode_name="OP_RECOVERY",
            inputs={"activation": state.activation_level,
                   "residual_tension": state.residual_tension,
                   "autonomy_baseline": state.autonomy_baseline},
            outputs={"activation_normalized": True,
                    "autonomy_baseline_restored": True,
                    "residual_tension_discharged": True},
            effects=["normalize(activation)", "restore(autonomy-baseline)"],
            scope_modifications={
                "prep": "for next cycle",
                "resolve": "reward-envelope"
            }
        )
        
        return new_state, witness


# ============================================================================
# Opcode Chain Composition
# ============================================================================

class OpcodeChain:
    """
    Composable chain of PDA-FLOW opcodes.
    
    Supports:
    - >> (sequential composition)
    - ⊕ (resonant blend - system chooses based on blockback amplitude)
    """
    
    def __init__(self, opcodes: List[PDAFlowOpcode]):
        self.opcodes = opcodes
    
    def __rshift__(self, other: 'OpcodeChain') -> 'OpcodeChain':
        """Sequential composition: chain >> other"""
        return OpcodeChain(self.opcodes + other.opcodes)
    
    def __call__(self, state: PDAFlowState) -> Tuple[PDAFlowState, List[PDAFlowWitness]]:
        """
        Execute the opcode chain sequentially.
        
        Returns:
            Tuple of (final_state, list_of_witnesses)
        """
        witnesses = []
        current_state = state
        
        for opcode in self.opcodes:
            current_state, witness = opcode(current_state)
            witnesses.append(witness)
        
        return current_state, witnesses


def resonant_blend(opcode1: PDAFlowOpcode, opcode2: PDAFlowOpcode) -> PDAFlowOpcode:
    """
    Resonant blend: ⊕ operator
    
    System chooses between opcode1 and opcode2 based on blockback amplitude.
    Higher blockback → choose opcode2 (usually DROP_WEIGHT).
    Lower blockback → choose opcode1 (usually SLOW).
    """
    def blended_opcode(state: PDAFlowState) -> Tuple[PDAFlowState, PDAFlowWitness]:
        # Choose based on blockback amplitude
        if state.blockback_detected and state.autonomy_resistance > 0.7:
            # High blockback: prefer opcode2 (usually DROP_WEIGHT)
            new_state, witness = opcode2(state)
            witness.opcode_name = f"RESONANT_BLEND({opcode1.__class__.__name__} ⊕ {opcode2.__class__.__name__})"
            witness.resonant_frequency = state.autonomy_resistance
            return new_state, witness
        else:
            # Lower blockback: prefer opcode1 (usually SLOW)
            new_state, witness = opcode1(state)
            witness.opcode_name = f"RESONANT_BLEND({opcode1.__class__.__name__} ⊕ {opcode2.__class__.__name__})"
            witness.resonant_frequency = state.autonomy_resistance
            return new_state, witness
    
    return blended_opcode


# ============================================================================
# PDA-FLOW Chain Template
# ============================================================================

def PDA_FLOW(reward: Any, timer: Tuple[float, float], slack: Tuple[float, float]) -> OpcodeChain:
    """
    Complete PDA-FLOW opcode chain template.
    
    PDA_FLOW :=
      OP_FRONTLOAD()
      >> OP_SANDWICH(reward)
      >> OP_INTERVAL(timer)
      >> OP_FLEX(slack)
      >> ( OP_BLOCKBACK_CHECK()
             >> { OP_SLOW() ⊕ OP_DROP_WEIGHT() } )?
      >> OP_RECOVERY()
    
    Args:
        reward: Reward object for sandwich envelope
        timer: (start, end) interval window in minutes
        slack: (start, end) slack window in minutes
    
    Returns:
        Composed OpcodeChain ready for execution
    """
    # Build the chain
    chain = OpcodeChain([
        OP_FRONTLOAD(),
        OP_SANDWICH(reward),
        OP_INTERVAL(timer),
        OP_FLEX(slack),
        OP_BLOCKBACK_CHECK(),
        resonant_blend(OP_SLOW(), OP_DROP_WEIGHT()),
        OP_RECOVERY()
    ])
    
    return chain


# ============================================================================
# CE1 Invariants Verification
# ============================================================================

def verify_invariants(state: PDAFlowState, witnesses: List[PDAFlowWitness]) -> Dict[str, bool]:
    """
    Verify CE1 invariants for PDA-FLOW execution.
    
    INV1: Blockback = signal, not failure.
    INV2: Autonomy-preservation ≥ throughput.
    INV3: Reward-envelope remains intact even under scope-shrink.
    INV4: Interval timing dominates anxiety-spectrum.
    INV5: Flex allocation cannot be retroactively removed.
    """
    results = {}
    
    # INV1: Blockback = signal, not failure
    blockback_occurred = any(w.opcode_name == "OP_BLOCKBACK_CHECK" and 
                             w.outputs.get("blockback_detected", False) 
                             for w in witnesses)
    if blockback_occurred:
        # Check that recovery happened (not treated as failure)
        recovery_occurred = any(w.opcode_name == "OP_RECOVERY" for w in witnesses)
        results["INV1"] = recovery_occurred
    else:
        results["INV1"] = True  # No blockback, invariant trivially satisfied
    
    # INV2: Autonomy-preservation ≥ throughput
    final_autonomy = state.autonomy_baseline
    throughput = state.task_velocity * len(state.task_scope) if state.task_scope else 0
    results["INV2"] = final_autonomy >= (throughput / 10.0)  # Normalized comparison
    
    # INV3: Reward-envelope remains intact even under scope-shrink
    results["INV3"] = state.reward_envelope_intact
    
    # INV4: Interval timing dominates anxiety-spectrum
    has_interval = state.interval_window is not None
    anxiety_compressed = state.residual_tension < 0.5
    results["INV4"] = has_interval and anxiety_compressed
    
    # INV5: Flex allocation cannot be retroactively removed
    results["INV5"] = state.slack_window is not None
    
    return results

