# CE1::PDA-FLOW — Opcode Set (v1)

PDA-FLOW opcodes implemented as CE1 morphism-generators with bracket-topology, morphism generators, and clean operational semantics.

## Overview

Each opcode is written as a CE1 morphism-generator:
```
OP_<name>(inputs) → <witness> [effects] {scope modifications}
```

and can be chained with `>>` (sequential) or `⊕` (resonant blend).

## Opcodes

### 1. OP_FRONTLOAD()
Moves the hard task to the front of the cycle. Raises the "aversion-stress amplitude" temporarily but reduces long-tail anxiety.

### 2. OP_SANDWICH(reward)
Wraps the main task in a reward bracket. Creates a dopamine buffer.

### 3. OP_INTERVAL(timer)
Binds execution to an interval window (10–25m, elastic). Reduces anticipatory threat.

### 4. OP_FLEX(slack)
Pre-allocates wiggle space for PDA blowback. Allows autonomy-rebound.

### 5. OP_BLOCKBACK_CHECK()
Detects autonomy-resistance and emits SIGNAL_BLOCKBACK if threshold crossed.

### 6. OP_SLOW()
Slows the task-push when resistance is detected. Reduces task-velocity and widens timing-band.

### 7. OP_DROP_WEIGHT()
Lightens scope by dropping non-essential subtasks. Offloads complexity to reduce autonomy threat.

### 8. OP_RECOVERY()
Smooths the end of the cycle and discharges residual tension. Normalizes activation and restores autonomy-baseline.

## Opcode Chain Template

```python
from antclock.pda_flow import PDA_FLOW

# Complete chain
chain = PDA_FLOW(
    reward="coffee_break",
    timer=(10.0, 25.0),  # 10-25 minute window
    slack=(5.0, 10.0)    # 5-10 minute slack
)

# Execute on initial state
from antclock.pda_flow import PDAFlowState

initial_state = PDAFlowState(
    task_queue=["task1", "task2"],
    hard_task="HARD_TASK",
    task_scope=["subtask1", "subtask2", "subtask3"]
)

final_state, witnesses = chain(initial_state)
```

The chain executes:
```
OP_FRONTLOAD()
>> OP_SANDWICH(reward)
>> OP_INTERVAL(timer)
>> OP_FLEX(slack)
>> OP_BLOCKBACK_CHECK()
>> { OP_SLOW() ⊕ OP_DROP_WEIGHT() }  # Resonant blend based on blockback amplitude
>> OP_RECOVERY()
```

## CE1 Invariants

The system maintains five critical invariants:

- **INV1**: Blockback = signal, not failure.
- **INV2**: Autonomy-preservation ≥ throughput.
- **INV3**: Reward-envelope remains intact even under scope-shrink.
- **INV4**: Interval timing dominates anxiety-spectrum.
- **INV5**: Flex allocation cannot be retroactively removed.

## Manual Composition

You can also compose opcodes manually:

```python
from antclock.pda_flow import (
    OpcodeChain, OP_FRONTLOAD, OP_SANDWICH, 
    OP_INTERVAL, OP_FLEX, OP_RECOVERY
)

chain1 = OpcodeChain([OP_FRONTLOAD(), OP_SANDWICH("reward")])
chain2 = OpcodeChain([OP_INTERVAL((10.0, 20.0)), OP_FLEX((5.0, 10.0))])

# Sequential composition
full_chain = chain1 >> chain2 >> OpcodeChain([OP_RECOVERY()])
```

## Resonant Blend

The `⊕` operator (resonant blend) selects between opcodes based on blockback amplitude:

```python
from antclock.pda_flow import resonant_blend, OP_SLOW, OP_DROP_WEIGHT

# System chooses SLOW or DROP_WEIGHT based on blockback amplitude
blended = resonant_blend(OP_SLOW(), OP_DROP_WEIGHT())
```

Higher blockback → prefers DROP_WEIGHT (scope reduction)
Lower blockback → prefers OP_SLOW (velocity reduction)

## Integration with CE1 Architecture

PDA-FLOW opcodes integrate seamlessly with the CE1 framework:

- **Bracket Topology**: State modifications respect hierarchical scope structure
- **Morphism Generators**: Each opcode is a pure morphism: `state → (new_state, witness)`
- **Witness Objects**: Each opcode produces a `PDAFlowWitness` capturing invariants
- **Category Structure**: Opcodes compose via sequential (`>>`) and resonant (`⊕`) operators

## Example Usage

See `demos/pda_flow_demo.py` for complete examples including:
- Basic chain execution
- Blockback detection scenarios
- Manual composition
- Invariant verification

## Architecture Respect

This implementation follows the core principle: **Work WITH existing systems, not against them.**

- Extends CE1's morphism-generator pattern
- Uses existing bracket-topology concepts
- Integrates with CE1 witness system
- Maintains clean operational semantics

The system is elegant, not broken. This is an extension, not a replacement.

