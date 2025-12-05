# Antclock Runtime Interface

This document describes the runtime wrapper and CE-layer decorators that enable external projects to become "antclock aware."

## Overview

The `AntRuntime` class wraps the `CurvatureClockWalker` and maintains experiential time state:

- **x**: Current walker position on the curvature manifold
- **phase**: Accumulated phase from curvature-driven dynamics
- **A**: Discrete experiential index (universal clock)
- **log**: The []_a memory log, storing event entries keyed by A

Each call to `tick()` advances the walker one step via `evolve_step(x, dt)`, updates the phase, increments A based on event type and CE layer, and appends a structured entry to the log.

## CE-Layer Decorators

Three decorator factories correspond to the CE layers:

- **@antce1()**: CE1 structural events (`event_type="bracket"`, layer=1). Default logs function return value as state.
- **@antce2()**: CE2 flow events (`event_type="flow_step"`, layer=2). Default logs function return value as state.
- **@antce3()**: CE3 emergent events (`event_type="simplex_flip"`, layer=3). Default logs function arguments as state.

Each decorator supports optional configuration:

- `log_state`: One of `"return"`, `"args"`, or `"none"` to control what gets logged
- `attach_A`: If `True`, injects the current `runtime.A` as a keyword argument
- `metadata_fn`: Optional function `(result, args, kwargs) -> dict` for custom metadata

## Usage Example

```python
from antclock.runtime import AntRuntime, make_ce_decorators

# Create runtime and decorators
ant_runtime = AntRuntime()
antce1, antce2, antce3 = make_ce_decorators(ant_runtime)

class Agent:
    def __init__(self):
        self.self_model = {}
    
    @antce1()
    def update_self_model(self, observation):
        """CE1: Structural update to agent's self-representation."""
        self.self_model['last_obs'] = observation
        return self.self_model
    
    @antce2(attach_A=True)
    def take_action(self, action, A=None):
        """CE2: Flow step in environment. A is injected automatically."""
        return f"Executed {action} at experiential time A={A}"
    
    @antce3()
    def evolve_grammar(self, pattern):
        """CE3: Emergent grammar evolution from compositional discrepancy."""
        return {"new_rule": pattern, "evolved": True}

# Each decorated call advances the antclock and logs an event
agent = Agent()
agent.update_self_model("sensory input")
agent.take_action("move forward")
agent.evolve_grammar("noun -> noun phrase")

# Query the experiential history
print(f"Total experiential time: A={ant_runtime.A}")
print(f"Events logged: {len(ant_runtime.log)}")
```

## API Reference

### AntRuntime

```python
class AntRuntime:
    """
    Runtime wrapper around CurvatureClockWalker that tracks experiential time A
    and maintains a []_a log of events.
    """
    
    def __init__(
        self,
        walker: Optional[CurvatureClockWalker] = None,
        x_0: float = 1.0,
        chi_feg: float = 0.638,
        enable_volte: bool = False
    ) -> None:
        ...
    
    def tick(
        self,
        dt: float = 0.01,
        event_type: str = "bracket",
        layer: int = 1,
        state: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Advance the antclock by one tick.
        Returns the log entry dict that was appended to self.log.
        """
        ...
```

### make_ce_decorators

```python
def make_ce_decorators(
    runtime: AntRuntime
) -> Tuple[
    Callable[..., Callable[[F], F]],
    Callable[..., Callable[[F], F]],
    Callable[..., Callable[[F], F]]
]:
    """
    Create CE-layer decorator factories bound to a runtime.
    
    Returns:
        A tuple of (antce1, antce2, antce3) decorator factories
    """
    ...
```

## Log Entry Structure

Each log entry contains:

```python
{
    'A': int,           # Experiential time index
    'x': float,         # Walker position
    'phase': float,     # Accumulated phase
    'digit_shell': int, # Current digit shell
    'clock_rate': float,# Clock rate at position
    'event_type': str,  # Event type (bracket, flow_step, simplex_flip)
    'layer': int,       # CE layer (1, 2, or 3)
    'state': Any,       # Optional state snapshot
    # ... additional metadata fields
}
```
