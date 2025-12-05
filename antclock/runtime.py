"""
AntClock Runtime Interface

Provides a thin runtime wrapper around CurvatureClockWalker for experiential time tracking,
along with CE-layer decorators for marking state transitions as CE1/CE2/CE3 events.
"""

import functools
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar

from .clock import CurvatureClockWalker, create_walker


# Type variables for decorator typing
F = TypeVar('F', bound=Callable[..., Any])


class AntRuntime:
    """
    Runtime wrapper around CurvatureClockWalker that tracks experiential time A
    and maintains a []_a log of events.

    Attributes:
        walker: The underlying CurvatureClockWalker instance
        x: Current walker position
        phase: Accumulated phase
        A: Discrete experiential index / universal clock
        log: List of []_a-style event entries
    """

    def __init__(
        self,
        walker: Optional[CurvatureClockWalker] = None,
        x_0: float = 1.0,
        chi_feg: float = 0.638,
        enable_volte: bool = False
    ) -> None:
        """
        Initialize the AntRuntime.

        Args:
            walker: Optional pre-configured CurvatureClockWalker. If None, creates one.
            x_0: Starting position for walker (used if walker is None)
            chi_feg: FEG coupling constant (used if walker is None)
            enable_volte: Whether to enable Volte corrections (used if walker is None)
        """
        if walker is not None:
            self.walker = walker
        else:
            self.walker = create_walker(x_0=x_0, chi_feg=chi_feg, enable_volte=enable_volte)

        self.x: float = self.walker.x_0
        self.phase: float = 0.0
        self.A: int = 0
        self.log: List[Dict[str, Any]] = []

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

        Evolves the walker one step, updates phase, increments A based on event type
        and layer, and logs the event.

        Args:
            dt: Time step for walker evolution
            event_type: Type of event ('bracket', 'flow_step', 'simplex_flip', etc.)
            layer: CE layer (1, 2, or 3)
            state: Optional state snapshot to log
            metadata: Optional extra metadata fields to merge into log entry

        Returns:
            The log entry dict that was appended to self.log
        """
        # Evolve the walker one step
        new_x, phase_increment = self.walker.evolve_step(self.x, dt)

        # Update phase
        self.phase += phase_increment

        # Increment A based on event type and layer
        a_increment = self.walker.universal_clock_increment(event_type, layer)
        self.A += a_increment

        # Compute current digit shell and clock rate
        digit_shell = len(str(int(self.x))) if self.x > 0 else 1
        clock_rate = self.walker.clock_rate(self.x)

        # Build log entry
        entry: Dict[str, Any] = {
            'A': self.A,
            'x': self.x,
            'phase': self.phase,
            'digit_shell': digit_shell,
            'clock_rate': clock_rate,
            'event_type': event_type,
            'layer': layer,
        }

        if state is not None:
            entry['state'] = state

        if metadata is not None:
            entry.update(metadata)

        # Update position after logging (log captures state before evolution)
        self.x = new_x

        # Append to log
        self.log.append(entry)

        return entry


LogStateOption = Literal["return", "args", "none"]


def _make_layer_decorator(
    runtime: AntRuntime,
    event_type: str,
    layer: int,
    default_log_state: LogStateOption
) -> Callable[..., Callable[[F], F]]:
    """
    Internal helper to construct a CE-layer decorator factory.

    Args:
        runtime: The AntRuntime instance to tick
        event_type: Event type string for this CE layer
        layer: CE layer number (1, 2, or 3)
        default_log_state: Default log_state behavior for this layer

    Returns:
        A decorator factory that takes optional kwargs and returns a decorator
    """

    def decorator_factory(
        log_state: Optional[LogStateOption] = None,
        attach_A: bool = False,
        metadata_fn: Optional[Callable[[Any, tuple, dict], Dict[str, Any]]] = None
    ) -> Callable[[F], F]:
        """
        Decorator factory with optional configuration.

        Args:
            log_state: One of "return", "args", "none", or None (use default)
            attach_A: If True, inject current runtime.A as kwarg 'A' if not already passed
            metadata_fn: Optional function(result, args, kwargs) -> dict of extra metadata

        Returns:
            A decorator that wraps functions to tick the antclock
        """
        effective_log_state = log_state if log_state is not None else default_log_state

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Optionally inject A into kwargs if attach_A is True
                if attach_A and 'A' not in kwargs:
                    kwargs['A'] = runtime.A

                # Call the wrapped function
                result = func(*args, **kwargs)

                # Compute state based on log_state setting
                state: Optional[Any] = None
                if effective_log_state == "return":
                    state = result
                elif effective_log_state == "args":
                    state = (args, kwargs)
                # "none" leaves state as None

                # Optionally compute extra metadata
                extra_metadata: Optional[Dict[str, Any]] = None
                if metadata_fn is not None:
                    try:
                        extra_metadata = metadata_fn(result, args, kwargs)
                    except Exception:
                        # Fail closed: if metadata_fn raises, don't add metadata
                        extra_metadata = None

                # Tick the runtime
                runtime.tick(
                    event_type=event_type,
                    layer=layer,
                    state=state,
                    metadata=extra_metadata
                )

                # Return the original function result unchanged
                return result

            return wrapper  # type: ignore[return-value]

        return decorator

    return decorator_factory


def make_ce_decorators(
    runtime: AntRuntime
) -> Tuple[
    Callable[..., Callable[[F], F]],
    Callable[..., Callable[[F], F]],
    Callable[..., Callable[[F], F]]
]:
    """
    Create CE-layer decorator factories bound to a runtime.

    Args:
        runtime: The AntRuntime instance to bind decorators to

    Returns:
        A tuple of (antce1, antce2, antce3) decorator factories

    Example:
        >>> ant_runtime = AntRuntime()
        >>> antce1, antce2, antce3 = make_ce_decorators(ant_runtime)
        >>>
        >>> @antce1()
        ... def update_self_model():
        ...     return {"state": "updated"}
        >>>
        >>> @antce2(attach_A=True)
        ... def take_action(A=None):
        ...     return f"action at A={A}"
        >>>
        >>> @antce3()
        ... def evolve_grammar(pattern):
        ...     return {"grammar": pattern}
    """
    antce1 = _make_layer_decorator(
        runtime=runtime,
        event_type="bracket",
        layer=1,
        default_log_state="return"
    )

    antce2 = _make_layer_decorator(
        runtime=runtime,
        event_type="flow_step",
        layer=2,
        default_log_state="return"
    )

    antce3 = _make_layer_decorator(
        runtime=runtime,
        event_type="simplex_flip",
        layer=3,
        default_log_state="args"
    )

    return antce1, antce2, antce3


if __name__ == "__main__":
    # Basic usage demonstration
    print("AntClock Runtime Demo")
    print("=" * 40)

    # Create runtime and decorators
    ant_runtime = AntRuntime()
    antce1, antce2, antce3 = make_ce_decorators(ant_runtime)

    # Example CE1 function (structural event)
    @antce1()
    def update_self_model(data: str) -> Dict[str, Any]:
        """CE1: Update internal model state."""
        return {"model": data, "updated": True}

    # Example CE2 function with attach_A
    @antce2(attach_A=True)
    def take_action(action: str, A: Optional[int] = None) -> str:
        """CE2: Take an action in the environment."""
        return f"Executed '{action}' at A={A}"

    # Example CE3 function (emergence event)
    @antce3()
    def evolve_grammar(pattern: str) -> Dict[str, Any]:
        """CE3: Evolve the compositional grammar."""
        return {"new_rule": pattern}

    # Run some operations
    print("\nExecuting decorated functions...")
    result1 = update_self_model("initial state")
    print(f"CE1 result: {result1}")

    result2 = take_action("move forward")
    print(f"CE2 result: {result2}")

    result3 = evolve_grammar("noun -> noun phrase")
    print(f"CE3 result: {result3}")

    # Show runtime state
    print(f"\nRuntime state:")
    print(f"  A (experiential time): {ant_runtime.A}")
    print(f"  x (position): {ant_runtime.x:.6f}")
    print(f"  phase: {ant_runtime.phase:.6f}")
    print(f"  log entries: {len(ant_runtime.log)}")

    print("\nLog entries:")
    for i, entry in enumerate(ant_runtime.log):
        print(f"  [{i}] A={entry['A']}, layer={entry['layer']}, type={entry['event_type']}")

    print("\n✨ Runtime demo complete!")
