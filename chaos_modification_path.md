# AntClock Chaos Modification Path

## Current State: Stratified Dynamics (Proto-Chaos)

AntClock currently exhibits **stratified behavior** rather than true chaos:

- **Lyapunov exponent λ ≈ 0**: No exponential divergence in state space
- **KS entropy h_KS > 0**: Information production from symbol transitions
- **Two-plateau curvature**: Oscillates between `log(2)` and `log(0.75)`
- **Linear state dynamics**: `x_{n+1} = x_n + 1` with boundary jumps

This creates **proto-chaos**: symbolic complexity without chaotic mixing.

## Goal: True Chaotic Dynamics

To achieve true chaos, we need:
- Lyapunov exponents > 0 (exponential divergence)
- Sensitive dependence on initial conditions
- KS entropy > 0 with state-space chaos
- Feedback loop between curvature and state evolution

## Approach D: Use Curvature as the Map Itself

### Core Modification

Replace the linear state update with a curvature-driven nonlinear map:

```python
# CURRENT (stratified):
x_{n+1} = x_n + 1  # Linear progression

# PROPOSED (chaotic):
# 1. Compute curvature κ(x_n) for current state
# 2. Use exp(κ(x_n)) as logistic map parameter
# 3. Normalize x to [0,1] within digit shell
# 4. Apply logistic map
# 5. Rescale back to digit shell range

def curvature_driven_step(self):
    """True chaotic dynamics: curvature drives state evolution"""

    # Get current curvature (stratified observable)
    curvature = digit_boundary_curvature(self.x)

    # Convert log curvature to logistic parameter r ∈ [0, 4]
    # curvature ranges: [-0.2877, 0.6931] → r ranges: [0.75, 2.0]
    r_parameter = 1.0 + curvature  # Maps to [0.712, 1.693]

    # For chaos, we want r ∈ [3.5, 4.0] (chaotic regime)
    # So scale and shift: r_chaos = 3.5 + (r_parameter - 0.712) * (4.0 - 3.5)/(1.693 - 0.712)
    r_chaos = 3.5 + (r_parameter - 0.712) * 0.5 / 0.981

    # Normalize x within its digit shell [10^{d-1}, 10^d - 1]
    d = digit_count(self.x)
    shell_min = 10**(d-1) if d > 0 else 0
    shell_max = 10**d - 1
    x_normalized = (self.x - shell_min) / (shell_max - shell_min)

    # Apply logistic map with curvature-determined parameter
    x_next_normalized = r_chaos * x_normalized * (1 - x_normalized)

    # Rescale back to digit shell
    self.x = int(shell_min + x_next_normalized * (shell_max - shell_min))

    # Handle digit boundary crossings (renormalization)
    new_d = digit_count(self.x)
    if new_d != d:
        # Digit shell transition - could trigger parameter changes
        pass
```

### Key Changes

1. **Feedback Loop**: Curvature now influences state evolution
2. **Nonlinear Dynamics**: Logistic map replaces linear increment
3. **Parameter Coupling**: `r = f(curvature)` creates state-dependent chaos
4. **Shell Normalization**: Proper scaling within digit shells

### Expected Outcomes

- **Lyapunov λ > 0**: Exponential divergence from logistic chaos
- **KS entropy h_KS > 0**: Both state and symbolic information production
- **Bifurcations**: Period-doubling cascade as χ_FEG varies
- **Strange Attractors**: Possible fractal structures

### Implementation Steps

1. **Add curvature_driven_step()** method to CurvatureClockWalker
2. **Modify step()** to optionally use chaotic dynamics
3. **Add chaos_mode parameter** to constructor
4. **Update chaos analysis** to detect true chaos vs proto-chaos
5. **Tune parameter mapping** for optimal chaotic regime

### Mathematical Foundation

The modification creates a **coupled system**:

```
κ(x) = curvature function (piecewise constant)
r(x) = logistic parameter derived from κ(x)
x_{n+1} = r(x_n) * x_n * (1 - x_n)  [normalized]
```

This establishes:
- **State → Curvature**: x determines which curvature plateau
- **Curvature → State**: κ determines chaos parameter r
- **Feedback**: Chaotic dynamics emerge from the coupling

### Validation

After implementation:
- Lyapunov exponents should show positive values
- KS entropy should increase significantly
- Bifurcation diagrams should show period-doubling cascades
- Chaos analysis should classify as "strong_chaos" regime

This transforms AntClock from stratified proto-chaos to genuine chaotic dynamics while preserving its fundamental curvature-based architecture.
