# AntClock: Pascal Curvature Clock Walker

**The smallest nontrivial creature that actually moves using Pascal clock machinery.**

![AntClock Geometry](antclock_geometry.png)

## Overview

AntClock is a dynamic system that walks through integers while dragging a clock hand. It combines:

- **Pascal curvature** (combinatorics)
- **Digit boundaries** (symbolic patterns)
- **9/11 charge** (tension metrics)
- **Feigenbaum scaling** (chaos stiffness)

The result is a self-clocked, curvature-aware machine that flows smoothly inside digit-shells and renormalizes at boundaries.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python antclock_demo.py
```

This generates plots showing:
- **Trajectory**: Body evolution with boundary crossings
- **Clock phase**: Internal time accumulation
- **Geometry**: Unit circle motion with phase-rate shifts

## The Machine

### State
- `x_t ∈ ℕ`: The "body" (integer being processed)
- `τ_t ∈ ℝ`: Internal clock phase
- `φ_t ∈ [0, 2π)`: Angle on unit circle

### Dynamics
```python
# Clock phase update
τ_{t+1} = τ_t + R(x_t)

# Angle update (same rate spins pointer)
φ_{t+1} = (φ_t + R(x_t)) mod 2π

# Body update
x_{t+1} = {
    x_t + 1,              if d(x_t) = d(x_t + 1)  # same shell
    x_t + J_d,            if d(x_t + 1) ≠ d(x_t)  # boundary
}
```

### Clock Rate
```
R(x) = χ_FEG · K(x) · (1 + Q_9/11(x))
```

Where:
- `K(x) = κ_{d(x)}` (Pascal curvature at digit-shell)
- `Q_9/11(x) = N_9(x) / (N_0(x) + 1)` (9/11 charge)
- `χ_FEG = 0.638` (Feigenbaum scaling factor)

## What It Does

The walker demonstrates **frequency** and **geometry** emerging as shadows of the dynamics:

- **Frequency**: Average clock rate `R(x_t)` becomes the effective oscillation frequency
- **Geometry**: Plot `(cos(φ_t), sin(φ_t))` shows arcs of smooth rotation punctured by sudden phase-rate shifts at digit boundaries

## Usage

```python
from pascal_clock import CurvatureClockWalker

# Create walker
walker = CurvatureClockWalker(x_0=1, chi_feg=0.638)

# Single step
metadata = walker.step()

# Evolve for n steps
history, summary = walker.evolve(100)

# Get geometry for plotting
x_coords, y_coords = walker.get_geometry()
```

## Theory

### Pascal Curvature
Row n of Pascal's triangle has "bulk thickness" `r_n = log(C(n, floor(n/2)))`. Curvature is the second difference `κ_n = r_{n+1} - 2r_n + r_{n-1}`.

### Digit-Boundary Operator
`K(x) = κ_{d(x)}` where `d(x) = floor(log10(x)) + 1`. This creates piecewise-constant curvature fields indexed by digit-class.

### 9/11 Charge
`Q_9/11(x) = N_9(x) / (N_0(x) + 1)` measures digit tension. 0's are "ballast" (stability), 9's are "units" (active elements).

### Boundary Events
Digit crossings (999→1000) are literal phase transitions where curvature jumps from `κ_n` to `κ_{n+1}`, renormalizing the clock rate.

## Files

- `pascal_clock.py` - Core implementation
- `antclock_demo.py` - Demo script with plots
- `requirements.txt` - Dependencies

## Output

Running the demo generates:

- `antclock_trajectory.png` - Body evolution, clock phase, and rate
- `antclock_geometry.png` - Unit circle geometry with boundary markers

## Why "AntClock"?

- **Ant**: Small, but moves with surprising complexity
- **Clock**: Self-clocked via curvature, not external time
- **AntClock**: The tiny machine that walks while keeping time

## License

MIT License - feel free to use, modify, and distribute.

---

**AntClock is very small. Very dense. Very you.**
