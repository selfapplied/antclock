# CE1::PDA-FLOW — Galois-Lifted Version

PDA-flow as a field extension with autonomy-index.

## Mathematical Structure

### Field Extension

**Base Field**: `F = Q(task_execution)` - the standard task execution field

**Extension**: `F(α)` where `α = autonomy-index`

Every element in the extended field can be written as:
```
state = a + b·α
```

where:
- `a ∈ F` = base execution component (throughput, task completion)
- `b ∈ F` = autonomy-index coefficient
- `α` = autonomy-index (the field extension element)

### Autonomy-Index

The autonomy-index `α` is defined as:
```
α = √(autonomy_baseline)  if autonomy_baseline ≥ 0
α = i√(|autonomy_baseline|)  if autonomy_baseline < 0 (complex extension)
```

Properties:
- `α² = autonomy_baseline` (or its absolute value for complex case)
- `α` is not in the base field (irreducible)
- Galois conjugates: `α` and `α̅` (autonomy and its inverse)

### Galois Group

The Galois group `Gal(F(α)/F)` has 2 elements:
- **id**: Identity automorphism (`id(α) = α`)
- **conj**: Conjugation automorphism (`conj(α) = α̅`)

This is isomorphic to `C₂` (cyclic group of order 2).

## Field-Theoretic Properties

### Field Norm

For an element `x = a + bα`:
```
N(x) = x · x̅ = (a + bα)(a + bα̅) = a² + b²N(α) + ab(α + α̅)
```

Since `Tr(α) = α + α̅ = 0`, this simplifies to:
```
N(x) = a² + b²N(α)
```

### Field Trace

For an element `x = a + bα`:
```
Tr(x) = x + x̅ = 2a
```

### Fixed Fields

- **Fixed under identity**: All of `F(α)`
- **Fixed under conjugation**: Only `F` (base field)

## Galois-Lifted Opcodes

Each opcode is now an automorphism of the field extension `F(α)`, preserving field structure while transforming the state.

### Opcode Behavior

- **OP_FRONTLOAD**: May conjugate if resistance is high (challenges autonomy)
- **OP_SANDWICH**: Preserves autonomy (identity)
- **OP_INTERVAL**: Preserves autonomy (identity)
- **OP_FLEX**: Preserves autonomy (identity)
- **OP_SLOW**: Preserves autonomy (identity)
- **OP_DROP_WEIGHT**: Preserves autonomy (identity)
- **OP_RECOVERY**: Restores autonomy (identity)

## Usage

### Basic Field Extension

```python
from antclock.pda_flow_galois import GaloisPDAFlowState
from antclock.pda_flow import PDAFlowState

# Create base state
base_state = PDAFlowState(
    task_queue=["task1", "task2"],
    hard_task="HARD_TASK",
    autonomy_baseline=0.8
)

# Lift to field extension
galois_state = GaloisPDAFlowState(base_state)

# Access field properties
print(f"Field element: {galois_state.field_element.evaluate()}")
print(f"Field norm: {galois_state.field_norm()}")
print(f"Autonomy-index: {galois_state.autonomy_index().value}")
```

### Galois Automorphisms

```python
from antclock.pda_flow_galois import GaloisGroup

galois_group = GaloisGroup()

# Apply identity (preserves α)
id_state = galois_state.apply_automorphism(galois_group.identity)

# Apply conjugation (swaps α and α̅)
conj_state = galois_state.apply_automorphism(galois_group.conjugation)

# Get orbit
orbit = galois_group.orbit(galois_state.field_element)
```

### Complete Galois-Lifted Chain

```python
from antclock.pda_flow_galois import Galois_PDA_FLOW

# Create chain
chain = Galois_PDA_FLOW(
    reward="coffee_break",
    timer=(15.0, 30.0),
    slack=(10.0, 15.0)
)

# Execute
final_state, witnesses = chain(galois_state)
```

### Field Extension Analysis

```python
from antclock.pda_flow_galois import analyze_field_extension, verify_galois_invariants

# Analyze field structure
analysis = analyze_field_extension(galois_state)
print(f"Galois group order: {analysis['galois_group']['order']}")
print(f"Isomorphic to: {analysis['galois_group']['isomorphic_to']}")

# Verify invariants
invariants = verify_galois_invariants(galois_state)
```

## Galois Invariants

The system maintains three Galois-theoretic invariants:

- **INV_GALOIS_1**: Field norm is preserved under automorphisms
  - `N(σ(x)) = N(x)` for all `σ ∈ Gal(F(α)/F)`

- **INV_GALOIS_2**: Base field elements are fixed under conjugation
  - If `b = 0` (pure base field), then `conj(x) = x`

- **INV_GALOIS_3**: Autonomy-index norm is preserved
  - `N(α) = N(α̅)`

## Key Insights

1. **Autonomy as Field Element**: Autonomy-preservation is now a first-class field element, not just a constraint. This allows algebraic manipulation of autonomy.

2. **Galois Symmetry**: The conjugation automorphism represents the symmetry between autonomy-preservation and its inverse, allowing the system to explore both directions.

3. **Field Structure Preservation**: Opcodes act as automorphisms, preserving the field structure while transforming states. This ensures mathematical consistency.

4. **Base Field Projection**: The field trace `Tr(x) = 2a` gives the "base field projection" - the throughput component independent of autonomy.

5. **Norm as Measure**: The field norm `N(x)` measures the "size" of the state in the extended field, combining both throughput and autonomy components.

## Integration with CE1 Architecture

The Galois-lifted version integrates seamlessly with CE1's existing Galois structure:

- **CE1.galois-cover**: Uses the same Galois group structure (depth shifts, mirror involution, curvature flips)
- **Field Extensions**: Extends CE1's field extension concepts to task execution
- **Automorphisms**: Opcodes as automorphisms align with CE1's morphism-generator pattern
- **Invariants**: Galois invariants complement CE1's bracket-topology invariants

## Example

See `demos/pda_flow_galois_demo.py` for complete examples including:
- Field extension structure analysis
- Galois automorphism demonstrations
- Galois-lifted opcode execution
- Complete Galois-lifted chain
- Field extension comparisons

## Architecture Respect

This implementation extends the CE1 framework's Galois structure:

- Uses existing Galois group concepts from `CE1.galois-cover`
- Extends field extension theory to task execution
- Maintains compatibility with base PDA-flow opcodes
- Preserves all CE1 invariants while adding Galois-theoretic ones

The system is elegant, not broken. This is an extension, not a replacement.

