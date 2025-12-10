# Zero-image μVM

A minimal, executable kernel designed to operationalize CE1 bracket algebra and tensor fields into a concrete, shippable artifact.

## Overview

The Zero-image μVM is the **"minimum shippable kernel"** of the AntClock CE system - a pragmatic realization that makes quantum aspects a "feature flag" rather than a prerequisite. It allows anyone to "run meaning" on a phone or standard computer immediately.

### Design Philosophy

- **Zero-Image**: Flat binary, read-only memory-mapped format - no complex OS images or boot processes
- **Minimal**: ~400 lines of C, ~18 kB binary size
- **Portable**: Docker and WASM compatible, zero dependencies
- **Cache-Friendly**: 16 KB memory footprint (L1 cache size)
- **Future-Ready**: Designed to accept `--quantum` flag without core rewrite

## Architecture

### File Format

The VM executes a flat binary format:
- **Header**: 32 bytes (magic, version, metadata)
- **Records**: 24-byte spectral data points

Each record contains:
- **ρ (Complex)**: 2×64-bit floats (Real and Imaginary parts)
- **Ultrametric Depth (d)**: 32-bit uint representing `-log_p |Im ρ|_p`
- **Monodromy Angle (m)**: 32-bit float (phase/angle in radians)

### Memory Model

Instead of general-purpose registers, the VM uses a **circular stack** of spectral frames:
- **Frame Size**: 64 bytes each
- **Stack Depth**: 256 frames (16 KB total)
- **L1 Cache Friendly**: Entire stack fits in modern CPU L1 cache

### Instruction Set (ISA)

Four opcodes corresponding to CE1 bracket algebra:

#### `{}` (0x10) - Project
Projects tensor path onto λ-isotypic component.
- **Pops**: Tensor path index
- **Pushes**: Spectral value ρ from program

#### `[]` (0x20) - Depth
Computes p-adic distance (memory operator).
- **Pops**: Two values ρ₁, ρ₂
- **Pushes**: P-adic distance |ρ₁ - ρ₂|_p

#### `()` (0x30) - Morph
Applies Hecke action (type-system flow).
- **Pops**: Value ρ and morphism φ
- **Pushes**: Result of T_φ(ρ)

#### `<>` (0x40) - Witness
Extracts full witness tuple.
- **Pops**: Value ρ
- **Pushes**: (Real, Imaginary, Depth, Monodromy) components

## The Guardian

The Guardian logic decides whether to **compose** (merge) or **protect** (keep separate) two states:

```c
GuardianDecision guardian_decide(double depth, double k_spec) {
    return (depth < k_spec) ? GUARDIAN_PROTECT : GUARDIAN_COMPOSE;
}
```

- **Efficiency**: ~14 x86 instructions, branch-predictor friendly
- **Logic**: Compares depth against spectral constant k_spec
- **Decision**: Below threshold → protect, above → compose

## Antclock Integration

Timing is handled without expensive system calls using a pre-computed lookup table:

- **Gamma Gap Table**: Spacing between first 100k zeta zeros
- **Tick Generation**: Hash of current imaginary component mod table size
- **Zero Cost**: Deterministic, no syscalls, pure computation

## Building

```bash
# Build the VM
make

# Run tests
make test

# Build examples
make examples

# View statistics
make stats

# Clean
make clean
```

### Docker

The easiest way to run the VM in Docker is using the provided Docker runner script:

```bash
# From repository root
./docker_run.sh build              # Build Docker image
./docker_run.sh run --help         # Run VM with help
./docker_run.sh run /programs/example.zero  # Run a program
./docker_run.sh shell              # Open interactive shell
./docker_run.sh clean              # Remove Docker image
```

Alternatively, you can use make directly:

```bash
# Build Docker image
make docker

# Run in container manually
docker run -v $(pwd)/programs:/programs zero-vm:latest /programs/example.zero
```

See the main repository README for more details on the Docker runner.

### WASM (Future)

```bash
# Requires emscripten installed
make wasm
```

## Usage

### Creating a Program

A Zero-image program is a binary file with this structure:

```
[32-byte header]
[24-byte record 0]
[24-byte record 1]
...
[24-byte record N]
```

See `examples/` directory for sample programs.

### Running a Program

```bash
./zero_vm program.zero
```

### VM State

The VM maintains:
- **Stack**: 256 spectral frames (circular)
- **SP**: Stack pointer (0-255)
- **PC**: Program counter
- **Antclock**: Current tick counter
- **k_spec**: Guardian threshold

## Examples

### Example 1: Simple Projection

```c
// Push index
vm_push(vm, 0.0 + 0.0*I, 0, 0.0);

// Project to get spectral value
vm_execute_opcode(vm, OP_PROJECT);

// Witness the result
vm_execute_opcode(vm, OP_WITNESS);
```

### Example 2: Computing Distance

```c
// Push two spectral values
vm_push(vm, 1.0 + 2.0*I, 5, 0.5);
vm_push(vm, 3.0 + 4.0*I, 3, 1.0);

// Compute p-adic distance
vm_execute_opcode(vm, OP_DEPTH);
```

### Example 3: Morphism Application

```c
// Push base value and morphism
vm_push(vm, 2.0 + 1.0*I, 4, 0.0);
vm_push(vm, 0.5 + 0.5*I, 2, M_PI/4);

// Apply Hecke action
vm_execute_opcode(vm, OP_MORPH);
```

## Integration with Python

Python bindings can be created using ctypes or cffi:

```python
from ctypes import *

# Load VM library
vm_lib = CDLL('./zero_vm.so')

# Create VM
vm = vm_lib.vm_create()

# Execute opcode
vm_lib.vm_execute_opcode(vm, 0x10)  # OP_PROJECT

# Cleanup
vm_lib.vm_destroy(vm)
```

## Future Extensions

### Quantum Mode

In future versions, the VM can accept a `--quantum` flag:

```bash
./zero_vm --quantum program.zero
```

In quantum mode:
- Spectral frames interpreted as qubits: |ρ⟩ = α|0⟩ + β|1⟩
- P-adic disjointness logic becomes error-sector detection
- Same opcodes, quantum backend

### Planned Features

- [ ] JIT compilation for hot paths
- [ ] Network-transparent execution (distribute computation)
- [ ] Persistent state checkpointing
- [ ] Interactive debugger
- [ ] Visual tracer for spectral trajectories

## Technical Details

### Memory Layout

```
VMState (total: 16 KB + overhead)
├── Stack: 256 × 64 bytes = 16384 bytes
├── SP: 1 byte
├── Antclock: 8 bytes
├── PC: 4 bytes
├── Flags: 4 bytes
└── Guardian params: 8 bytes
```

### Performance

- **Frame Access**: O(1) - direct indexing
- **Stack Operations**: O(1) - circular buffer
- **Guardian Decision**: ~14 instructions, ~5 CPU cycles
- **Antclock Tick**: ~10 instructions, hash lookup

### Compatibility

- **C Standard**: C11
- **Platform**: Linux, macOS, Windows (with MinGW)
- **Architecture**: x86-64, ARM64, WASM
- **Dependencies**: Standard C library only

## Analogy: The Acoustic Guitar

Think of the Zero-image μVM like an **acoustic guitar** version of a song that will eventually be played by a full orchestra:

- The **Sheet Music** (Tensor Field) is the same
- The **Notes** (Opcodes) are the same
- The acoustic guitar (the μVM) proves the *melody* works right now
- It's portable and can be played anywhere (Docker container)
- Later, plug into an amplifier and bring in the orchestra (Quantum Hardware)
- But you don't need the orchestra just to verify the song is beautiful

## License

Part of the AntClock project, licensed under CC BY-SA 4.0.

## Contributing

See the main AntClock repository for contribution guidelines.

## References

- CE1 Bracket Algebra: See `docs/spec.md` in main repository
- Riemann Zeta Function: Classical analytic number theory
- P-adic Numbers: Koblitz, "p-adic Numbers, p-adic Analysis, and Zeta-Functions"
- Hecke Operators: Diamond & Shurman, "A First Course in Modular Forms"
