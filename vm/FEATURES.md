# Planned Features Implementation Guide

This document describes the five implemented features of the Zero-image μVM and how to use them.

## Overview

All five planned features from the README have been implemented with a minimal, composable approach. Each feature provides functional hooks and extension points that can be enhanced in the future.

## 1. JIT Compilation for Hot Paths

### Description
Tracks opcode execution counts and triggers JIT compilation when a hot path is detected (after 1000 executions).

### Usage

```c
VMState *vm = vm_create();

// Enable JIT compilation
vm->flags |= VM_FLAG_JIT;

// Execute opcodes - counters are tracked automatically
for (int i = 0; i < 2000; i++) {
    vm_execute_opcode(vm, OP_PROJECT);
}
// After 1000 executions, JIT hook is triggered

vm_destroy(vm);
```

### Extension Points
- `vm_jit_compile_hot_path()` - Stub for JIT compiler integration
- `vm->opcode_counts[]` - Per-opcode execution counters
- `JIT_HOT_PATH_THRESHOLD` - Configurable threshold constant

### Future Enhancement
Link with LLVM, DynASM, or similar JIT compiler library to generate native code for hot loops.

---

## 2. Network-Transparent Execution

### Description
Enables distributed computation by serializing and deserializing VM state for network transmission.

### Usage

```c
VMState *vm = vm_create();

// Configure remote endpoint
vm->flags |= VM_FLAG_REMOTE;
vm_remote_exec(vm, "tcp://worker.example.com:5555");

// Execute some computation
vm_push(vm, 0.5 + 14.134725*I, 3, 1.57);
vm_execute_opcode(vm, OP_PROJECT);

// Serialize state for transmission
vm_serialize_state(vm, "state.bin");

// On remote machine: deserialize and continue
VMState *remote_vm = vm_create();
vm_deserialize_state(remote_vm, "state.bin");

vm_destroy(vm);
vm_destroy(remote_vm);
```

### Extension Points
- `vm_serialize_state()` - Binary state serialization
- `vm_deserialize_state()` - State restoration
- `vm_remote_exec()` - Remote execution stub
- `vm->remote_endpoint` - Endpoint configuration

### Future Enhancement
Implement network protocol (gRPC, WebSocket, ZeroMQ) for actual remote execution.

---

## 3. Persistent State Checkpointing

### Description
Fully functional checkpoint/restore system for saving and loading complete VM state.

### Usage

```c
VMState *vm = vm_create();

// Run some computation
vm_push(vm, 0.5 + 14.134725*I, 3, 1.57);
vm->antclock = 12345;
vm->pc = 42;

// Save checkpoint
vm_checkpoint_save(vm, "checkpoint.ckpt");

// Later: restore from checkpoint
VMState *restored = vm_create();
vm_checkpoint_load(restored, "checkpoint.ckpt");

// State is fully restored
assert(restored->antclock == 12345);
assert(restored->pc == 42);

vm_destroy(vm);
vm_destroy(restored);
```

### Features
- Magic header validation (0x434B5054 - "CKPT")
- Version checking
- Complete state preservation (stack, program, counters)
- Binary format for efficiency

### File Format

```
[32 bits] Magic: 0x434B5054 ("CKPT")
[32 bits] Version: 1
[8 bits]  Stack pointer
[64 bits] Antclock
[32 bits] Program counter
[32 bits] Flags
[64 bits] k_spec
[16 KB]   Stack frames
[32 bits] Program size
[...]     Program data
```

---

## 4. Interactive Debugger

### Description
Breakpoint management and step-through execution for debugging VM programs.

### Usage

```c
VMState *vm = vm_create();

// Enable debug mode
vm->flags |= VM_FLAG_DEBUG;

// Add breakpoints
vm_debug_add_breakpoint(vm, 5);   // Break at PC=5
vm_debug_add_breakpoint(vm, 10);  // Break at PC=10

// Load program
vm_load_program(vm, "program.zero");

// Step through with breakpoint checking
while (vm->pc < vm->program_size) {
    vm_debug_step(vm);  // Pauses at breakpoints
}

// Remove breakpoint
vm_debug_remove_breakpoint(vm, 5);

vm_destroy(vm);
```

### Features
- Up to 16 concurrent breakpoints
- Automatic pause at breakpoints (waits for Enter key)
- State display at each breakpoint
- Breakpoint management (add/remove/check)

### API Functions
- `vm_debug_add_breakpoint(vm, pc)` - Add breakpoint
- `vm_debug_remove_breakpoint(vm, pc)` - Remove breakpoint
- `vm_debug_is_breakpoint(vm, pc)` - Check if breakpoint exists
- `vm_debug_step(vm)` - Step with breakpoint checking

---

## 5. Visual Tracer for Spectral Trajectories

### Description
Records spectral frame history and exports to CSV/JSON for visualization.

### Usage

```c
VMState *vm = vm_create();

// Enable tracing
vm_trace_enable(vm, true);

// Execute operations - frames are recorded automatically
for (int i = 0; i < 100; i++) {
    vm->antclock = 1000 + i;
    double imag = 14.134725 + i * 3.5;
    vm_push(vm, 0.5 + imag*I, i % 10, 1.57);
    vm_execute_opcode(vm, OP_PROJECT);
}

// Export trace data
vm_trace_export(vm, "trace.csv", "csv");   // CSV for gnuplot
vm_trace_export(vm, "trace.json", "json"); // JSON for web viz

// Disable tracing
vm_trace_enable(vm, false);

vm_destroy(vm);
```

### Visualization

```bash
# Using the included Python script
python3 demos/trace_visualization.py trace.csv

# Using gnuplot
gnuplot> plot "trace.csv" using 2:3 with lines title "Trajectory"

# Using matplotlib (Python)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trace.csv')
plt.plot(df['rho_real'], df['rho_imag'])
plt.xlabel('Re(ρ)')
plt.ylabel('Im(ρ)')
plt.title('Spectral Trajectory')
plt.show()
```

### CSV Format
```
index,rho_real,rho_imag,depth,monodromy,timestamp
0,0.5000000000,14.1347250000,0,1.570000,1000
1,0.5000000000,17.6347250000,1,1.670000,1001
...
```

### JSON Format
```json
{
  "traces": [
    {
      "index": 0,
      "rho": {"real": 0.5, "imag": 14.134725},
      "depth": 0,
      "monodromy": 1.57,
      "timestamp": 1000
    },
    ...
  ]
}
```

---

## Integration Example

All features work together:

```c
VMState *vm = vm_create();

// Enable all features
vm->flags = VM_FLAG_JIT | VM_FLAG_DEBUG | VM_FLAG_TRACE | VM_FLAG_REMOTE;
vm_trace_enable(vm, true);
vm_debug_add_breakpoint(vm, 10);
vm_remote_exec(vm, "tcp://localhost:8080");

// Run computation
vm_load_program(vm, "program.zero");
vm_run(vm);

// Save checkpoint with all features active
vm_checkpoint_save(vm, "full_state.ckpt");

// Export trace for visualization
vm_trace_export(vm, "results.csv", "csv");

vm_destroy(vm);
```

---

## Testing

Comprehensive test suite available in `tests/test_planned_features.c`:

```bash
# Build and run tests
cd vm
make clean && make
./test_planned_features

# Expected output:
# Tests run: 6
# Tests passed: 6
# Tests failed: 0
```

---

## Performance Considerations

- **JIT Tracking**: Adds 1 integer increment per opcode execution (negligible overhead)
- **Tracing**: Adds frame copy when enabled (~64 bytes per trace)
- **Checkpointing**: I/O bound, ~16 KB stack + program size
- **Debugging**: Only active when VM_FLAG_DEBUG is set
- **Network**: Serialization is ~16 KB + program size

---

## Future Enhancements

### JIT Compilation
- Integrate LLVM or DynASM backend
- Generate native x86-64 code for hot loops
- Add compilation cache and invalidation

### Network Execution
- Implement gRPC or ZeroMQ protocol
- Add authentication and encryption
- Support work stealing and load balancing

### Debugger
- Add memory watchpoints
- Implement conditional breakpoints
- Add reverse execution support

### Tracer
- Add real-time WebSocket streaming
- Implement 3D visualization of complex trajectories
- Add statistical analysis tools

---

## Architecture Notes

The implementation follows a "stigmergic" design philosophy:
- **Minimal footprint**: ~500 LOC added to ~500 LOC base
- **Composable hooks**: Each feature is independent but integrates cleanly
- **Clear upgrade paths**: Stubs and extension points are well-documented
- **Zero breaking changes**: All existing functionality preserved

This approach creates "pheromone trails" that guide future development without committing to specific implementations prematurely.
