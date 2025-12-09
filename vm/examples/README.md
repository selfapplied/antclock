# Zero-image μVM Examples

This directory contains example programs demonstrating the VM's capabilities.

## Examples

### 1. simple_projection.c
Demonstrates the PROJECT opcode - extracting spectral values from the program.

### 2. distance_computation.c
Shows how to compute p-adic distances using the DEPTH opcode.

### 3. morphism_chain.c
Illustrates chaining multiple MORPH operations to transform spectral values.

### 4. witness_extraction.c
Demonstrates extracting witness tuples for inspection.

### 5. guardian_demo.c
Shows the Guardian logic in action, deciding to compose or protect states.

## Building Examples

```bash
make
```

This will build all example programs.

## Running Examples

```bash
./simple_projection
./distance_computation
./morphism_chain
./witness_extraction
./guardian_demo
```

## Creating Your Own Programs

To create a Zero-image program binary, you can:

1. Write a C program that generates the binary format
2. Use the provided `create_program` utility
3. Manually construct the header and records

### Binary Format

```
[Header - 32 bytes]
  uint32_t magic = 0x5A45524F
  uint32_t version = 0x00010000
  uint32_t record_count
  uint32_t flags
  uint64_t timestamp
  uint64_t reserved

[Records - 24 bytes each]
  double rho_real
  double rho_imag
  uint32_t depth
  float monodromy
```

### Example: Creating a Program

```c
#include "../zero_vm.h"

void create_example_program(const char *filename) {
    FILE *f = fopen(filename, "wb");
    
    // Write header
    ZeroHeader header = {
        .magic = ZERO_VM_MAGIC,
        .version = ZERO_VM_VERSION,
        .record_count = 3,
        .flags = 0,
        .timestamp = time(NULL),
        .reserved = 0
    };
    fwrite(&header, sizeof(ZeroHeader), 1, f);
    
    // Write records
    SpectralRecord records[3] = {
        {1.0, 2.0, 5, 0.5},
        {3.0, 4.0, 3, 1.0},
        {5.0, 6.0, 2, 1.5}
    };
    fwrite(records, sizeof(SpectralRecord), 3, f);
    
    fclose(f);
}
```
