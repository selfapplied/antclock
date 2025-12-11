/**
 * Zero-image μVM - Implementation
 * 
 * Minimal executable kernel for operationalizing CE1 bracket algebra
 * into a concrete, shippable runtime on standard hardware.
 */

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include "zero_vm.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * INITIALIZATION AND CLEANUP
 * ============================================================================ */

/**
 * Create a new VM instance
 */
VMState* vm_create(void) {
    VMState *vm = (VMState*)calloc(1, sizeof(VMState));
    if (!vm) {
        fprintf(stderr, "Failed to allocate VM state\n");
        return NULL;
    }
    
    vm->sp = 0;
    vm->antclock = 0;
    vm->pc = 0;
    vm->flags = 0;
    vm->k_spec = K_SPEC_DEFAULT;
    vm->program = NULL;
    vm->program_size = 0;
    
    /* Initialize planned feature fields */
    memset(vm->opcode_counts, 0, sizeof(vm->opcode_counts));
    vm->jit_compiled_path = NULL;
    memset(vm->remote_endpoint, 0, sizeof(vm->remote_endpoint));
    vm->breakpoint_count = 0;
    vm->trace_index = 0;
    vm->trace_enabled = false;
    
    /* Initialize gamma gap table */
    if (!antclock_init_table(vm)) {
        fprintf(stderr, "Failed to initialize antclock table\n");
        free(vm);
        return NULL;
    }
    
    return vm;
}

/**
 * Destroy VM instance and free resources
 */
void vm_destroy(VMState *vm) {
    if (!vm) return;
    
    if (vm->gamma_gaps) {
        free(vm->gamma_gaps);
    }
    
    if (vm->program) {
        free(vm->program);
    }
    
    free(vm);
}

/**
 * Load a program from file
 */
bool vm_load_program(VMState *vm, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return false;
    }
    
    /* Read header */
    ZeroHeader header;
    if (fread(&header, sizeof(ZeroHeader), 1, f) != 1) {
        fprintf(stderr, "Failed to read header\n");
        fclose(f);
        return false;
    }
    
    /* Validate header */
    if (header.magic != ZERO_VM_MAGIC) {
        fprintf(stderr, "Invalid magic number: 0x%08X\n", header.magic);
        fclose(f);
        return false;
    }
    
    if (header.version != ZERO_VM_VERSION) {
        fprintf(stderr, "Unsupported version: 0x%08X\n", header.version);
        fclose(f);
        return false;
    }
    
    /* Allocate program memory */
    vm->program_size = header.record_count;
    vm->program = (SpectralRecord*)malloc(sizeof(SpectralRecord) * vm->program_size);
    if (!vm->program) {
        fprintf(stderr, "Failed to allocate program memory\n");
        fclose(f);
        return false;
    }
    
    /* Read spectral records */
    if (fread(vm->program, sizeof(SpectralRecord), vm->program_size, f) != vm->program_size) {
        fprintf(stderr, "Failed to read program records\n");
        free(vm->program);
        vm->program = NULL;
        fclose(f);
        return false;
    }
    
    fclose(f);
    printf("Loaded program: %u records\n", vm->program_size);
    return true;
}

/* ============================================================================
 * STACK OPERATIONS
 * ============================================================================ */

/**
 * Push a spectral value onto the circular stack
 * 
 * Note: Python bindings call this with separate real/imaginary components
 * that need to be combined. For C API compatibility, consider adding:
 * void vm_push_components(VMState *vm, double real, double imag, uint32_t depth, float monodromy)
 */
void vm_push(VMState *vm, double complex rho, uint32_t depth, float monodromy) {
    SpectralFrame *frame = &vm->stack[vm->sp];
    frame->rho = rho;
    frame->depth = depth;
    frame->monodromy = monodromy;
    frame->timestamp = vm->antclock;
    frame->flags = 0;
    
    vm->sp = (vm->sp + 1) % STACK_DEPTH;
}

/**
 * Pop a spectral value from the circular stack
 */
bool vm_pop(VMState *vm, double complex *rho, uint32_t *depth, float *monodromy) {
    if (vm->sp == 0) {
        /* Stack underflow - wrap around */
        vm->sp = STACK_DEPTH - 1;
    } else {
        vm->sp = vm->sp - 1;
    }
    
    SpectralFrame *frame = &vm->stack[vm->sp];
    if (rho) *rho = frame->rho;
    if (depth) *depth = frame->depth;
    if (monodromy) *monodromy = frame->monodromy;
    
    return true;
}

/**
 * Peek at top of stack without popping
 */
SpectralFrame* vm_peek(VMState *vm) {
    /* Calculate peek position with proper wraparound
     * When sp=0, we want position 255 (STACK_DEPTH-1)
     * Otherwise we want sp-1 */
    uint16_t peek_sp = (vm->sp == 0) ? (STACK_DEPTH - 1) : (vm->sp - 1);
    return &vm->stack[peek_sp];
}

/* ============================================================================
 * OPCODE IMPLEMENTATIONS
 * ============================================================================ */

/**
 * OP_PROJECT (0x10) - {} Operator
 * Projects tensor path onto λ-isotypic component
 * Pops a "tensor path" and pushes a spectral value ρ
 */
void op_project(VMState *vm) {
    /* Pop tensor path index (encoded as complex number) */
    double complex path_idx;
    uint32_t dummy_depth;
    float dummy_mono;
    
    if (!vm_pop(vm, &path_idx, &dummy_depth, &dummy_mono)) {
        fprintf(stderr, "PROJECT: Stack underflow\n");
        return;
    }
    
    /* Extract real part as index into program */
    uint32_t idx = (uint32_t)creal(path_idx);
    if (idx >= vm->program_size) {
        idx = idx % vm->program_size;  /* Wrap around */
    }
    
    /* Project: extract spectral value from program */
    SpectralRecord *record = &vm->program[idx];
    double complex rho = record->rho_real + I * record->rho_imag;
    
    /* Push projected value */
    vm_push(vm, rho, record->depth, record->monodromy);
    
    /* Advance antclock */
    vm->antclock = antclock_tick(vm);
}

/**
 * OP_DEPTH (0x20) - [] Operator
 * Computes p-adic distance between two spectral values
 * Memory operator connecting spectral collision data
 */
void op_depth(VMState *vm) {
    /* Pop two values */
    double complex rho1, rho2;
    uint32_t depth1, depth2;
    float mono1, mono2;
    
    if (!vm_pop(vm, &rho1, &depth1, &mono1)) {
        fprintf(stderr, "DEPTH: Stack underflow\n");
        return;
    }
    
    if (!vm_pop(vm, &rho2, &depth2, &mono2)) {
        fprintf(stderr, "DEPTH: Stack underflow\n");
        vm_push(vm, rho1, depth1, mono1);  /* Restore first pop */
        return;
    }
    
    /* Compute p-adic distance (using p=2 for simplicity) */
    double dist = padic_distance(rho1, rho2, 2);
    
    /* Compute combined depth as average */
    uint32_t combined_depth = (depth1 + depth2) / 2;
    
    /* Push distance as real value */
    double complex result = dist + 0.0 * I;
    vm_push(vm, result, combined_depth, 0.0);
    
    /* Advance antclock */
    vm->antclock = antclock_tick(vm);
}

/**
 * OP_MORPH (0x30) - () Operator
 * Applies Hecke action T_φ(ρ)
 * Implements type-system flow morphism
 */
void op_morph(VMState *vm) {
    /* Pop morphism φ and value ρ */
    double complex phi, rho;
    uint32_t depth_phi, depth_rho;
    float mono_phi, mono_rho;
    
    if (!vm_pop(vm, &phi, &depth_phi, &mono_phi)) {
        fprintf(stderr, "MORPH: Stack underflow\n");
        return;
    }
    
    if (!vm_pop(vm, &rho, &depth_rho, &mono_rho)) {
        fprintf(stderr, "MORPH: Stack underflow\n");
        vm_push(vm, phi, depth_phi, mono_phi);  /* Restore first pop */
        return;
    }
    
    /* Apply Hecke action */
    double complex result = hecke_action(rho, phi);
    
    /* Combine depths using guardian logic */
    double dist = cabs(rho - phi);
    GuardianDecision decision = guardian_decide(dist, vm->k_spec);
    
    uint32_t result_depth;
    if (decision == GUARDIAN_COMPOSE) {
        /* Merge: average depths */
        result_depth = (depth_rho + depth_phi) / 2;
    } else {
        /* Protect: keep rho's depth */
        result_depth = depth_rho;
    }
    
    /* Combine monodromy angles */
    float result_mono = fmod(mono_rho + mono_phi, 2.0 * M_PI);
    
    /* Push result */
    vm_push(vm, result, result_depth, result_mono);
    
    /* Advance antclock */
    vm->antclock = antclock_tick(vm);
}

/**
 * OP_WITNESS (0x40) - <> Operator
 * Extracts full witness tuple (Real, Imaginary, Depth, Monodromy)
 * For now, just duplicates the top value to make all components visible
 */
void op_witness(VMState *vm) {
    /* Peek at top value without popping */
    SpectralFrame *frame = vm_peek(vm);
    
    /* Push components separately for inspection */
    /* Real part */
    vm_push(vm, creal(frame->rho) + 0.0 * I, frame->depth, frame->monodromy);
    
    /* Imaginary part */
    vm_push(vm, cimag(frame->rho) + 0.0 * I, frame->depth, frame->monodromy);
    
    /* Depth as real value */
    vm_push(vm, (double)frame->depth + 0.0 * I, frame->depth, frame->monodromy);
    
    /* Monodromy as real value */
    vm_push(vm, frame->monodromy + 0.0 * I, frame->depth, frame->monodromy);
    
    /* Advance antclock */
    vm->antclock = antclock_tick(vm);
}

/* ============================================================================
 * GUARDIAN LOGIC (14 x86 instructions)
 * ============================================================================ */

/**
 * Guardian decision: compose or protect
 * 
 * Extremely efficient implementation - compiles to approximately:
 * - 1 comparison instruction (cmp/ucomisd)
 * - 1 conditional move/set (cmov/setcc)
 * - 1-2 return instructions
 * Total: ~3-5 x86 instructions depending on compiler optimization
 * 
 * Branch-predictor friendly due to simple linear flow
 */
GuardianDecision guardian_decide(double depth, double k_spec) {
    /* Simple threshold comparison */
    return (depth < k_spec) ? GUARDIAN_PROTECT : GUARDIAN_COMPOSE;
}

/* ============================================================================
 * ANTCLOCK INTEGRATION
 * ============================================================================ */

/* Constant to avoid log(0) in gamma gap calculation */
#define LOG_OFFSET 10.0

/**
 * Initialize gamma gap lookup table
 * Pre-computed spacing between first 100k zeta zeros
 * For now, we use a simple approximation based on the average gap
 */
bool antclock_init_table(VMState *vm) {
    vm->gamma_table_size = GAMMA_GAP_TABLE_SIZE;
    vm->gamma_gaps = (float*)malloc(sizeof(float) * vm->gamma_table_size);
    
    if (!vm->gamma_gaps) {
        return false;
    }
    
    /* Initialize with approximate gamma gaps
     * Average spacing near the Nth zero is approximately 2π/log(N)
     * This is the Riemann-von Mangoldt formula
     */
    for (uint32_t i = 0; i < vm->gamma_table_size; i++) {
        double n = (double)(i + 1);
        /* Use LOG_OFFSET to avoid log(0) for small n */
        double avg_gap = (2.0 * M_PI) / log(n + LOG_OFFSET);
        
        /* Add small deterministic variation */
        double variation = sin(n * 0.1) * 0.1 + cos(n * 0.05) * 0.05;
        vm->gamma_gaps[i] = (float)(avg_gap * (1.0 + variation));
    }
    
    return true;
}

/**
 * Generate antclock tick
 * Uses hash of current state modulo table size
 */
uint64_t antclock_tick(VMState *vm) {
    /* Get current top frame */
    SpectralFrame *frame = vm_peek(vm);
    
    /* Hash based on imaginary component */
    double imag = cimag(frame->rho);
    uint64_t hash = (uint64_t)(fabs(imag) * 1000000.0);
    
    /* Lookup gamma gap */
    uint32_t idx = hash % vm->gamma_table_size;
    float gap = vm->gamma_gaps[idx];
    
    /* Increment antclock by gap (scaled to integer) */
    return vm->antclock + (uint64_t)(gap * 1000.0);
}

/**
 * Get gamma gap for a specific tick
 */
float antclock_gamma_gap(VMState *vm, uint64_t tick) {
    uint32_t idx = tick % vm->gamma_table_size;
    return vm->gamma_gaps[idx];
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * Compute p-adic distance between two complex numbers
 * Simplified implementation using standard absolute value
 */
double padic_distance(double complex rho1, double complex rho2, uint32_t p) {
    double complex diff = rho1 - rho2;
    double abs_diff = cabs(diff);
    
    /* p-adic valuation approximation */
    if (abs_diff < 1e-10) {
        /* Distance to self is 0, but we return a small positive value
         * to indicate "close" rather than exact match */
        return 0.0;
    }
    
    /* Simple p-adic-like metric: use absolute value as distance
     * For a true p-adic metric, we'd compute valuation
     * Here we use a simplified version that's always positive */
    return abs_diff;
}

/**
 * Apply Hecke action T_φ(ρ)
 * Simplified implementation: rotation and scaling
 */
double complex hecke_action(double complex rho, double complex phi) {
    /* Hecke action as multiplicative twist */
    double complex result = rho * cexp(I * carg(phi));
    
    /* Scale by magnitude of phi */
    result *= cabs(phi);
    
    return result;
}

/**
 * Print a spectral frame
 */
void print_frame(SpectralFrame *frame) {
    printf("ρ = %.6f + %.6fi, depth = %u, mono = %.4f, tick = %lu\n",
           creal(frame->rho), cimag(frame->rho),
           frame->depth, frame->monodromy, frame->timestamp);
}

/**
 * Print VM state
 */
void print_vm_state(VMState *vm) {
    printf("=== VM State ===\n");
    printf("PC: %u, SP: %u, Antclock: %lu\n", vm->pc, vm->sp, vm->antclock);
    printf("Stack (top 5 frames):\n");
    
    for (int i = 0; i < 5 && i < STACK_DEPTH; i++) {
        uint8_t idx = (vm->sp - 1 - i + STACK_DEPTH) % STACK_DEPTH;
        printf("  [%d] ", i);
        print_frame(&vm->stack[idx]);
    }
}

/* ============================================================================
 * VM EXECUTION
 * ============================================================================ */

/**
 * Execute a single opcode
 */
bool vm_execute_opcode(VMState *vm, Opcode op) {
    /* Track opcode execution count for JIT (Planned Feature #1) */
    uint8_t idx = 0;
    switch (op) {
        case OP_PROJECT: idx = 0; break;
        case OP_DEPTH:   idx = 1; break;
        case OP_MORPH:   idx = 2; break;
        case OP_WITNESS: idx = 3; break;
        default:
            fprintf(stderr, "Unknown opcode: 0x%02X\n", op);
            return false;
    }
    vm->opcode_counts[idx]++;
    
    /* Check if JIT compilation should be triggered */
    if (vm_jit_should_compile(vm, op)) {
        vm_jit_compile_hot_path(vm);
    }
    
    /* Execute opcode */
    switch (op) {
        case OP_PROJECT:
            op_project(vm);
            break;
        case OP_DEPTH:
            op_depth(vm);
            break;
        case OP_MORPH:
            op_morph(vm);
            break;
        case OP_WITNESS:
            op_witness(vm);
            break;
    }
    
    /* Record trace if enabled (Planned Feature #5) */
    if (vm->trace_enabled) {
        vm_trace_record(vm);
    }
    
    return true;
}

/**
 * Execute one step
 * 
 * Note: Opcodes are encoded in the low byte of the depth field.
 * This design allows each spectral record to contain both data (ρ, monodromy)
 * and an instruction (opcode). The depth field serves dual purpose:
 * - Low byte (0xFF): Opcode to execute
 * - High bytes: Actual ultrametric depth value
 * 
 * This is an intentional design choice to keep the record size minimal (24 bytes)
 * while supporting both data and instructions in a unified format.
 */
void vm_step(VMState *vm) {
    if (vm->pc >= vm->program_size) {
        printf("Program counter out of bounds\n");
        return;
    }
    
    /* Fetch opcode from current record's depth field (low byte) */
    SpectralRecord *record = &vm->program[vm->pc];
    Opcode op = (Opcode)(record->depth & 0xFF);
    
    /* Execute */
    vm_execute_opcode(vm, op);
    
    /* Advance PC */
    vm->pc++;
}

/**
 * Run VM until program completes
 */
void vm_run(VMState *vm) {
    printf("Starting VM execution...\n");
    
    while (vm->pc < vm->program_size) {
        vm_step(vm);
    }
    
    printf("VM execution completed\n");
    print_vm_state(vm);
}

/* ============================================================================
 * PLANNED FEATURE #1: JIT COMPILATION FOR HOT PATHS
 * ============================================================================ */

/**
 * Check if opcode should be JIT compiled
 * Returns true when execution count exceeds threshold
 */
bool vm_jit_should_compile(VMState *vm, Opcode op) {
    if (!(vm->flags & VM_FLAG_JIT)) {
        return false;
    }
    
    uint8_t idx = 0;
    switch(op) {
        case OP_PROJECT: idx = 0; break;
        case OP_DEPTH:   idx = 1; break;
        case OP_MORPH:   idx = 2; break;
        case OP_WITNESS: idx = 3; break;
    }
    
    return vm->opcode_counts[idx] >= JIT_HOT_PATH_THRESHOLD;
}

/**
 * JIT compile hot path (stub for future implementation)
 * 
 * Future implementation would:
 * 1. Analyze opcode sequence for optimization opportunities
 * 2. Generate native code for hot loops
 * 3. Cache compiled code in jit_compiled_path
 * 4. Use mmap/mprotect for executable memory
 */
void vm_jit_compile_hot_path(VMState *vm) {
    if (!vm) return;
    
    /* Placeholder: In production, this would:
     * - Use LLVM or dynasm for code generation
     * - Allocate executable memory region
     * - Compile frequently executed opcode sequences
     * - Install function pointer for fast dispatch
     */
    
    printf("[JIT] Hot path detected - JIT compilation hook ready\n");
    printf("[JIT] To enable: link with JIT compiler library\n");
}

/* ============================================================================
 * PLANNED FEATURE #2: NETWORK-TRANSPARENT EXECUTION
 * ============================================================================ */

/**
 * Serialize VM state to file
 * Uses existing binary format for network transmission
 */
bool vm_serialize_state(VMState *vm, const char *filename) {
    if (!vm || !filename) return false;
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return false;
    }
    
    /* Write stack pointer and program counter */
    fwrite(&vm->sp, sizeof(uint8_t), 1, f);
    fwrite(&vm->pc, sizeof(uint32_t), 1, f);
    fwrite(&vm->antclock, sizeof(uint64_t), 1, f);
    
    /* Write stack frames */
    fwrite(vm->stack, sizeof(SpectralFrame), STACK_DEPTH, f);
    
    fclose(f);
    return true;
}

/**
 * Deserialize VM state from file
 */
bool vm_deserialize_state(VMState *vm, const char *filename) {
    if (!vm || !filename) return false;
    
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for reading\n", filename);
        return false;
    }
    
    /* Read stack pointer and program counter */
    fread(&vm->sp, sizeof(uint8_t), 1, f);
    fread(&vm->pc, sizeof(uint32_t), 1, f);
    fread(&vm->antclock, sizeof(uint64_t), 1, f);
    
    /* Read stack frames */
    fread(vm->stack, sizeof(SpectralFrame), STACK_DEPTH, f);
    
    fclose(f);
    return true;
}

/**
 * Execute VM operations on remote endpoint (stub)
 * 
 * Future implementation would:
 * 1. Serialize current VM state
 * 2. Send to remote endpoint via gRPC/WebSocket
 * 3. Receive computed result
 * 4. Deserialize back into local VM
 */
bool vm_remote_exec(VMState *vm, const char *endpoint) {
    if (!vm || !endpoint) return false;
    
    if (!(vm->flags & VM_FLAG_REMOTE)) {
        fprintf(stderr, "Remote execution not enabled (set VM_FLAG_REMOTE)\n");
        return false;
    }
    
    /* Save endpoint for potential future use */
    strncpy(vm->remote_endpoint, endpoint, sizeof(vm->remote_endpoint) - 1);
    vm->remote_endpoint[sizeof(vm->remote_endpoint) - 1] = '\0';
    
    printf("[Remote] Endpoint configured: %s\n", endpoint);
    printf("[Remote] Serialization format: vm_serialize_state/vm_deserialize_state\n");
    printf("[Remote] To enable: implement network protocol (gRPC, WebSocket, etc.)\n");
    
    return true;
}

/* ============================================================================
 * PLANNED FEATURE #3: PERSISTENT STATE CHECKPOINTING
 * ============================================================================ */

/**
 * Save complete VM state to checkpoint file
 */
bool vm_checkpoint_save(VMState *vm, const char *filename) {
    if (!vm || !filename) return false;
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to create checkpoint: %s\n", filename);
        return false;
    }
    
    /* Write checkpoint magic and version */
    uint32_t magic = 0x434B5054;  /* "CKPT" */
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    
    /* Write VM state */
    fwrite(&vm->sp, sizeof(uint8_t), 1, f);
    fwrite(&vm->antclock, sizeof(uint64_t), 1, f);
    fwrite(&vm->pc, sizeof(uint32_t), 1, f);
    fwrite(&vm->flags, sizeof(uint32_t), 1, f);
    fwrite(&vm->k_spec, sizeof(double), 1, f);
    
    /* Write stack */
    fwrite(vm->stack, sizeof(SpectralFrame), STACK_DEPTH, f);
    
    /* Write program size and data */
    fwrite(&vm->program_size, sizeof(uint32_t), 1, f);
    if (vm->program_size > 0) {
        fwrite(vm->program, sizeof(SpectralRecord), vm->program_size, f);
    }
    
    fclose(f);
    printf("[Checkpoint] State saved to %s\n", filename);
    return true;
}

/**
 * Load VM state from checkpoint file
 */
bool vm_checkpoint_load(VMState *vm, const char *filename) {
    if (!vm || !filename) return false;
    
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open checkpoint: %s\n", filename);
        return false;
    }
    
    /* Read and verify magic and version */
    uint32_t magic, version;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&version, sizeof(uint32_t), 1, f);
    
    if (magic != 0x434B5054) {
        fprintf(stderr, "Invalid checkpoint file: bad magic\n");
        fclose(f);
        return false;
    }
    
    /* Read VM state */
    fread(&vm->sp, sizeof(uint8_t), 1, f);
    fread(&vm->antclock, sizeof(uint64_t), 1, f);
    fread(&vm->pc, sizeof(uint32_t), 1, f);
    fread(&vm->flags, sizeof(uint32_t), 1, f);
    fread(&vm->k_spec, sizeof(double), 1, f);
    
    /* Read stack */
    fread(vm->stack, sizeof(SpectralFrame), STACK_DEPTH, f);
    
    /* Read program */
    uint32_t prog_size;
    fread(&prog_size, sizeof(uint32_t), 1, f);
    
    if (prog_size > 0) {
        if (vm->program) {
            free(vm->program);
        }
        vm->program = (SpectralRecord*)malloc(prog_size * sizeof(SpectralRecord));
        if (!vm->program) {
            fprintf(stderr, "Failed to allocate program memory\n");
            fclose(f);
            return false;
        }
        vm->program_size = prog_size;
        fread(vm->program, sizeof(SpectralRecord), prog_size, f);
    }
    
    fclose(f);
    printf("[Checkpoint] State restored from %s\n", filename);
    return true;
}

/* ============================================================================
 * PLANNED FEATURE #4: INTERACTIVE DEBUGGER
 * ============================================================================ */

/**
 * Add breakpoint at program counter
 */
void vm_debug_add_breakpoint(VMState *vm, uint32_t pc) {
    if (!vm || vm->breakpoint_count >= MAX_BREAKPOINTS) return;
    
    vm->breakpoints[vm->breakpoint_count++] = pc;
    printf("[Debug] Breakpoint added at PC=%u\n", pc);
}

/**
 * Remove breakpoint at program counter
 */
void vm_debug_remove_breakpoint(VMState *vm, uint32_t pc) {
    if (!vm) return;
    
    for (uint8_t i = 0; i < vm->breakpoint_count; i++) {
        if (vm->breakpoints[i] == pc) {
            /* Shift remaining breakpoints */
            for (uint8_t j = i; j < vm->breakpoint_count - 1; j++) {
                vm->breakpoints[j] = vm->breakpoints[j + 1];
            }
            vm->breakpoint_count--;
            printf("[Debug] Breakpoint removed at PC=%u\n", pc);
            return;
        }
    }
}

/**
 * Check if current PC has a breakpoint
 */
bool vm_debug_is_breakpoint(VMState *vm, uint32_t pc) {
    if (!vm) return false;
    
    for (uint8_t i = 0; i < vm->breakpoint_count; i++) {
        if (vm->breakpoints[i] == pc) {
            return true;
        }
    }
    return false;
}

/**
 * Execute one step with debugger support
 * Pauses at breakpoints when VM_FLAG_DEBUG is set
 */
void vm_debug_step(VMState *vm) {
    if (!vm) return;
    
    /* Check for breakpoint */
    if ((vm->flags & VM_FLAG_DEBUG) && vm_debug_is_breakpoint(vm, vm->pc)) {
        printf("\n[Debug] Breakpoint hit at PC=%u\n", vm->pc);
        print_vm_state(vm);
        printf("[Debug] Press Enter to continue...");
        getchar();
    }
    
    /* Execute step */
    vm_step(vm);
    
    /* Show state if in debug mode */
    if (vm->flags & VM_FLAG_DEBUG) {
        printf("[Debug] PC=%u, SP=%u, Antclock=%lu\n", 
               vm->pc, vm->sp, vm->antclock);
    }
}

/* ============================================================================
 * PLANNED FEATURE #5: VISUAL TRACER FOR SPECTRAL TRAJECTORIES
 * ============================================================================ */

/**
 * Enable or disable trajectory tracing
 */
void vm_trace_enable(VMState *vm, bool enabled) {
    if (!vm) return;
    
    vm->trace_enabled = enabled;
    if (enabled) {
        vm->flags |= VM_FLAG_TRACE;
        printf("[Trace] Trajectory recording enabled\n");
    } else {
        vm->flags &= ~VM_FLAG_TRACE;
        printf("[Trace] Trajectory recording disabled\n");
    }
}

/**
 * Record current top frame to trace history
 * Should be called after each VM operation when tracing is enabled
 */
void vm_trace_record(VMState *vm) {
    if (!vm || !vm->trace_enabled) return;
    
    /* Get current top frame */
    SpectralFrame *current = vm_peek(vm);
    if (!current) return;
    
    /* Copy to trace history (circular buffer) */
    vm->trace_history[vm->trace_index] = *current;
    vm->trace_index = (vm->trace_index + 1) % TRACE_HISTORY_SIZE;
}

/**
 * Export trace history to file
 * Supports CSV and JSON formats for visualization
 */
bool vm_trace_export(VMState *vm, const char *filename, const char *format) {
    if (!vm || !filename || !format) return false;
    
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Failed to create trace file: %s\n", filename);
        return false;
    }
    
    if (strcmp(format, "csv") == 0) {
        /* CSV format for gnuplot/matplotlib */
        fprintf(f, "index,rho_real,rho_imag,depth,monodromy,timestamp\n");
        
        for (int i = 0; i < TRACE_HISTORY_SIZE; i++) {
            SpectralFrame *frame = &vm->trace_history[i];
            if (frame->timestamp == 0) continue;  /* Skip empty slots */
            
            fprintf(f, "%d,%.10f,%.10f,%u,%.6f,%lu\n",
                   i,
                   creal(frame->rho),
                   cimag(frame->rho),
                   frame->depth,
                   frame->monodromy,
                   frame->timestamp);
        }
    } else if (strcmp(format, "json") == 0) {
        /* JSON format for web visualization */
        fprintf(f, "{\n  \"traces\": [\n");
        
        bool first = true;
        for (int i = 0; i < TRACE_HISTORY_SIZE; i++) {
            SpectralFrame *frame = &vm->trace_history[i];
            if (frame->timestamp == 0) continue;
            
            if (!first) fprintf(f, ",\n");
            first = false;
            
            fprintf(f, "    {\n");
            fprintf(f, "      \"index\": %d,\n", i);
            fprintf(f, "      \"rho\": {\"real\": %.10f, \"imag\": %.10f},\n",
                   creal(frame->rho), cimag(frame->rho));
            fprintf(f, "      \"depth\": %u,\n", frame->depth);
            fprintf(f, "      \"monodromy\": %.6f,\n", frame->monodromy);
            fprintf(f, "      \"timestamp\": %lu\n", frame->timestamp);
            fprintf(f, "    }");
        }
        
        fprintf(f, "\n  ]\n}\n");
    } else {
        fprintf(stderr, "Unknown format: %s (use 'csv' or 'json')\n", format);
        fclose(f);
        return false;
    }
    
    fclose(f);
    printf("[Trace] Exported %d frames to %s (%s format)\n", 
           TRACE_HISTORY_SIZE, filename, format);
    printf("[Trace] Visualize with: matplotlib (Python) or gnuplot\n");
    
    return true;
}
