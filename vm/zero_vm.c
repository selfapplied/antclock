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
    uint8_t peek_sp = (vm->sp == 0) ? (STACK_DEPTH - 1) : (vm->sp - 1);
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
 * Extremely efficient: ~14 x86 instructions, branch-predictor friendly
 */
GuardianDecision guardian_decide(double depth, double k_spec) {
    /* Simple threshold comparison */
    return (depth < k_spec) ? GUARDIAN_PROTECT : GUARDIAN_COMPOSE;
}

/* ============================================================================
 * ANTCLOCK INTEGRATION
 * ============================================================================ */

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
        double avg_gap = (2.0 * M_PI) / log(n + 10.0);  /* +10 to avoid log(0) */
        
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
        default:
            fprintf(stderr, "Unknown opcode: 0x%02X\n", op);
            return false;
    }
    return true;
}

/**
 * Execute one step
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
