/**
 * Zero-image μVM - Minimal Executable Kernel for CE System
 * 
 * A 400-line C implementation of a spectral virtual machine that operationalizes
 * CE1 bracket algebra and tensor fields into a concrete, shippable artifact.
 * 
 * Design Goals:
 * - Binary size: ~18 kB
 * - Memory footprint: 16 KB (L1 cache friendly)
 * - Zero dependencies (standard C library only)
 * - Docker and WASM compatible
 * 
 * Version: 0.1.0
 */

#ifndef ZERO_VM_H
#define ZERO_VM_H

#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

/* ============================================================================
 * FILE FORMAT CONSTANTS
 * ============================================================================ */

#define ZERO_VM_MAGIC 0x5A45524F  /* "ZERO" in ASCII */
#define ZERO_VM_VERSION 0x00010000 /* v0.1.0 */
#define HEADER_SIZE 32
#define RECORD_SIZE 24

/* ============================================================================
 * VM ARCHITECTURE CONSTANTS
 * ============================================================================ */

#define FRAME_SIZE 64       /* Size of each spectral frame in bytes */
#define STACK_DEPTH 256     /* Maximum stack depth (16 KB total) */
#define GAMMA_GAP_TABLE_SIZE 100000  /* Pre-computed zeta zero spacings */

/* JIT compilation constants */
#define JIT_HOT_PATH_THRESHOLD 1000  /* Executions before JIT compilation */

/* Debugger constants */
#define MAX_BREAKPOINTS 16   /* Maximum number of breakpoints */

/* Tracer constants */
#define TRACE_HISTORY_SIZE 256  /* Number of frames to keep in trace history */

/* ============================================================================
 * OPCODES - CE1 Bracket Algebra
 * ============================================================================ */

typedef enum {
    OP_PROJECT = 0x10,  /* {} - Project tensor path to spectral value ρ */
    OP_DEPTH   = 0x20,  /* [] - Compute p-adic distance (memory operator) */
    OP_MORPH   = 0x30,  /* () - Apply Hecke action (type-system flow) */
    OP_WITNESS = 0x40   /* <> - Extract full witness tuple */
} Opcode;

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * File Header (32 bytes)
 * Describes the binary format and metadata
 */
typedef struct {
    uint32_t magic;           /* Magic number: "ZERO" */
    uint32_t version;         /* Version: 0x00010000 for v0.1.0 */
    uint32_t record_count;    /* Number of spectral records */
    uint32_t flags;           /* Feature flags (quantum mode, etc.) */
    uint64_t timestamp;       /* Creation timestamp */
    uint64_t reserved;        /* Reserved for future use */
} ZeroHeader;

/**
 * Spectral Record (24 bytes)
 * Represents a point in the spectral field with p-adic structure
 */
typedef struct {
    double rho_real;          /* Real part of ρ (8 bytes) */
    double rho_imag;          /* Imaginary part of ρ (8 bytes) */
    uint32_t depth;           /* Ultrametric depth: -log_p |Im ρ|_p (4 bytes) */
    float monodromy;          /* Monodromy angle in radians (4 bytes) */
} SpectralRecord;

/**
 * Spectral Frame (64 bytes)
 * Single frame in the circular stack
 */
typedef struct {
    double complex rho;       /* Complex spectral value (16 bytes) */
    uint32_t depth;           /* Ultrametric depth (4 bytes) */
    float monodromy;          /* Monodromy angle (4 bytes) */
    uint64_t timestamp;       /* Antclock tick (8 bytes) */
    uint32_t flags;           /* Frame flags (4 bytes) */
    uint8_t padding[28];      /* Pad to 64 bytes for alignment */
} SpectralFrame;

/**
 * VM State
 * Complete state of the Zero-image μVM
 */
typedef struct {
    SpectralFrame stack[STACK_DEPTH];  /* Circular spectral stack (16 KB) */
    uint8_t sp;                         /* Stack pointer (0-255) */
    uint64_t antclock;                  /* Current antclock tick */
    uint32_t pc;                        /* Program counter */
    uint32_t flags;                     /* VM flags */
    
    /* Guardian parameters */
    double k_spec;                      /* Spectral threshold constant */
    
    /* Antclock gamma gap table (for deterministic ticks) */
    float *gamma_gaps;                  /* Pointer to gamma gap lookup table */
    uint32_t gamma_table_size;          /* Size of gamma gap table */
    
    /* Program data */
    SpectralRecord *program;            /* Loaded program records */
    uint32_t program_size;              /* Number of records in program */
    
    /* JIT compilation support */
    uint64_t opcode_counts[4];          /* Execution count per opcode */
    void (*jit_compiled_path)(void);    /* JIT compiled hot path function */
    
    /* Network-transparent execution */
    char remote_endpoint[256];          /* Remote execution endpoint */
    
    /* Interactive debugger */
    uint32_t breakpoints[MAX_BREAKPOINTS];  /* Breakpoint program counters */
    uint8_t breakpoint_count;           /* Number of active breakpoints */
    
    /* Visual tracer */
    SpectralFrame trace_history[TRACE_HISTORY_SIZE];  /* Circular trace buffer */
    uint8_t trace_index;                /* Current position in trace buffer */
    bool trace_enabled;                 /* Whether tracing is active */
} VMState;

/* ============================================================================
 * GUARDIAN CONSTANTS
 * ============================================================================ */

/* Spectral threshold for compose/protect decision */
#define K_SPEC_DEFAULT 0.5

/* Guardian decision: compose (merge) or protect (separate) */
typedef enum {
    GUARDIAN_PROTECT = 0,   /* States are too different - keep separate */
    GUARDIAN_COMPOSE = 1    /* States are compatible - merge them */
} GuardianDecision;

/* VM flags */
#define VM_FLAG_DEBUG       0x01  /* Enable interactive debugging */
#define VM_FLAG_TRACE       0x02  /* Enable trajectory tracing */
#define VM_FLAG_JIT         0x04  /* Enable JIT compilation */
#define VM_FLAG_REMOTE      0x08  /* Enable remote execution */

/* ============================================================================
 * FUNCTION DECLARATIONS
 * ============================================================================ */

/* Initialization and cleanup */
VMState* vm_create(void);
void vm_destroy(VMState *vm);
bool vm_load_program(VMState *vm, const char *filename);

/* Core VM operations */
bool vm_execute_opcode(VMState *vm, Opcode op);
void vm_step(VMState *vm);
void vm_run(VMState *vm);

/* Stack operations */
void vm_push(VMState *vm, double complex rho, uint32_t depth, float monodromy);
bool vm_pop(VMState *vm, double complex *rho, uint32_t *depth, float *monodromy);
SpectralFrame* vm_peek(VMState *vm);

/* Opcode implementations */
void op_project(VMState *vm);      /* {} - Project to λ-isotypic component */
void op_depth(VMState *vm);        /* [] - Compute p-adic distance */
void op_morph(VMState *vm);        /* () - Apply Hecke action */
void op_witness(VMState *vm);      /* <> - Extract witness tuple */

/* Guardian logic */
GuardianDecision guardian_decide(double depth, double k_spec);

/* Antclock integration */
uint64_t antclock_tick(VMState *vm);
float antclock_gamma_gap(VMState *vm, uint64_t tick);
bool antclock_init_table(VMState *vm);

/* Utility functions */
double padic_distance(double complex rho1, double complex rho2, uint32_t p);
double complex hecke_action(double complex rho, double complex phi);
void print_frame(SpectralFrame *frame);
void print_vm_state(VMState *vm);

/* JIT compilation (Planned Feature #1) */
void vm_jit_compile_hot_path(VMState *vm);
bool vm_jit_should_compile(VMState *vm, Opcode op);

/* Network-transparent execution (Planned Feature #2) */
bool vm_remote_exec(VMState *vm, const char *endpoint);
bool vm_serialize_state(VMState *vm, const char *filename);
bool vm_deserialize_state(VMState *vm, const char *filename);

/* Persistent state checkpointing (Planned Feature #3) */
bool vm_checkpoint_save(VMState *vm, const char *filename);
bool vm_checkpoint_load(VMState *vm, const char *filename);

/* Interactive debugger (Planned Feature #4) */
void vm_debug_step(VMState *vm);
void vm_debug_add_breakpoint(VMState *vm, uint32_t pc);
void vm_debug_remove_breakpoint(VMState *vm, uint32_t pc);
bool vm_debug_is_breakpoint(VMState *vm, uint32_t pc);

/* Visual tracer (Planned Feature #5) */
void vm_trace_record(VMState *vm);
bool vm_trace_export(VMState *vm, const char *filename, const char *format);
void vm_trace_enable(VMState *vm, bool enabled);

#endif /* ZERO_VM_H */
