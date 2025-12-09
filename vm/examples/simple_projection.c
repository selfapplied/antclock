/**
 * Simple Projection Example
 * 
 * Demonstrates the PROJECT opcode by loading spectral values
 * from a program and projecting them onto the λ-isotypic component.
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "../zero_vm.h"

int main(void) {
    printf("=== Simple Projection Example ===\n\n");
    
    /* Create VM */
    VMState *vm = vm_create();
    if (!vm) {
        fprintf(stderr, "Failed to create VM\n");
        return 1;
    }
    
    /* Create a simple in-memory program */
    vm->program_size = 5;
    vm->program = (SpectralRecord*)malloc(sizeof(SpectralRecord) * vm->program_size);
    if (!vm->program) {
        fprintf(stderr, "Failed to allocate program\n");
        vm_destroy(vm);
        return 1;
    }
    
    /* Initialize spectral records */
    printf("Creating spectral program with %u records:\n", vm->program_size);
    for (uint32_t i = 0; i < vm->program_size; i++) {
        vm->program[i].rho_real = (double)(i + 1) * 1.5;
        vm->program[i].rho_imag = (double)(i + 1) * 0.7;
        vm->program[i].depth = (i + 1) * 2;
        vm->program[i].monodromy = (float)i * 0.3;
        
        printf("  [%u] ρ = %.3f + %.3fi, depth = %u, mono = %.3f\n",
               i, vm->program[i].rho_real, vm->program[i].rho_imag,
               vm->program[i].depth, vm->program[i].monodromy);
    }
    
    printf("\n--- Executing Projections ---\n\n");
    
    /* Project each spectral value */
    for (uint32_t i = 0; i < vm->program_size; i++) {
        printf("Projection %u:\n", i);
        
        /* Push index to project */
        vm_push(vm, (double)i + 0.0 * I, 0, 0.0);
        printf("  Pushed index: %u\n", i);
        
        /* Execute PROJECT opcode */
        op_project(vm);
        printf("  Executed PROJECT\n");
        
        /* Examine result */
        SpectralFrame *frame = vm_peek(vm);
        printf("  Result: ρ = %.3f + %.3fi, depth = %u, mono = %.3f\n",
               creal(frame->rho), cimag(frame->rho),
               frame->depth, frame->monodromy);
        printf("  Antclock: %lu\n\n", vm->antclock);
    }
    
    /* Print final VM state */
    printf("--- Final VM State ---\n");
    print_vm_state(vm);
    
    /* Cleanup */
    vm_destroy(vm);
    
    printf("\n✓ Projection example completed\n");
    return 0;
}
