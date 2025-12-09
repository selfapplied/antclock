/**
 * Witness Extraction Example
 * 
 * Demonstrates the WITNESS opcode by extracting full witness tuples
 * (Real, Imaginary, Depth, Monodromy) from spectral values.
 */

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "../zero_vm.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("=== Witness Extraction Example ===\n\n");
    
    /* Create VM */
    VMState *vm = vm_create();
    if (!vm) {
        fprintf(stderr, "Failed to create VM\n");
        return 1;
    }
    
    printf("Extracting witness tuples from spectral values:\n\n");
    
    /* Test witness 1: Simple value */
    printf("Witness 1: Simple spectral value\n");
    double complex rho1 = 3.0 + 4.0 * I;
    uint32_t depth1 = 7;
    float mono1 = M_PI / 3;
    
    printf("  Input: ρ = %.3f + %.3fi, depth = %u, mono = %.4f\n",
           creal(rho1), cimag(rho1), depth1, mono1);
    
    vm_push(vm, rho1, depth1, mono1);
    
    uint8_t sp_before = vm->sp;
    op_witness(vm);
    
    printf("  Witness components (pushed %u values):\n", vm->sp - sp_before);
    
    /* Pop and display all components */
    double complex mono_val, depth_val, imag_val, real_val;
    uint32_t dummy_depth;
    float dummy_mono;
    
    vm_pop(vm, &mono_val, &dummy_depth, &dummy_mono);
    printf("    Monodromy: %.4f\n", creal(mono_val));
    
    vm_pop(vm, &depth_val, &dummy_depth, &dummy_mono);
    printf("    Depth: %.0f\n", creal(depth_val));
    
    vm_pop(vm, &imag_val, &dummy_depth, &dummy_mono);
    printf("    Imaginary: %.3f\n", creal(imag_val));
    
    vm_pop(vm, &real_val, &dummy_depth, &dummy_mono);
    printf("    Real: %.3f\n", creal(real_val));
    
    printf("  Antclock: %lu\n\n", vm->antclock);
    
    /* Test witness 2: Complex value */
    printf("Witness 2: Complex spectral value\n");
    double complex rho2 = 1.414 + 2.718 * I;
    uint32_t depth2 = 12;
    float mono2 = 2 * M_PI / 3;
    
    printf("  Input: ρ = %.3f + %.3fi, depth = %u, mono = %.4f\n",
           creal(rho2), cimag(rho2), depth2, mono2);
    printf("  |ρ| = %.3f, arg(ρ) = %.4f\n", cabs(rho2), carg(rho2));
    
    vm_push(vm, rho2, depth2, mono2);
    
    sp_before = vm->sp;
    op_witness(vm);
    
    printf("  Witness extracted (%u components on stack)\n", vm->sp - sp_before);
    printf("  Antclock: %lu\n\n", vm->antclock);
    
    /* Test witness 3: Zero value */
    printf("Witness 3: Zero spectral value\n");
    double complex rho3 = 0.0 + 0.0 * I;
    uint32_t depth3 = 0;
    float mono3 = 0.0;
    
    printf("  Input: ρ = %.3f + %.3fi, depth = %u, mono = %.4f\n",
           creal(rho3), cimag(rho3), depth3, mono3);
    
    vm_push(vm, rho3, depth3, mono3);
    op_witness(vm);
    
    printf("  Witness extracted\n");
    printf("  Antclock: %lu\n\n", vm->antclock);
    
    /* Print final VM state */
    printf("--- Final VM State ---\n");
    print_vm_state(vm);
    
    /* Cleanup */
    vm_destroy(vm);
    
    printf("\n✓ Witness extraction example completed\n");
    return 0;
}
