/**
 * Distance Computation Example
 * 
 * Demonstrates the DEPTH opcode by computing p-adic distances
 * between spectral values.
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "../zero_vm.h"

int main(void) {
    printf("=== Distance Computation Example ===\n\n");
    
    /* Create VM */
    VMState *vm = vm_create();
    if (!vm) {
        fprintf(stderr, "Failed to create VM\n");
        return 1;
    }
    
    printf("Computing p-adic distances between spectral values:\n\n");
    
    /* Test case 1: Close values */
    printf("Test 1: Close values\n");
    double complex rho1 = 1.0 + 1.0 * I;
    double complex rho2 = 1.1 + 1.1 * I;
    
    vm_push(vm, rho1, 5, 0.0);
    vm_push(vm, rho2, 5, 0.0);
    printf("  ρ₁ = %.3f + %.3fi\n", creal(rho1), cimag(rho1));
    printf("  ρ₂ = %.3f + %.3fi\n", creal(rho2), cimag(rho2));
    
    op_depth(vm);
    
    SpectralFrame *frame1 = vm_peek(vm);
    printf("  Distance: %.6f\n", creal(frame1->rho));
    printf("  Antclock: %lu\n\n", vm->antclock);
    
    /* Test case 2: Far values */
    printf("Test 2: Distant values\n");
    double complex rho3 = 1.0 + 1.0 * I;
    double complex rho4 = 10.0 + 10.0 * I;
    
    vm_push(vm, rho3, 3, 0.5);
    vm_push(vm, rho4, 2, 1.0);
    printf("  ρ₃ = %.3f + %.3fi\n", creal(rho3), cimag(rho3));
    printf("  ρ₄ = %.3f + %.3fi\n", creal(rho4), cimag(rho4));
    
    op_depth(vm);
    
    SpectralFrame *frame2 = vm_peek(vm);
    printf("  Distance: %.6f\n", creal(frame2->rho));
    printf("  Antclock: %lu\n\n", vm->antclock);
    
    /* Test case 3: Identical values */
    printf("Test 3: Identical values\n");
    double complex rho5 = 3.14 + 2.71 * I;
    
    vm_push(vm, rho5, 4, 0.7);
    vm_push(vm, rho5, 4, 0.7);
    printf("  ρ₅ = %.3f + %.3fi (both values)\n", creal(rho5), cimag(rho5));
    
    op_depth(vm);
    
    SpectralFrame *frame3 = vm_peek(vm);
    printf("  Distance: %.6f (should be ~0)\n", creal(frame3->rho));
    printf("  Antclock: %lu\n\n", vm->antclock);
    
    /* Print final VM state */
    printf("--- Final VM State ---\n");
    print_vm_state(vm);
    
    /* Cleanup */
    vm_destroy(vm);
    
    printf("\n✓ Distance computation example completed\n");
    return 0;
}
