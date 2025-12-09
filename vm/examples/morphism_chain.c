/**
 * Morphism Chain Example
 * 
 * Demonstrates the MORPH opcode by chaining multiple Hecke actions
 * to transform spectral values through type-system flow.
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
    printf("=== Morphism Chain Example ===\n\n");
    
    /* Create VM */
    VMState *vm = vm_create();
    if (!vm) {
        fprintf(stderr, "Failed to create VM\n");
        return 1;
    }
    
    printf("Chaining Hecke morphisms to transform spectral values:\n\n");
    
    /* Initial value */
    double complex initial = 1.0 + 1.0 * I;
    printf("Initial value: ρ₀ = %.3f + %.3fi\n", creal(initial), cimag(initial));
    printf("  |ρ₀| = %.3f, arg(ρ₀) = %.3f\n\n", cabs(initial), carg(initial));
    
    vm_push(vm, initial, 5, 0.0);
    
    /* Morphism 1: Rotation by π/4 */
    printf("Morphism 1: Rotation by π/4\n");
    double complex phi1 = 1.0 * cexp(I * M_PI / 4);
    printf("  φ₁ = %.3f + %.3fi\n", creal(phi1), cimag(phi1));
    
    vm_push(vm, phi1, 3, M_PI / 4);
    op_morph(vm);
    
    SpectralFrame *frame1 = vm_peek(vm);
    printf("  Result: ρ₁ = %.3f + %.3fi\n", creal(frame1->rho), cimag(frame1->rho));
    printf("  |ρ₁| = %.3f, arg(ρ₁) = %.3f\n", cabs(frame1->rho), carg(frame1->rho));
    printf("  Guardian depth: %u\n", frame1->depth);
    printf("  Antclock: %lu\n\n", vm->antclock);
    
    /* Morphism 2: Scaling by 0.5 */
    printf("Morphism 2: Scaling by 0.5\n");
    double complex phi2 = 0.5 + 0.0 * I;
    printf("  φ₂ = %.3f + %.3fi\n", creal(phi2), cimag(phi2));
    
    vm_push(vm, phi2, 2, 0.0);
    op_morph(vm);
    
    SpectralFrame *frame2 = vm_peek(vm);
    printf("  Result: ρ₂ = %.3f + %.3fi\n", creal(frame2->rho), cimag(frame2->rho));
    printf("  |ρ₂| = %.3f, arg(ρ₂) = %.3f\n", cabs(frame2->rho), carg(frame2->rho));
    printf("  Guardian depth: %u\n", frame2->depth);
    printf("  Antclock: %lu\n\n", vm->antclock);
    
    /* Morphism 3: Rotation by -π/2 */
    printf("Morphism 3: Rotation by -π/2\n");
    double complex phi3 = 1.0 * cexp(-I * M_PI / 2);
    printf("  φ₃ = %.3f + %.3fi\n", creal(phi3), cimag(phi3));
    
    vm_push(vm, phi3, 4, -M_PI / 2);
    op_morph(vm);
    
    SpectralFrame *frame3 = vm_peek(vm);
    printf("  Result: ρ₃ = %.3f + %.3fi\n", creal(frame3->rho), cimag(frame3->rho));
    printf("  |ρ₃| = %.3f, arg(ρ₃) = %.3f\n", cabs(frame3->rho), carg(frame3->rho));
    printf("  Guardian depth: %u\n", frame3->depth);
    printf("  Antclock: %lu\n\n", vm->antclock);
    
    /* Summary */
    printf("Transformation summary:\n");
    printf("  ρ₀ → ρ₁ → ρ₂ → ρ₃\n");
    printf("  Final magnitude: %.3f\n", cabs(frame3->rho));
    printf("  Final argument: %.3f\n", carg(frame3->rho));
    printf("  Total antclock ticks: %lu\n\n", vm->antclock);
    
    /* Print final VM state */
    printf("--- Final VM State ---\n");
    print_vm_state(vm);
    
    /* Cleanup */
    vm_destroy(vm);
    
    printf("\n✓ Morphism chain example completed\n");
    return 0;
}
