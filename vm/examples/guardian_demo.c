/**
 * Guardian Demo Example
 * 
 * Demonstrates the Guardian decision logic that determines whether
 * to compose (merge) or protect (keep separate) spectral states.
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

void test_guardian_decision(VMState *vm, double complex rho, double complex phi,
                           const char *description) {
    printf("%s\n", description);
    printf("  ρ = %.3f + %.3fi\n", creal(rho), cimag(rho));
    printf("  φ = %.3f + %.3fi\n", creal(phi), cimag(phi));
    
    double dist = cabs(rho - phi);
    printf("  Distance |ρ - φ| = %.6f\n", dist);
    
    GuardianDecision decision = guardian_decide(dist, vm->k_spec);
    printf("  k_spec threshold = %.3f\n", vm->k_spec);
    
    if (decision == GUARDIAN_PROTECT) {
        printf("  ✓ Guardian decision: PROTECT (states too different)\n");
    } else {
        printf("  ✓ Guardian decision: COMPOSE (states compatible)\n");
    }
    
    /* Execute the morph to see the effect */
    vm_push(vm, rho, 5, 0.0);
    vm_push(vm, phi, 3, 0.0);
    op_morph(vm);
    
    SpectralFrame *frame = vm_peek(vm);
    printf("  Result depth: %u (", frame->depth);
    if (decision == GUARDIAN_COMPOSE) {
        printf("averaged: (5+3)/2 = 4)\n");
    } else {
        printf("protected: kept original = 5)\n");
    }
    printf("\n");
}

int main(void) {
    printf("=== Guardian Decision Demo ===\n\n");
    
    /* Create VM */
    VMState *vm = vm_create();
    if (!vm) {
        fprintf(stderr, "Failed to create VM\n");
        return 1;
    }
    
    printf("Guardian threshold k_spec = %.3f\n", vm->k_spec);
    printf("Logic: distance < k_spec → PROTECT, distance >= k_spec → COMPOSE\n\n");
    
    /* Test case 1: Very close values - should PROTECT */
    test_guardian_decision(
        vm,
        1.0 + 1.0 * I,
        1.1 + 1.0 * I,
        "Test 1: Very close values (small distance)"
    );
    
    /* Test case 2: Moderately distant values - should COMPOSE */
    test_guardian_decision(
        vm,
        1.0 + 1.0 * I,
        2.0 + 1.0 * I,
        "Test 2: Moderately distant values (medium distance)"
    );
    
    /* Test case 3: Very distant values - should COMPOSE */
    test_guardian_decision(
        vm,
        1.0 + 1.0 * I,
        10.0 + 10.0 * I,
        "Test 3: Very distant values (large distance)"
    );
    
    /* Test case 4: At threshold boundary */
    test_guardian_decision(
        vm,
        1.0 + 1.0 * I,
        1.0 + (1.0 + vm->k_spec) * I,
        "Test 4: At threshold boundary"
    );
    
    /* Test case 5: Identical values */
    test_guardian_decision(
        vm,
        3.14 + 2.71 * I,
        3.14 + 2.71 * I,
        "Test 5: Identical values (zero distance)"
    );
    
    /* Demonstrate adjusting k_spec */
    printf("=== Adjusting Guardian Threshold ===\n\n");
    
    double complex test_rho = 1.0 + 1.0 * I;
    double complex test_phi = 1.3 + 1.0 * I;
    double test_dist = cabs(test_rho - test_phi);
    
    printf("Test values: ρ = %.3f + %.3fi, φ = %.3f + %.3fi\n",
           creal(test_rho), cimag(test_rho),
           creal(test_phi), cimag(test_phi));
    printf("Distance: %.6f\n\n", test_dist);
    
    /* Try different thresholds */
    double thresholds[] = {0.2, 0.3, 0.4, 0.5, 0.6};
    for (int i = 0; i < 5; i++) {
        vm->k_spec = thresholds[i];
        GuardianDecision d = guardian_decide(test_dist, vm->k_spec);
        printf("  k_spec = %.1f → Decision: %s\n",
               vm->k_spec,
               (d == GUARDIAN_PROTECT) ? "PROTECT" : "COMPOSE");
    }
    
    printf("\n");
    
    /* Print final VM state */
    printf("--- Final VM State ---\n");
    print_vm_state(vm);
    
    printf("\nGuardian Insights:\n");
    printf("  • Low k_spec (e.g., 0.2) → More protective, preserves distinctions\n");
    printf("  • High k_spec (e.g., 0.8) → More composing, merges similar states\n");
    printf("  • Efficiency: ~14 x86 instructions, branch-predictor friendly\n");
    printf("  • Used in MORPH opcode to decide depth combination strategy\n");
    
    /* Cleanup */
    vm_destroy(vm);
    
    printf("\n✓ Guardian demo completed\n");
    return 0;
}
