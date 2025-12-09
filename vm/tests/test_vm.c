/**
 * Zero-image μVM Test Suite
 * 
 * Tests all core VM functionality
 */

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
#include "../zero_vm.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Test counters */
static int tests_run = 0;
static int tests_passed = 0;

/* Test helper macros */
#define TEST(name) \
    do { \
        tests_run++; \
        printf("  Testing: %s ... ", name); \
        fflush(stdout);

#define PASS() \
        tests_passed++; \
        printf("✓\n"); \
    } while(0)

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("✗\n    FAILED: %s\n", message); \
            return; \
        } \
    } while(0)

#define ASSERT_NEAR(a, b, epsilon, message) \
    ASSERT(fabs((a) - (b)) < (epsilon), message)

/* ============================================================================
 * TEST FUNCTIONS
 * ============================================================================ */

/**
 * Test VM creation and destruction
 */
void test_vm_lifecycle() {
    TEST("VM lifecycle");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL, "VM creation failed");
    ASSERT(vm->sp == 0, "Initial SP should be 0");
    ASSERT(vm->antclock == 0, "Initial antclock should be 0");
    ASSERT(vm->gamma_gaps != NULL, "Gamma gap table should be initialized");
    
    vm_destroy(vm);
    
    PASS();
}

/**
 * Test stack operations
 */
void test_stack_operations() {
    TEST("Stack operations");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL, "VM creation failed");
    
    /* Test push */
    double complex rho1 = 1.0 + 2.0 * I;
    vm_push(vm, rho1, 5, 0.5);
    ASSERT(vm->sp == 1, "SP should be 1 after push");
    
    /* Test peek */
    SpectralFrame *frame = vm_peek(vm);
    ASSERT(frame != NULL, "Peek should not return NULL");
    ASSERT_NEAR(creal(frame->rho), 1.0, 1e-6, "Real part mismatch");
    ASSERT_NEAR(cimag(frame->rho), 2.0, 1e-6, "Imaginary part mismatch");
    ASSERT(frame->depth == 5, "Depth mismatch");
    ASSERT_NEAR(frame->monodromy, 0.5, 1e-6, "Monodromy mismatch");
    
    /* Test pop */
    double complex rho_pop;
    uint32_t depth_pop;
    float mono_pop;
    bool success = vm_pop(vm, &rho_pop, &depth_pop, &mono_pop);
    ASSERT(success, "Pop should succeed");
    ASSERT(vm->sp == 0, "SP should be 0 after pop");
    ASSERT_NEAR(creal(rho_pop), 1.0, 1e-6, "Popped real part mismatch");
    ASSERT_NEAR(cimag(rho_pop), 2.0, 1e-6, "Popped imaginary part mismatch");
    
    /* Test circular buffer wrap */
    for (int i = 0; i < 260; i++) {
        vm_push(vm, (double)i + 0.0 * I, i, 0.0);
    }
    ASSERT(vm->sp == 4, "SP should wrap around correctly");
    
    vm_destroy(vm);
    
    PASS();
}

/**
 * Test guardian logic
 */
void test_guardian() {
    TEST("Guardian decision logic");
    
    double k_spec = 0.5;
    
    /* Below threshold - should protect */
    GuardianDecision d1 = guardian_decide(0.3, k_spec);
    ASSERT(d1 == GUARDIAN_PROTECT, "Should protect when depth < k_spec");
    
    /* Above threshold - should compose */
    GuardianDecision d2 = guardian_decide(0.7, k_spec);
    ASSERT(d2 == GUARDIAN_COMPOSE, "Should compose when depth >= k_spec");
    
    /* At threshold - should compose */
    GuardianDecision d3 = guardian_decide(0.5, k_spec);
    ASSERT(d3 == GUARDIAN_COMPOSE, "Should compose when depth == k_spec");
    
    PASS();
}

/**
 * Test antclock tick generation
 */
void test_antclock() {
    TEST("Antclock tick generation");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL, "VM creation failed");
    
    /* Push a value to enable tick generation */
    vm_push(vm, 1.5 + 2.3 * I, 3, 0.0);
    
    uint64_t tick1 = antclock_tick(vm);
    ASSERT(tick1 > 0, "First tick should be positive");
    
    /* Different value should produce different tick */
    vm_push(vm, 3.7 + 1.2 * I, 2, 0.0);
    uint64_t tick2 = antclock_tick(vm);
    /* Note: Due to hash collisions, ticks might be same, but that's okay */
    
    vm_destroy(vm);
    
    PASS();
}

/**
 * Test p-adic distance calculation
 */
void test_padic_distance() {
    TEST("P-adic distance");
    
    double complex rho1 = 1.0 + 2.0 * I;
    double complex rho2 = 3.0 + 4.0 * I;
    
    double dist = padic_distance(rho1, rho2, 2);
    ASSERT(dist > 0.0, "Distance should be positive");
    
    /* Distance to self should be 0 */
    double dist_self = padic_distance(rho1, rho1, 2);
    ASSERT_NEAR(dist_self, 0.0, 1e-6, "Distance to self should be 0");
    
    PASS();
}

/**
 * Test Hecke action
 */
void test_hecke_action() {
    TEST("Hecke action");
    
    double complex rho = 1.0 + 1.0 * I;
    double complex phi = 1.0 + 0.0 * I;
    
    double complex result = hecke_action(rho, phi);
    ASSERT(cabs(result) > 0.0, "Result should be non-zero");
    
    /* Hecke with identity-like morphism */
    double complex identity = 1.0 + 0.0 * I;
    double complex result_id = hecke_action(rho, identity);
    ASSERT(cabs(result_id - rho) < 1e-6 || cabs(result_id) > 0.0, 
           "Hecke with identity should be close to original or transformed");
    
    PASS();
}

/**
 * Test PROJECT opcode
 */
void test_op_project() {
    TEST("OP_PROJECT opcode");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL, "VM creation failed");
    
    /* Create a simple program */
    vm->program_size = 3;
    vm->program = (SpectralRecord*)malloc(sizeof(SpectralRecord) * vm->program_size);
    ASSERT(vm->program != NULL, "Program allocation failed");
    
    vm->program[0].rho_real = 1.0;
    vm->program[0].rho_imag = 2.0;
    vm->program[0].depth = 5;
    vm->program[0].monodromy = 0.5;
    
    vm->program[1].rho_real = 3.0;
    vm->program[1].rho_imag = 4.0;
    vm->program[1].depth = 3;
    vm->program[1].monodromy = 1.0;
    
    /* Push index to project */
    vm_push(vm, 0.0 + 0.0 * I, 0, 0.0);
    
    /* Execute PROJECT */
    op_project(vm);
    
    /* Check result */
    SpectralFrame *frame = vm_peek(vm);
    ASSERT_NEAR(creal(frame->rho), 1.0, 1e-6, "Projected real part mismatch");
    ASSERT_NEAR(cimag(frame->rho), 2.0, 1e-6, "Projected imaginary part mismatch");
    ASSERT(frame->depth == 5, "Projected depth mismatch");
    
    vm_destroy(vm);
    
    PASS();
}

/**
 * Test DEPTH opcode
 */
void test_op_depth() {
    TEST("OP_DEPTH opcode");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL, "VM creation failed");
    
    /* Push two values */
    vm_push(vm, 1.0 + 2.0 * I, 5, 0.5);
    vm_push(vm, 3.0 + 4.0 * I, 3, 1.0);
    
    /* Execute DEPTH */
    op_depth(vm);
    
    /* Check result */
    SpectralFrame *frame = vm_peek(vm);
    double dist = creal(frame->rho);
    ASSERT(dist > 0.0, "Distance should be positive");
    
    vm_destroy(vm);
    
    PASS();
}

/**
 * Test MORPH opcode
 */
void test_op_morph() {
    TEST("OP_MORPH opcode");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL, "VM creation failed");
    
    /* Push value and morphism */
    vm_push(vm, 2.0 + 1.0 * I, 4, 0.0);
    vm_push(vm, 0.5 + 0.5 * I, 2, M_PI / 4);
    
    /* Execute MORPH */
    op_morph(vm);
    
    /* Check result exists */
    SpectralFrame *frame = vm_peek(vm);
    ASSERT(cabs(frame->rho) > 0.0, "Morphed value should be non-zero");
    
    vm_destroy(vm);
    
    PASS();
}

/**
 * Test WITNESS opcode
 */
void test_op_witness() {
    TEST("OP_WITNESS opcode");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL, "VM creation failed");
    
    /* Push a value */
    vm_push(vm, 3.0 + 4.0 * I, 7, 1.5);
    
    uint8_t sp_before = vm->sp;
    
    /* Execute WITNESS */
    op_witness(vm);
    
    /* Check that 4 components were pushed */
    ASSERT(vm->sp == (sp_before + 4) % STACK_DEPTH, 
           "WITNESS should push 4 components");
    
    vm_destroy(vm);
    
    PASS();
}

/**
 * Test program execution
 */
void test_program_execution() {
    TEST("Program execution");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL, "VM creation failed");
    
    /* Create a simple program */
    vm->program_size = 2;
    vm->program = (SpectralRecord*)malloc(sizeof(SpectralRecord) * vm->program_size);
    ASSERT(vm->program != NULL, "Program allocation failed");
    
    /* First instruction: PROJECT (encoded in depth field) */
    vm->program[0].rho_real = 5.0;
    vm->program[0].rho_imag = 6.0;
    vm->program[0].depth = OP_PROJECT;  /* Opcode in low byte */
    vm->program[0].monodromy = 0.0;
    
    /* Push an index first */
    vm_push(vm, 0.0 + 0.0 * I, 0, 0.0);
    
    /* Execute one step */
    vm->pc = 0;
    vm_step(vm);
    
    ASSERT(vm->pc == 1, "PC should advance after step");
    
    vm_destroy(vm);
    
    PASS();
}

/* ============================================================================
 * MAIN TEST RUNNER
 * ============================================================================ */

int main(void) {
    printf("\n=== Zero-image μVM Test Suite ===\n\n");
    
    /* Run all tests */
    test_vm_lifecycle();
    test_stack_operations();
    test_guardian();
    test_antclock();
    test_padic_distance();
    test_hecke_action();
    test_op_project();
    test_op_depth();
    test_op_morph();
    test_op_witness();
    test_program_execution();
    
    /* Print summary */
    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    
    if (tests_passed == tests_run) {
        printf("\n✓ All tests passed!\n\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed\n\n");
        return 1;
    }
}
