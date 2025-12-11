/**
 * Test Suite for Planned Features
 * Tests the five planned features added to Zero-image μVM
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "../zero_vm.h"

int tests_run = 0;
int tests_passed = 0;

#define TEST(name) \
    printf("  Testing: %s ... ", name); \
    tests_run++;

#define ASSERT(condition) \
    if (!(condition)) { \
        printf("✗\n    Assertion failed: %s\n", #condition); \
        return; \
    }

#define PASS() \
    printf("✓\n"); \
    tests_passed++;

/**
 * Test JIT compilation hooks
 */
void test_jit_compilation(void) {
    TEST("JIT compilation hooks");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL);
    
    /* Initially, no JIT flag set */
    ASSERT(!vm_jit_should_compile(vm, OP_PROJECT));
    
    /* Enable JIT */
    vm->flags |= VM_FLAG_JIT;
    
    /* Simulate hot path by incrementing counter */
    for (int i = 0; i < JIT_HOT_PATH_THRESHOLD; i++) {
        vm->opcode_counts[0]++;
    }
    
    /* Now it should trigger */
    ASSERT(vm_jit_should_compile(vm, OP_PROJECT));
    
    /* Test compilation function (just checks it doesn't crash) */
    vm_jit_compile_hot_path(vm);
    
    vm_destroy(vm);
    PASS();
}

/**
 * Test network-transparent execution
 */
void test_network_execution(void) {
    TEST("Network-transparent execution");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL);
    
    /* Set up for remote execution */
    vm->flags |= VM_FLAG_REMOTE;
    const char *endpoint = "tcp://localhost:5555";
    
    ASSERT(vm_remote_exec(vm, endpoint));
    ASSERT(strcmp(vm->remote_endpoint, endpoint) == 0);
    
    /* Test serialization */
    vm->sp = 5;
    vm->pc = 10;
    vm->antclock = 12345;
    
    ASSERT(vm_serialize_state(vm, "/tmp/vm_state.bin"));
    
    /* Create new VM and deserialize */
    VMState *vm2 = vm_create();
    ASSERT(vm2 != NULL);
    ASSERT(vm_deserialize_state(vm2, "/tmp/vm_state.bin"));
    
    /* Verify state matches */
    ASSERT(vm2->sp == 5);
    ASSERT(vm2->pc == 10);
    ASSERT(vm2->antclock == 12345);
    
    /* Cleanup */
    unlink("/tmp/vm_state.bin");
    vm_destroy(vm);
    vm_destroy(vm2);
    PASS();
}

/**
 * Test persistent state checkpointing
 */
void test_checkpointing(void) {
    TEST("Persistent state checkpointing");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL);
    
    /* Set some state */
    vm->antclock = 99999;
    vm->pc = 7;
    vm->k_spec = 1.5;
    
    /* Push some frames - sp will be 2 after this */
    vm_push(vm, 0.5 + 14.134725*I, 3, 1.57);
    vm_push(vm, 0.5 + 21.022040*I, 5, 3.14);
    
    uint8_t expected_sp = vm->sp;  /* Should be 2 */
    
    /* Save checkpoint */
    const char *ckpt_file = "/tmp/vm_checkpoint.ckpt";
    ASSERT(vm_checkpoint_save(vm, ckpt_file));
    
    /* Create new VM and load checkpoint */
    VMState *vm2 = vm_create();
    ASSERT(vm2 != NULL);
    ASSERT(vm_checkpoint_load(vm2, ckpt_file));
    
    /* Verify state restored correctly */
    ASSERT(vm2->sp == expected_sp);
    ASSERT(vm2->antclock == 99999);
    ASSERT(vm2->pc == 7);
    ASSERT(vm2->k_spec == 1.5);
    
    /* Cleanup */
    unlink(ckpt_file);
    vm_destroy(vm);
    vm_destroy(vm2);
    PASS();
}

/**
 * Test interactive debugger
 */
void test_debugger(void) {
    TEST("Interactive debugger");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL);
    
    /* Enable debug mode */
    vm->flags |= VM_FLAG_DEBUG;
    
    /* Add breakpoints */
    vm_debug_add_breakpoint(vm, 5);
    vm_debug_add_breakpoint(vm, 10);
    vm_debug_add_breakpoint(vm, 15);
    
    ASSERT(vm->breakpoint_count == 3);
    ASSERT(vm_debug_is_breakpoint(vm, 5));
    ASSERT(vm_debug_is_breakpoint(vm, 10));
    ASSERT(vm_debug_is_breakpoint(vm, 15));
    ASSERT(!vm_debug_is_breakpoint(vm, 7));
    
    /* Remove a breakpoint */
    vm_debug_remove_breakpoint(vm, 10);
    ASSERT(vm->breakpoint_count == 2);
    ASSERT(!vm_debug_is_breakpoint(vm, 10));
    ASSERT(vm_debug_is_breakpoint(vm, 5));
    ASSERT(vm_debug_is_breakpoint(vm, 15));
    
    vm_destroy(vm);
    PASS();
}

/**
 * Test visual tracer
 */
void test_visual_tracer(void) {
    TEST("Visual tracer");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL);
    
    /* Enable tracing */
    vm_trace_enable(vm, true);
    ASSERT(vm->trace_enabled);
    ASSERT(vm->flags & VM_FLAG_TRACE);
    
    /* Push some frames and record */
    for (int i = 0; i < 10; i++) {
        vm->antclock = 1000 + i;  /* Set non-zero timestamp */
        double imag = 14.134725 + i * 7.0;
        vm_push(vm, 0.5 + imag*I, i, 1.57);
        vm_trace_record(vm);
    }
    
    /* Export to CSV */
    const char *csv_file = "/tmp/trace.csv";
    ASSERT(vm_trace_export(vm, csv_file, "csv"));
    
    /* Verify file was created */
    FILE *f = fopen(csv_file, "r");
    ASSERT(f != NULL);
    
    /* Check header */
    char line[256];
    fgets(line, sizeof(line), f);
    ASSERT(strstr(line, "rho_real") != NULL);
    ASSERT(strstr(line, "rho_imag") != NULL);
    fclose(f);
    
    /* Export to JSON */
    const char *json_file = "/tmp/trace.json";
    ASSERT(vm_trace_export(vm, json_file, "json"));
    
    /* Verify JSON file - read entire content to find "traces" */
    f = fopen(json_file, "r");
    ASSERT(f != NULL);
    bool found_traces = false;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "traces") != NULL) {
            found_traces = true;
            break;
        }
    }
    ASSERT(found_traces);
    fclose(f);
    
    /* Disable tracing */
    vm_trace_enable(vm, false);
    ASSERT(!vm->trace_enabled);
    ASSERT(!(vm->flags & VM_FLAG_TRACE));
    
    /* Cleanup */
    unlink(csv_file);
    unlink(json_file);
    vm_destroy(vm);
    PASS();
}

/**
 * Test integration - all features work together
 */
void test_integration(void) {
    TEST("Feature integration");
    
    VMState *vm = vm_create();
    ASSERT(vm != NULL);
    
    /* Enable all features */
    vm->flags = VM_FLAG_DEBUG | VM_FLAG_TRACE | VM_FLAG_JIT | VM_FLAG_REMOTE;
    vm_trace_enable(vm, true);
    
    /* Add a breakpoint */
    vm_debug_add_breakpoint(vm, 0);
    
    /* Configure remote endpoint */
    vm_remote_exec(vm, "tcp://localhost:8080");
    
    /* Push some data and trace */
    vm_push(vm, 0.5 + 14.134725*I, 3, 1.57);
    vm_trace_record(vm);
    
    /* Increment JIT counter */
    vm->opcode_counts[0] = JIT_HOT_PATH_THRESHOLD;
    
    /* Save checkpoint with all features active */
    const char *ckpt = "/tmp/integration_test.ckpt";
    ASSERT(vm_checkpoint_save(vm, ckpt));
    
    /* Load into new VM */
    VMState *vm2 = vm_create();
    ASSERT(vm_checkpoint_load(vm2, ckpt));
    
    /* Verify flags preserved */
    ASSERT(vm2->flags == vm->flags);
    
    /* Cleanup */
    unlink(ckpt);
    vm_destroy(vm);
    vm_destroy(vm2);
    PASS();
}

int main(void) {
    printf("\n=== Planned Features Test Suite ===\n\n");
    
    test_jit_compilation();
    test_network_execution();
    test_checkpointing();
    test_debugger();
    test_visual_tracer();
    test_integration();
    
    printf("\n=== Test Summary ===\n");
    printf("Tests run: %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_run - tests_passed);
    
    if (tests_passed == tests_run) {
        printf("\n✓ All planned features tests passed!\n\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed\n\n");
        return 1;
    }
}
