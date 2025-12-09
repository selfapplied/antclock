/**
 * Zero-image μVM - Main Entry Point
 * 
 * Simple CLI for running Zero-image programs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zero_vm.h"

void print_usage(const char *program_name) {
    printf("Zero-image μVM v0.1.0\n");
    printf("Usage: %s [options] <program.zero>\n\n", program_name);
    printf("Options:\n");
    printf("  -h, --help     Show this help message\n");
    printf("  -v, --verbose  Enable verbose output\n");
    printf("  -k <value>     Set Guardian k_spec threshold (default: 0.5)\n");
    printf("\n");
}

int main(int argc, char *argv[]) {
    /* Parse command line arguments */
    bool verbose = false;
    double k_spec = K_SPEC_DEFAULT;
    const char *program_file = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-k") == 0) {
            if (i + 1 < argc) {
                k_spec = atof(argv[++i]);
            } else {
                fprintf(stderr, "Error: -k requires a value\n");
                return 1;
            }
        } else if (argv[i][0] != '-') {
            program_file = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    /* Check if program file was provided */
    if (!program_file) {
        fprintf(stderr, "Error: No program file specified\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    /* Create VM */
    if (verbose) {
        printf("Creating VM...\n");
    }
    
    VMState *vm = vm_create();
    if (!vm) {
        fprintf(stderr, "Failed to create VM\n");
        return 1;
    }
    
    /* Set Guardian threshold if specified */
    vm->k_spec = k_spec;
    
    if (verbose) {
        printf("Guardian k_spec: %.3f\n", vm->k_spec);
    }
    
    /* Load program */
    if (verbose) {
        printf("Loading program: %s\n", program_file);
    }
    
    if (!vm_load_program(vm, program_file)) {
        fprintf(stderr, "Failed to load program\n");
        vm_destroy(vm);
        return 1;
    }
    
    /* Run program */
    if (verbose) {
        printf("\n=== Executing Program ===\n\n");
    }
    
    vm_run(vm);
    
    /* Cleanup */
    if (verbose) {
        printf("\n=== Cleanup ===\n");
    }
    
    vm_destroy(vm);
    
    if (verbose) {
        printf("VM destroyed\n");
    }
    
    return 0;
}
