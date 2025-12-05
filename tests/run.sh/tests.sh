#!/bin/bash
# Test runner for run.sh functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

cd "$PROJECT_ROOT"

echo "Running run.sh tests..."
echo "======================"

FAILED_TESTS=()

# Define expected behaviors for each test
get_expected_behavior() {
    case "$1" in
        "test_already_run_sh") echo "direct-self" ;;
        "test_broken_shebang") echo "foreign-fallback" ;;
        "test_fallback") echo "foreign-fallback" ;;
        "test_simple_import") echo "proto-self-upgrade" ;;
        *) echo "" ;;
    esac
}

for test_file in "$SCRIPT_DIR"/*.py; do
    test_name=$(basename "$test_file" .py)
    expected_behavior=$(get_expected_behavior "$test_name")

    echo ""
    echo "Running test: $test_name"
    echo "------------------------"
    echo "Expected behavior: $expected_behavior"

    if [ -n "$expected_behavior" ]; then
        # Use dry-run to test expected behavior without side effects
        if ./run.sh "--dry-run=$expected_behavior" "$test_file" >/dev/null 2>&1; then
            echo "✅ $test_name: PASSED (dry-run validated expected behavior)"
        else
            echo "❌ $test_name: FAILED (dry-run detected unexpected behavior)"
            FAILED_TESTS+=("$test_name")
        fi
    else
        echo "⚠️  $test_name: No expected behavior defined, skipping"
        FAILED_TESTS+=("$test_name")
    fi
done

echo ""
echo "======================"

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "🎉 All tests PASSED!"
    exit 0
else
    echo "💥 Failed tests: ${FAILED_TESTS[*]}"
    exit 1
fi
