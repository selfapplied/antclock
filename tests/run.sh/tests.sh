#!/bin/bash
# Test runner for run.sh functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

cd "$PROJECT_ROOT"

echo "Running run.sh tests..."
echo "======================"

FAILED_TESTS=()

for test_file in "$SCRIPT_DIR"/*.py; do
    test_name=$(basename "$test_file" .py)
    echo ""
    echo "Running test: $test_name"
    echo "------------------------"

    if ./run.sh "$test_file"; then
        echo "✅ $test_name: PASSED"
    else
        echo "❌ $test_name: FAILED"
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
