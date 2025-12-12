# run.sh Tests

Test suite for the `run.sh` script's identity recognition and fallback logic.

## Test Files

### `test_fallback.py`
- **Purpose**: Tests normal execution with a valid shebang
- **Shebang**: `#!/usr/bin/env python3`
- **Expected**: Should execute normally without triggering fallback logic

### `test_broken_shebang.py`
- **Purpose**: Tests fallback logic when interpreter fails
- **Shebang**: `#!/usr/bin/nonexistent_interpreter` (invalid)
- **Expected**: Should trigger fallback, replace shebang with `#!/run.sh`, and execute successfully

### `test_already_run_sh.py`
- **Purpose**: Tests that existing `#!/run.sh` shebangs are preserved
- **Shebang**: `#!/run.sh`
- **Expected**: Should be recognized as Direct Self identity and executed without modification

## Running Tests

```bash
# Test individual scripts
./run.sh tests/run.sh/test_fallback.py
./run.sh tests/run.sh/test_broken_shebang.py
./run.sh tests/run.sh/test_already_run_sh.py

# Or run all tests
for test in tests/run.sh/*.py; do
    echo "Running $test..."
    ./run.sh "$test"
    echo "---"
done
```

## Expected Behaviors

1. **Normal execution**: Scripts with valid interpreters execute directly
2. **Fallback on failure**: Scripts with broken shebangs get their shebang replaced with `#!/run.sh`
3. **Preservation**: Scripts already using `#!/run.sh` are not modified
4. **Self-awareness**: No infinite recursion occurs due to identity recognition

