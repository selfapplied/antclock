# Why Volte Might Be Unavailable

## Root Cause

The issue is that `antclock/__init__.py` imports all modules including `learner.py`, which requires `torch`. When you do:
```python
from antclock.volte import DiscreteVolteSystem
```

Python first loads `antclock/__init__.py`, which tries to import `learner`, which fails if `torch` isn't installed.

## Solution Implemented

We now import `volte.py` directly using `importlib`, bypassing `__init__.py`:
- ✅ Works even if `torch` isn't installed
- ✅ Only requires `numpy` (which is in requirements.txt)
- ✅ Avoids triggering unnecessary imports

## Current Status

✅ **Volte is available when using `run.sh`** - The direct import works correctly.

## When Volte Import Might Still Fail

The fallback mechanism exists for these scenarios:

### 1. **Running Without `run.sh`**
If the script is run directly without `run.sh`:
```bash
python3 biology/erv/analyze_erv.py sequences.fasta
```

**Potential issues**:
- Virtual environment not activated
- Dependencies (numpy, scipy) not installed
- `antclock` package not in Python path

### 2. **Missing Dependencies**
`antclock.volte` requires:
- `numpy` - For numerical operations
- Standard library only (typing, math)

If numpy isn't installed, the import will fail.

### 3. **Path Issues**
The import uses:
```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from antclock.volte import DiscreteVolteSystem
```

This assumes the project structure is:
```
antclock/
  antclock/
    volte.py
  biology/
    erv/
      analyze_erv.py
```

If the script is run from a different location or the structure changes, the path might be wrong.

### 4. **Sandboxed Environments**
Some environments (like certain CI/CD systems) might restrict imports or have limited package access.

## Solution: Use `run.sh`

The recommended approach is to use `run.sh`, which:
- ✅ Activates virtual environment
- ✅ Installs dependencies from `requirements.txt`
- ✅ Sets up proper Python paths
- ✅ Handles environment configuration

```bash
./run.sh biology/erv/analyze_erv.py sequences.fasta
```

## Fallback Behavior

The code gracefully handles Volte unavailability:
- Uses standalone `ERVVolteSystem` implementation
- All functionality still works
- Just doesn't use the unified `DiscreteVolteSystem` class

This ensures the biology module works even if:
- Volte isn't available
- Dependencies are missing
- Running in restricted environments

## Recommendation

For production use, always use `run.sh`:
```bash
./run.sh biology/erv/analyze_erv.py data/sequences.fasta
```

This guarantees:
- ✅ All dependencies available
- ✅ Proper environment setup
- ✅ Volte system integrated
- ✅ Consistent behavior

