#!/usr/bin/env python3
"""
Test the new benchmark implementations.
"""

import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_pcfg():
    """Test PCFG benchmark."""
    print("Testing PCFG...")
    from benchmarks.pcfg import run_pcfg_baseline
    results = run_pcfg_baseline(num_epochs=5)
    print(f"PCFG Results: {results}")
    return results

def test_cfq():
    """Test CFQ benchmark."""
    print("Testing CFQ...")
    from benchmarks.cfq import run_cfq_baseline
    results = run_cfq_baseline(num_epochs=5)
    print(f"CFQ Results: {results}")
    return results

def test_rpm():
    """Test RPM benchmark."""
    print("Testing RPM...")
    from benchmarks.rpm import run_rpm_baseline
    results = run_rpm_baseline(num_epochs=5)
    print(f"RPM Results: {results}")
    return results

def test_math():
    """Test Math Reasoning benchmark."""
    print("Testing Math Reasoning...")
    from benchmarks.math_reasoning import run_math_baseline
    results = run_math_baseline(num_epochs=5)
    print(f"Math Results: {results}")
    return results

def test_ce_pcfg():
    """Test CE-enhanced PCFG."""
    print("Testing CE-PCFG...")
    from benchmarks.ce_pcfg import run_ce_pcfg_experiment
    results = run_ce_pcfg_experiment(num_epochs=5)
    print(f"CE-PCFG Results: {results}")
    return results

if __name__ == "__main__":
    print("🧪 Testing New Benchmarks\n")

    # Test individual benchmarks
    test_pcfg()
    print()
    test_cfq()
    print()
    test_rpm()
    print()
    test_math()
    print()
    test_ce_pcfg()

    print("\n✅ All tests completed!")
