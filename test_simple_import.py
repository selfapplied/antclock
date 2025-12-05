#!run.sh
"""
Simple test to verify PYTHONPATH works
"""
try:
    from antclock.clock import compute_enhanced_betti_numbers
    print("SUCCESS: antclock import worked!")
except ImportError as e:
    print(f"FAILED: {e}")
