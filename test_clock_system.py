#!/usr/bin/env python3
"""
test_clock_system.py - Verify the AntClock modular system is working
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("🔬 Testing AntClock Modular System...")
print("=" * 50)

def test_imports():
    """Test that all modules can be imported"""
    print("\n1. Testing imports...")
    try:
        from clock import AntClock, pascal_curvature
        print("   ✓ Main clock module imported")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic AntClock functionality"""
    print("\n2. Testing basic functionality...")
    try:
        from clock import AntClock

        # Create clock instance
        clock = AntClock()
        print("   ✓ AntClock instance created")

        # Test pascal analysis
        result = clock.pascal_analysis(3)
        expected_keys = ['row', 'radius', 'curvature', 'shell_digit']
        if all(key in result for key in expected_keys):
            print(f"   ✓ Pascal analysis works: row={result['row']}, curvature={result['curvature']:.3f}")
        else:
            print(f"   ❌ Pascal analysis missing keys: {result.keys()}")
            return False

        # Test digit analysis
        digit_result = clock.digit_shell_analysis(100.0)
        expected_digit_keys = ['digit_count', 'boundary_curvature', 'clock_rate']
        if all(key in digit_result for key in expected_digit_keys):
            print(f"   ✓ Digit analysis works: digits={digit_result['digit_count']}, rate={digit_result['clock_rate']:.3f}")
        else:
            print(f"   ❌ Digit analysis missing keys: {digit_result.keys()}")
            return False

        return True

    except Exception as e:
        print(f"   ❌ Basic functionality failed: {e}")
        return False

def test_direct_functions():
    """Test direct function imports"""
    print("\n3. Testing direct function imports...")
    try:
        from clock import pascal_curvature, digit_count, clock_rate

        # Test individual functions
        kappa = pascal_curvature(3)
        digits = digit_count(1000.0)
        rate = clock_rate(100.0)

        print(f"   ✓ pascal_curvature(3) = {kappa}")
        print(f"   ✓ digit_count(1000.0) = {digits}")
        print(f"   ✓ clock_rate(100.0) = {rate}")

        return True

    except Exception as e:
        print(f"   ❌ Direct function test failed: {e}")
        return False

def test_mathematical_correctness():
    """Test basic mathematical correctness"""
    print("\n4. Testing mathematical correctness...")
    try:
        from clock import pascal_curvature, pascal_radius

        # Test that curvature is derivative of radius
        # κ_n = r_{n+1} - 2r_n + r_{n-1}
        n = 3
        kappa_direct = pascal_curvature(n)
        kappa_calculated = pascal_radius(n+1) - 2*pascal_radius(n) + pascal_radius(n-1)

        if abs(kappa_direct - kappa_calculated) < 1e-10:
            print(f"   ✓ Curvature recurrence relation holds for n={n}")
        else:
            print(f"   ❌ Curvature recurrence failed: direct={kappa_direct}, calculated={kappa_calculated}")
            return False

        # Test that radius is positive for reasonable n
        if pascal_radius(3) > 0:
            print("   ✓ Pascal radius is positive")
        else:
            print("   ❌ Pascal radius is not positive")
            return False

        return True

    except Exception as e:
        print(f"   ❌ Mathematical correctness test failed: {e}")
        return False

def main():
    """Run all tests"""
    tests = [
        test_imports,
        test_basic_functionality,
        test_direct_functions,
        test_mathematical_correctness
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 SUCCESS: AntClock modular system is fully operational!")
        print("\nThe reorganization from 4235-line monolithic file to")
        print("modular Python package has been completed successfully!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
