#!/usr/bin/env python3
"""
clock.test_clock_module - Test the complete clock module functionality.

Tests that the modular clock system works correctly as a unified package.
"""

import sys
import os

# Add parent directory to path to test the module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_clock_module_imports():
    """Test that all components can be imported from the clock module."""
    print("Testing clock module imports...")

    try:
        # Test main interface
        from clock import AntClock, quick_trajectory
        print("  ✓ Main interface imported")

        # Test all submodules
        from clock.pascal_core import pascal_curvature, digit_count
        from clock.homology_engine import DigitHomologyComplex
        from clock.clock_mechanics import CurvatureClockWalker
        from clock.analysis_framework import ChaosAnalysisFramework
        print("  ✓ All submodules imported")

        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_clock_module_functionality():
    """Test that the clock module provides expected functionality."""
    print("Testing clock module functionality...")

    try:
        from clock import AntClock

        # Create clock instance
        clock = AntClock()

        # Test basic operations
        pascal_result = clock.pascal_analysis(3)
        assert 'radius' in pascal_result
        assert 'curvature' in pascal_result

        digit_result = clock.digit_shell_analysis(100.0)
        assert 'digit_count' in digit_result
        assert 'clock_rate' in digit_result

        # Test trajectory generation
        trajectory = clock.run_trajectory(steps=10)
        assert 'history' in trajectory
        assert 'analysis' in trajectory
        assert len(trajectory['history']) > 0

        print("  ✓ All functionality tests passed")
        return True

    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that key functions still work as expected."""
    print("Testing backward compatibility...")

    try:
        from clock.pascal_core import pascal_curvature, digit_count
        from clock.clock_mechanics import CurvatureClockWalker

        # Test core functions match expected values
        kappa_3 = pascal_curvature(3)
        assert abs(kappa_3 - (-0.4054651081081644)) < 1e-10  # Known value

        d_100 = digit_count(100.0)
        assert d_100 == 3

        # Test walker creation
        walker = CurvatureClockWalker(x_0=1)
        history, summary = walker.evolve(5)
        assert len(history) == 6  # Initial + 5 steps
        assert walker.x > 1  # Should have moved

        print("  ✓ Backward compatibility maintained")
        return True

    except Exception as e:
        print(f"  ❌ Compatibility test failed: {e}")
        return False

def test_module_structure():
    """Test that the module has proper structure."""
    print("Testing module structure...")

    import clock

    # Check version
    assert hasattr(clock, '__version__')
    assert hasattr(clock, '__author__')

    # Check main exports
    assert hasattr(clock, 'AntClock')
    assert hasattr(clock, 'quick_trajectory')

    print("  ✓ Module structure is correct")
    return True

def run_all_tests():
    """Run all clock module tests."""
    print("🕰️  Testing AntClock Module")
    print("=" * 40)

    tests = [
        test_clock_module_imports,
        test_clock_module_functionality,
        test_backward_compatibility,
        test_module_structure
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 Clock module test suite passed! The modular system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the module implementation.")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
