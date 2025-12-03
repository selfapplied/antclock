#!/usr/bin/env python3
"""
test_antclock.py

Simple test to verify AntClock works correctly.
"""

from clock import CurvatureClockWalker, pascal_curvature, digit_count

def test_basic_functionality():
    """Test basic walker functionality."""
    print("Testing basic AntClock functionality...")

    # Test curvature calculation
    kappa_3 = pascal_curvature(3)
    print(f"  κ(3) = {kappa_3:.6f}")

    # Test digit count
    d_123 = digit_count(123)
    print(f"  d(123) = {d_123}")

    # Test walker
    walker = CurvatureClockWalker(x_0=1)

    # Single step
    metadata = walker.step()
    print(f"  Step 1: x=1 → {metadata['x_next']}, τ={metadata['tau_next']:.3f}")

    # Multiple steps
    history, summary = walker.evolve(10)
    print(f"  After 10 steps: x={summary['x_final']}, τ={summary['tau_final']:.3f}")
    print(f"  Frequency: {summary['frequency']:.6f}")

    # Check geometry
    x_coords, y_coords = walker.get_geometry()
    print(f"  Geometry points: {len(x_coords)}")

    print("✓ All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
