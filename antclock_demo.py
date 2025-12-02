#!/usr/bin/env python3
"""
antclock_demo.py

AntClock: Pascal Curvature Clock Walker Demo.

The smallest nontrivial creature that actually moves using
Pascal clock machinery.

Author: Joel
"""

import numpy as np
import matplotlib.pyplot as plt
from pascal_clock import CurvatureClockWalker

def main():
    """Run the AntClock demo."""
    print("=" * 70)
    print("ANTCLOCK: PASCAL CURVATURE CLOCK WALKER")
    print("=" * 70)
    print("\nThe Tiny Machine: A complete dynamic system that moves using")
    print("Pascal curvature clock machinery.")
    print("\nState: x_t (body), τ_t (clock phase), φ_t (angle)")
    print("Dynamics: Self-clocked, curvature-aware, digit-sensitive")

    # Create walker
    walker = CurvatureClockWalker(
        x_0=1,
        tau_0=0.0,
        phi_0=0.0,
        chi_feg=0.638  # Feigenbaum scaling factor
    )

    # Evolve for 100 steps
    print("\n" + "-" * 50)
    print("EVOLVING WALKER FOR 100 STEPS")
    print("-" * 50)

    history, summary = walker.evolve(100)

    print("
Summary:")
    print(f"  Final x: {summary['x_final']}")
    print(f"  Final τ: {summary['tau_final']:.3f}")
    print(f"  Final φ: {summary['phi_final']:.3f} rad ({summary['phi_final']*180/np.pi:.1f}°)")
    print(f"  Frequency (avg R): {summary['frequency']:.6f}")
    print(f"  R range: [{summary['R_min']:.6f}, {summary['R_max']:.6f}]")
    print(f"  Boundary crossings: {summary['boundary_count']}")
    print(f"  x range: {summary['x_range']}")

    # Show first few steps
    print("
First 10 steps:")
    for h in history[:10]:
        print(f"  t={h['t']:3d}: x={h['x']:4d} → {h['x_next']:4d}, "
              f"τ={h['tau']:6.3f}, φ={h['phi']:6.3f}, R={h['R']:.6f}, "
              f"d={h['d']}, {'BOUNDARY' if h['boundary_crossed'] else ''}")

    # Show boundary events
    if walker.boundary_events:
        print(f"\nBoundary Events ({len(walker.boundary_events)}):")
        for event in walker.boundary_events[:5]:  # Show first 5
            print(f"  t={event['t']:3d}: x={event['x_old']:4d} → {event['x_new']:4d}, "
                  f"d={event['d_old']}→{event['d_new']}, "
                  f"K={event['K_old']:.4f}→{event['K_new']:.4f}, "
                  f"jump={event['jump']}")

    # Generate plots
    print("\n" + "-" * 50)
    print("GENERATING PLOTS")
    print("-" * 50)

    plot_trajectory(walker, history)
    plot_geometry(walker, history)

    print("✓ Saved: antclock_trajectory.png")
    print("✓ Saved: antclock_geometry.png")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\n✓ The walker demonstrates:")
    print("  - Smooth flow inside digit-shells")
    print("  - Phase transitions at digit-boundaries")
    print("  - Frequency from average clock rate")
    print("  - Geometry as arcs of rotation with phase-rate shifts")
    print("\n✓ This is the smallest nontrivial creature that moves using")
    print("  all the Pascal clock machinery: combinatorics, symbolic")
    print("  patterns, chaos stiffness, and conservation framing.")

def plot_trajectory(walker, history):
    """Plot walker trajectory."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    t_seq = [h['t'] for h in history]
    x_seq = [h['x'] for h in history]
    tau_seq = [h['tau'] for h in history]
    R_seq = [h['R'] for h in history]

    # Plot 1: Body x(t)
    axes[0].plot(t_seq, x_seq, 'b-', linewidth=2, marker='o', markersize=3, label='x(t)')
    # Mark boundary crossings
    for event in walker.boundary_events:
        axes[0].axvline(x=event['t'], color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Time step t')
    axes[0].set_ylabel('Body x(t)')
    axes[0].set_title('Body Evolution (red dashes = digit boundaries)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Clock phase τ(t)
    axes[1].plot(t_seq, tau_seq, 'g-', linewidth=2, marker='s', markersize=3, label='τ(t)')
    for event in walker.boundary_events:
        axes[1].axvline(x=event['t'], color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time step t')
    axes[1].set_ylabel('Clock Phase τ(t)')
    axes[1].set_title('Internal Clock Phase')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Clock rate R(t)
    axes[2].plot(t_seq, R_seq, 'm-', linewidth=2, marker='^', markersize=3, label='R(x(t))')
    for event in walker.boundary_events:
        axes[2].axvline(x=event['t'], color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time step t')
    axes[2].set_ylabel('Clock Rate R(x(t))')
    axes[2].set_title('Clock Rate: R(x) = χ_FEG · K(x) · (1 + Q_9/11(x))')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('antclock_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_geometry(walker, history):
    """Plot geometry: (cos(φ_t), sin(φ_t)) on unit circle."""
    x_coords, y_coords = walker.get_geometry()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot unit circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=1, alpha=0.3, label='Unit circle')

    # Plot trajectory
    ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='φ(t) trajectory')
    ax.scatter(x_coords[0], y_coords[0], color='g', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(x_coords[-1], y_coords[-1], color='r', s=100, marker='s', label='End', zorder=5)

    # Mark boundary crossings
    for i, event in enumerate(walker.boundary_events):
        if event['t'] < len(x_coords):
            ax.scatter(x_coords[event['t']], y_coords[event['t']],
                      color='orange', s=50, marker='x', zorder=4)

    ax.set_xlabel('cos(φ)')
    ax.set_ylabel('sin(φ)')
    ax.set_title('Arcs of Rotation Punctured by Phase-Rate Shifts')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('antclock_geometry.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
