#!/usr/bin/env python3
"""
row7_demo.py

CE1.row7-digit-mirror: Symmetry Breaking at Shell 7

Demonstrates the row7 digit morphism and its effects on the curvature walker.
"""

import numpy as np
import matplotlib.pyplot as plt
from clock import (
    CurvatureClockWalker, Row7DigitMirror, Row7ActivatedWalker
)

def demo_row7_morphism():
    """Demonstrate the row7 digit morphism properties."""
    print("=" * 80)
    print("CE1.ROW7-DIGIT-MIRROR: SYMMETRY BREAKING")
    print("=" * 80)

    print("\nRow7 morphism: d^7 mod 10")
    print("Digit mapping:")
    for d in range(10):
        mapped = Row7DigitMirror.digit_morphism(d)
        cycle = Row7DigitMirror.get_cycle(d)
        fixed = "FIXED" if Row7DigitMirror.is_fixed_point(d) else "MOVES"
        print(f"  {d} → {mapped}  ({fixed}, cycle: {cycle})")

    print("\nPartition:")
    print(f"  Fixed points: {sorted(Row7DigitMirror.FIXED_POINTS)}")
    print(f"  Oscillating pairs: {list(Row7DigitMirror.CYCLES)}")

    # Test on some numbers
    test_numbers = [1, 10, 100, 123, 999, 1024, 2023, 7890]
    print("\nApplying to numbers:")
    for x in test_numbers:
        morphed = Row7DigitMirror.apply_to_number(x)
        analysis = Row7DigitMirror.analyze_digit_partition(x)
        energy = Row7DigitMirror.morphism_energy(x)
        changed = "CHANGED" if x != morphed else "unchanged"

        print("8d")

        if analysis['cycles']:
            cycle_str = ", ".join([f"{cycle}: {count}" for cycle, count in analysis['cycles'].items()])
            print(f"           Cycles: {cycle_str}")

def demo_walker_comparison():
    """Compare normal walker vs row7-activated walker."""
    print("\n" + "=" * 80)
    print("COMPARING WALKERS: NORMAL VS ROW7-ACTIVATED")
    print("=" * 80)

    # Create both walkers
    normal_walker = CurvatureClockWalker(x_0=1)
    row7_walker = Row7ActivatedWalker(x_0=1, activate_row7_at_shell=7)

    # Evolve both
    steps = 150
    normal_history, normal_summary = normal_walker.evolve(steps)
    row7_history, row7_summary = row7_walker.evolve(steps)

    print(f"\nEvolved both walkers for {steps} steps:")
    print(f"Normal walker:  x_final={normal_summary['x_final']}, boundaries={normal_summary['boundary_count']}")
    print(f"Row7 walker:    x_final={row7_summary['x_final']}, boundaries={row7_summary['boundary_count']}")
    print(f"Row7 events:    {len(row7_walker.row7_events)}")

    # Show row7 events
    if row7_walker.row7_events:
        print("\nRow7 morphism events:")
        for event in row7_walker.row7_events[:5]:  # Show first 5
            print("3d")

    # Compare trajectories
    print("\nTrajectory comparison (first 10 steps):")
    print("Step | Normal x     | Row7 x       | Row7 Applied?")
    print("-" * 50)
    for i in range(min(10, len(normal_history))):
        normal_x = normal_history[i]['x_next']
        row7_x = row7_history[i]['x_next']
        applied = "YES" if row7_history[i].get('row7_applied', False) else "   "
        print("4d")

    # Compare at larger scales
    print("\nComparison at larger steps:")
    check_points = [49, 99, 149]
    for t in check_points:
        if t < len(normal_history):
            normal_x = normal_history[t]['x_next']
            row7_x = row7_history[t]['x_next']
            normal_d = normal_history[t]['d_next']
            row7_d = row7_history[t]['d_next']
            applied = "YES" if row7_history[t].get('row7_applied', False) else "   "
            print("3d")

def plot_row7_comparison():
    """Plot comparison between normal and row7-activated walkers."""
    print("\n" + "=" * 80)
    print("VISUALIZING ROW7 SYMMETRY BREAKING")
    print("=" * 80)

    # Create walkers
    normal_walker = CurvatureClockWalker(x_0=1)
    row7_walker = Row7ActivatedWalker(x_0=1, activate_row7_at_shell=7)

    # Evolve
    steps = 200
    normal_history, _ = normal_walker.evolve(steps)
    row7_history, _ = row7_walker.evolve(steps)

    # Extract sequences
    t_seq = [h['t'] for h in normal_history]
    normal_x_seq = [h['x'] for h in normal_history]
    row7_x_seq = [h['x'] for h in row7_history]
    normal_phi_seq = [h['phi'] for h in normal_history]
    row7_phi_seq = [h['phi'] for h in row7_history]
    row7_energy_seq = [h.get('row7_energy', 0) for h in row7_history]
    row7_applied_seq = [1 if h.get('row7_applied', False) else 0 for h in row7_history]

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Body trajectories
    ax1 = axes[0, 0]
    ax1.plot(t_seq, normal_x_seq, 'b-', linewidth=2, label='Normal walker', alpha=0.7)
    ax1.plot(t_seq, row7_x_seq, 'r-', linewidth=2, label='Row7 walker', alpha=0.7)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Body x(t)')
    ax1.set_title('Body Trajectories: Normal vs Row7')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark where row7 activates (shell >= 7)
    activation_t = None
    for i, h in enumerate(row7_history):
        if h['d'] >= 7:
            activation_t = i
            break
    if activation_t:
        ax1.axvline(x=activation_t, color='orange', linestyle='--', alpha=0.7,
                   label='Row7 activation')
        ax1.legend()

    # Plot 2: Phase space comparison
    ax2 = axes[0, 1]
    ax2.plot(normal_x_seq, normal_phi_seq, 'b.', alpha=0.6, label='Normal', markersize=2)
    ax2.plot(row7_x_seq, row7_phi_seq, 'r.', alpha=0.6, label='Row7', markersize=2)
    ax2.set_xlabel('Body x')
    ax2.set_ylabel('Phase φ')
    ax2.set_title('Phase Space: Normal vs Row7')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Row7 energy and applications
    ax3 = axes[1, 0]
    ax3.plot(t_seq, row7_energy_seq, 'g-', linewidth=2, label='Row7 energy')
    ax3.fill_between(t_seq, 0, row7_applied_seq, alpha=0.3, color='orange', label='Row7 applied')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Row7 energy')
    ax3.set_title('Row7 Energy & Application Events')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Difference analysis
    ax4 = axes[1, 1]
    x_diff = [r - n for r, n in zip(row7_x_seq, normal_x_seq)]
    phi_diff = [(r - n) % (2*np.pi) for r, n in zip(row7_phi_seq, normal_phi_seq)]
    # Wrap phi differences to [-pi, pi]
    phi_diff = [(d - 2*np.pi if d > np.pi else d) for d in phi_diff]

    ax4.plot(t_seq, x_diff, 'purple', linewidth=2, label='x difference')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Difference (Row7 - Normal)')
    ax4.set_title('Trajectory Differences')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(t_seq, phi_diff, 'orange', linewidth=1, alpha=0.7, label='φ difference')
    ax4_twin.set_ylabel('Phase difference', color='orange')
    ax4_twin.tick_params(axis='y', labelcolor='orange')

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('row7_symmetry_breaking.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Saved: row7_symmetry_breaking.png")
    print("  - Body trajectories: normal vs row7-activated")
    print("  - Phase space comparison")
    print("  - Row7 energy evolution and application events")
    print("  - Trajectory differences (x and φ)")

    # Summary statistics
    print("\nSummary statistics:")
    total_row7_applications = sum(row7_applied_seq)
    avg_energy_after_activation = np.mean(row7_energy_seq[len(row7_energy_seq)//2:])
    max_x_diff = max(abs(d) for d in x_diff)
    rms_phi_diff = np.sqrt(np.mean([d**2 for d in phi_diff]))

    print(f"  Row7 applications: {total_row7_applications}")
    print(".3f")
    print(f"  Max x difference: {max_x_diff}")
    print(".3f")

def demo_shell_7_transition():
    """Demonstrate what happens exactly at shell 7 transition."""
    print("\n" + "=" * 80)
    print("SHELL 7 TRANSITION ANALYSIS")
    print("=" * 80)

    # Test numbers around shell 7 boundary
    test_range = range(999999, 1000010)  # Around 10^6 boundary

    print("\nNumbers crossing into 7-digit shell:")
    print("Number      | Morphism   | Energy | Analysis")
    print("-" * 60)

    for x in test_range:
        morphed = Row7DigitMirror.apply_to_number(x)
        energy = Row7DigitMirror.morphism_energy(x)
        analysis = Row7DigitMirror.analyze_digit_partition(x)

        if x != morphed or energy > 0:
            cycles_str = ", ".join([f"{k}:{v}" for k,v in analysis['cycles'].items()]) if analysis['cycles'] else "none"
            print("10d")

def main():
    """Run the complete row7 demonstration."""
    print("=" * 80)
    print("CE1.ROW7-DIGIT-MIRROR: SYMMETRY BREAKING AT SHELL 7")
    print("=" * 80)
    print("\nRow 7 is where Pascal curvature symmetry first truly breaks.")
    print("The 7th power mod 10 creates exactly that local digit structure.")
    print("\nFixed sector: {0,1,4,5,6,9} (ballast/tension digits)")
    print("Oscillating: 2↔8, 3↔7 (mirror pairs summing to 10)")

    # Demonstrate the morphism
    demo_row7_morphism()

    # Compare walkers
    demo_walker_comparison()

    # Visualize the effects
    plot_row7_comparison()

    # Analyze shell 7 transition
    demo_shell_7_transition()

    print("\n" + "=" * 80)
    print("ROW7 SYMMETRY BREAKING COMPLETE")
    print("=" * 80)
    print("\n✓ Row7 morphism: d^7 mod 10 partitions digits into stable/oscillating")
    print("✓ Activated at shell 7 boundaries in the curvature walker")
    print("✓ Creates bifurcations in trajectory and phase evolution")
    print("✓ Local digit mirrors feed into global curvature clock")
    print("\nThe first true symmetry break in Pascal's triangle")
    print("now has its local digit-level realization.")

if __name__ == "__main__":
    main()
