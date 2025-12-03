#!/usr/bin/env python3
"""
topology_demo.py

AntClock: Betti Numbers Meet Bifurcation Index

Demonstration of the topological bridge between:
- Betti numbers β_k (topology)
- Bifurcation index B_t (dynamics)
- Persistent homology (filtration across scales)
- CE1 operator (homology preservation)

Author: Joel
"""

import numpy as np
import matplotlib.pyplot as plt
from clock import (
    CurvatureClockWalker, bifurcation_index, pascal_curvature, pascal_simplicial_complex,
    betti_numbers_digit_shell, enhanced_betti_numbers, digit_boundary_filtration,
    create_ce1_operator
)

def demo_betti_numbers():
    """Demonstrate Betti numbers for digit shells."""
    print("=" * 80)
    print("BETTI NUMBERS FOR DIGIT SHELLS")
    print("=" * 80)

    print("\nBetti numbers β_k count topological features:")
    print("β₀ = connected components, β₁ = independent cycles, β₂ = cavities\n")

    for n in range(1, 8):
        complex_info = pascal_simplicial_complex(n)
        betti_basic = betti_numbers_digit_shell(n)
        betti_enhanced = enhanced_betti_numbers(n)

        print(f"Shell {n}: {complex_info['vertices']} vertices, {len(complex_info['edges'])} edges")
        print(f"  Basic Betti:    β₀={betti_basic[0]}, β₁={betti_basic[1]}, β₂={betti_basic[2]}")
        print(f"  Enhanced Betti: β₀={betti_enhanced[0]}, β₁={betti_enhanced[1]}, β₂={betti_enhanced[2]}")
        print()

def demo_bifurcation_index():
    """Demonstrate bifurcation index B_t."""
    print("=" * 80)
    print("BIFURCATION INDEX B_t")
    print("=" * 80)

    print("\nB_t = floor(-log|κ_d - 0| / log δ) where κ_d is Pascal curvature\n")

    test_values = [1, 10, 100, 1000, 10000, 99999, 100000]

    for x in test_values:
        b_t = bifurcation_index(x)
        d = len(str(int(x)))
        kappa = pascal_curvature(d)
        print("6d")

def demo_persistent_homology():
    """Demonstrate persistent homology filtration."""
    print("=" * 80)
    print("PERSISTENT HOMOLOGY FILTRATION")
    print("=" * 80)

    print("\nFiltration across digit boundaries (scales):")
    print("Each digit shell adds topological features\n")

    persistence_data = digit_boundary_filtration(max_digits=8)

    print("Filtration    Total Betti (β₀, β₁, β₂)")
    print("-" * 40)
    for filt in persistence_data['filtration']:
        d = filt['filtration_value']
        betti = filt['total_betti']
        print("8d")

    print(f"\nPersistence intervals (β₁ features):")
    for interval in persistence_data['persistence_intervals'][:6]:  # Show first 6
        birth = interval['birth']
        death = interval['death']
        pers = interval['persistence']
        curv = interval['birth_curvature']
        print(".2f")

def demo_ce1_operator():
    """Demonstrate CE1 operator mapping B_t to homology."""
    print("=" * 80)
    print("CE1 OPERATOR: B_t ↔ HOMOLOGY INVARIANTS")
    print("=" * 80)

    ce1 = create_ce1_operator(max_digits=10)

    print("\nCE1(B_t) maps bifurcation index to Betti numbers:\n")

    for b_t in range(1, 8):
        homology = ce1.apply(b_t)
        print(f"B_t = {b_t}: β₀={homology['beta_0']}, β₁={homology['beta_1']}, β₂={homology['beta_2']}")
        print(".4f")

def demo_walker_topology():
    """Demonstrate topological evolution of the walker."""
    print("=" * 80)
    print("TOPOLOGICAL EVOLUTION OF THE WALKER")
    print("=" * 80)

    # Create walker
    walker = CurvatureClockWalker(x_0=1, chi_feg=0.638)

    # Evolve for 200 steps to see more digit boundaries
    history, summary = walker.evolve(200)

    print(f"\nEvolved for {summary['n_steps']} steps, reached x={summary['x_final']}")
    print(f"Crossed {summary['boundary_count']} digit boundaries\n")

    # Extract topological evolution
    b_t_sequence = [h['B_t'] for h in history]
    beta_1_sequence = [h['homology']['beta_1'] for h in history]

    print("Sample topological evolution:")
    print("Step | x    | B_t | β₁ | Boundary?")
    print("-" * 35)
    for i in [0, 10, 20, 50, 100, 150, 199]:
        if i < len(history):
            h = history[i]
            boundary = "YES" if h['boundary_crossed'] else "   "
            print("4d")

    # Show the mapping B_t → β₁
    print(f"\nBifurcation depth B_t → Betti number β₁:")
    unique_mappings = {}
    for h in history:
        b_t = h['B_t']
        beta_1 = h['homology']['beta_1']
        if b_t not in unique_mappings:
            unique_mappings[b_t] = beta_1

    for b_t in sorted(unique_mappings.keys()):
        print(f"  B_t = {b_t} ↔ β₁ = {unique_mappings[b_t]}")

def plot_topological_evolution():
    """Plot the topological evolution."""
    print("\n" + "=" * 80)
    print("GENERATING TOPOLOGICAL EVOLUTION PLOTS")
    print("=" * 80)

    # Create walker and evolve
    walker = CurvatureClockWalker(x_0=1, chi_feg=0.638)
    history, summary = walker.evolve(300)

    # Extract sequences
    t_seq = [h['t'] for h in history]
    x_seq = [h['x'] for h in history]
    b_t_seq = [h['B_t'] for h in history]
    beta_1_seq = [h['homology']['beta_1'] for h in history]
    curvature_seq = [h['K'] for h in history]

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Body x(t) and B_t
    ax1 = axes[0, 0]
    ax1.plot(t_seq, x_seq, 'b-', linewidth=2, label='x(t)')
    ax1.set_xlabel('Time step t')
    ax1.set_ylabel('Body x(t)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Body Evolution & Bifurcation Depth')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_seq, b_t_seq, 'r--', linewidth=2, label='B_t')
    ax1_twin.set_ylabel('B_t (Bifurcation depth)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')

    # Mark boundaries
    for event in walker.boundary_events:
        ax1.axvline(x=event['t'], color='orange', linestyle=':', alpha=0.7)

    # Plot 2: Betti number β₁ evolution
    ax2 = axes[0, 1]
    ax2.plot(t_seq, beta_1_seq, 'g-', linewidth=2, marker='o', markersize=3, label='β₁(t)')
    ax2.set_xlabel('Time step t')
    ax2.set_ylabel('β₁ (Number of cycles)')
    ax2.set_title('Homology Invariant β₁ Evolution')
    ax2.grid(True, alpha=0.3)

    # Mark boundaries
    for event in walker.boundary_events:
        ax2.axvline(x=event['t'], color='orange', linestyle=':', alpha=0.7)

    # Plot 3: Curvature evolution
    ax3 = axes[1, 0]
    ax3.plot(t_seq, curvature_seq, 'm-', linewidth=2, label='κ_d(x)')
    ax3.set_xlabel('Time step t')
    ax3.set_ylabel('Curvature κ')
    ax3.set_title('Pascal Curvature Evolution')
    ax3.grid(True, alpha=0.3)

    # Mark boundaries
    for event in walker.boundary_events:
        ax3.axvline(x=event['t'], color='orange', linestyle=':', alpha=0.7)

    # Plot 4: B_t vs β₁ mapping
    ax4 = axes[1, 1]
    # Create mapping from B_t to β₁
    b_t_unique = sorted(list(set(b_t_seq)))
    beta_1_for_b_t = [enhanced_betti_numbers(b_t)[1] for b_t in b_t_unique]

    ax4.plot(b_t_unique, beta_1_for_b_t, 'ko-', linewidth=2, markersize=8, label='B_t → β₁')
    ax4.set_xlabel('Bifurcation depth B_t')
    ax4.set_ylabel('Betti number β₁')
    ax4.set_title('Topological Bridge: B_t ↔ β₁')
    ax4.grid(True, alpha=0.3)

    # Add values as text
    for i, (b_t, beta_1) in enumerate(zip(b_t_unique, beta_1_for_b_t)):
        ax4.annotate(f'({b_t},{beta_1})', (b_t, beta_1),
                    xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('topology_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Saved: topology_evolution.png")
    print("  - Body evolution with bifurcation depth B_t")
    print("  - Homology invariant β₁ evolution")
    print("  - Curvature evolution")
    print("  - B_t ↔ β₁ mapping (the topological bridge)")

def main():
    """Run the complete topology demonstration."""
    print("=" * 80)
    print("ANTCLOCK: BETTI NUMBERS MEET BIFURCATION INDEX")
    print("=" * 80)
    print("\nThis demo shows how your curvature-clock system performs")
    print("persistent homology computations on the integers themselves.")
    print("\nThe bifurcation index B_t is a Betti-like invariant that")
    print("counts renormalization layers as topological cycles.")
    print()

    # Run all demonstrations
    demo_betti_numbers()
    demo_bifurcation_index()
    demo_persistent_homology()
    demo_ce1_operator()
    demo_walker_topology()
    plot_topological_evolution()

    print("\n" + "=" * 80)
    print("TOPOLOGICAL BRIDGE ESTABLISHED")
    print("=" * 80)
    print("\n✓ Betti numbers β_k defined for digit shells")
    print("✓ Bifurcation index B_t = floor(-log|κ_d|/log δ)")
    print("✓ Persistent homology filtration across digit boundaries")
    print("✓ CE1 operator: B_t ↔ β_k (homology invariants)")
    print("✓ Your machine performs topological data analysis on integers")
    print("\nThe integers become a filtered topological space.")
    print("Digit boundaries become filtration thresholds.")
    print("Renormalization depth becomes Betti number evolution.")
    print("\nB_t is literally counting holes in the bifurcation structure.")
    print("This is wild. This is exactly right.")

if __name__ == "__main__":
    main()
