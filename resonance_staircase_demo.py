#!/usr/bin/env python3
"""
resonance_staircase_demo.py

The Renormalization Staircase: Where Global + Local Symmetry Breaking Align

Shows:
1. Mirror phase resonance: exponents e ≡ 3 mod 4
2. Curvature-critical rows: where Pascal curvature flips
3. Renormalization staircase: alignment points
4. Resonance spectrum: frequency map across shells
"""

import numpy as np
import matplotlib.pyplot as plt
from clock import (
    MirrorPhaseResonance, CurvatureCriticalRows, RenormalizationStaircase,
    pascal_curvature, pascal_radius
)

def demo_mirror_phase_resonance():
    """Demonstrate mirror phase resonance across exponents."""
    print("=" * 80)
    print("🎯 MIRROR PHASE RESONANCE: e ≡ 3 mod 4")
    print("=" * 80)

    print("\nφ(10) = 4 fundamental cycle in exponentiation mod 10")
    print("Mirror phase activates when e ≡ 3 mod 4\n")

    # Show the 4-cycle structure
    print("Exponent classes mod 4:")
    for e_mod4 in range(4):
        exponents = [e for e in range(1, 20) if e % 4 == e_mod4][:5]
        if exponents:
            print(f"  e ≡ {e_mod4} mod 4: {exponents}")

    print("\nMirror phase exponents (e ≡ 3 mod 4):")
    mirror_exponents = MirrorPhaseResonance.find_all_mirror_exponents(24)
    print(f"  {mirror_exponents}")

    print("\nAll mirror exponents give the SAME digit pattern:")
    for e in [7, 11, 15, 19, 23]:
        pattern = MirrorPhaseResonance.get_mirror_pattern(e)
        is_equiv = MirrorPhaseResonance.is_equivalent_to_row7(e)
        print(f"  e={e}: {pattern} {'✓' if is_equiv else '✗'}")

def demo_curvature_critical_rows():
    """Demonstrate curvature-critical rows."""
    print("\n" + "=" * 80)
    print("🌊 CURVATURE-CRITICAL ROWS: Where Pascal Curvature Flips")
    print("=" * 80)

    critical_rows = CurvatureCriticalRows.find_critical_rows(30)
    print(f"\nCurvature-critical rows (inflection points): {critical_rows}")

    print("\nDetailed analysis:")
    print("Row | Curvature κ | Sign Change | Magnitude Jump | Critical?")
    print("-" * 65)

    for n in range(5, 25):
        props = CurvatureCriticalRows.get_row_properties(n)
        kappa = props['curvature']
        sign_change = CurvatureCriticalRows.curvature_sign_change(n)
        mag_jump = CurvatureCriticalRows.curvature_magnitude_jump(n)
        critical = props['is_critical']

        print("3d")

def demo_renormalization_staircase():
    """Demonstrate the renormalization staircase."""
    print("\n" + "=" * 80)
    print("🧩 RENORMALIZATION STAIRCASE: Global + Local Alignment")
    print("=" * 80)

    staircase_computer = RenormalizationStaircase(max_shell=50)
    staircase = staircase_computer.get_staircase()

    print(f"\nRenormalization staircase (excitation levels): {len(staircase)} steps")
    print("\nThe CE1 excitation chain:")
    print("Level | Shell | Description")
    print("-" * 50)

    for step in staircase:
        level = step['excitation_level']
        shell = step['shell']
        desc = step['description']
        print("5d")

    print("\nComplete mapping:")
    print("Shell 1-6: Flat manifold (no symmetry breaking)")
    print("Shell 7:   First excitation (curvature break + mirror onset)")
    print("Shell 11:  Second excitation (entropy saddle + mirror)")
    print("Shell 15:  Third excitation (recovery + mirror)")
    print("Shell 17:  Deep excitation (major inflection + mirror)")
    print("Shell 23:  Monstrous excitation (chaotic amplification)")

def plot_resonance_spectrum():
    """Plot the resonance spectrum across all shells."""
    print("\n" + "=" * 80)
    print("🎭 RESONANCE SPECTRUM: Frequency Map Across Shells")
    print("=" * 80)

    staircase_computer = RenormalizationStaircase(max_shell=50)
    spectrum = staircase_computer.get_resonance_spectrum()

    # Prepare data for plotting
    shells = list(spectrum.keys())
    curvatures = [spectrum[n]['curvature'] for n in shells]
    phases = [spectrum[n]['phase'] for n in shells]
    excitation_levels = [spectrum[n]['excitation_level'] for n in shells]

    # Phase colors
    phase_colors = {
        'Flat_Manifold': 'lightgray',
        'Regular': 'lightblue',
        'Mirror_Phase': 'orange',
        'Critical_Only': 'red',
        'Excitation_1': 'darkred',
        'Excitation_2': 'purple',
        'Excitation_3': 'darkgreen',
        'Excitation_4': 'darkblue',
        'Excitation_5': 'black',
        'Excitation_6': 'gold'
    }

    colors = [phase_colors.get(phase, 'gray') for phase in phases]

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Phase classification
    ax1 = axes[0, 0]
    for phase in set(phases):
        phase_shells = [n for n, data in spectrum.items() if data['phase'] == phase]
        phase_curvatures = [spectrum[n]['curvature'] for n in phase_shells]
        ax1.scatter(phase_shells, phase_curvatures,
                   c=phase_colors.get(phase, 'gray'),
                   label=phase, alpha=0.7, s=50)
    ax1.set_xlabel('Shell n')
    ax1.set_ylabel('Pascal Curvature κ_n')
    ax1.set_title('Resonance Spectrum: Phase Classification')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Excitation staircase
    ax2 = axes[0, 1]
    excitation_shells = [n for n, data in spectrum.items() if data['excitation_level'] > 0]
    excitation_curvatures = [spectrum[n]['curvature'] for n in excitation_shells]
    excitation_levels_plot = [spectrum[n]['excitation_level'] for n in excitation_shells]

    scatter = ax2.scatter(excitation_shells, excitation_curvatures,
                         c=excitation_levels_plot, cmap='viridis', s=100)
    ax2.set_xlabel('Shell n')
    ax2.set_ylabel('Pascal Curvature κ_n')
    ax2.set_title('Renormalization Staircase: Excitation Levels')
    plt.colorbar(scatter, ax=ax2, label='Excitation Level')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Curvature evolution with markers
    ax3 = axes[1, 0]
    ax3.plot(shells, curvatures, 'b-', linewidth=2, alpha=0.7)

    # Mark staircase points
    for step in staircase_computer.get_staircase():
        shell = step['shell']
        curvature = spectrum[shell]['curvature']
        level = step['excitation_level']
        ax3.scatter([shell], [curvature], c='red', s=100, zorder=5)
        ax3.annotate(f'Ex{level}', (shell, curvature),
                    xytext=(5, 5), textcoords='offset points')

    ax3.set_xlabel('Shell n')
    ax3.set_ylabel('Pascal Curvature κ_n')
    ax3.set_title('Curvature Evolution with Staircase Markers')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mirror phase alignment
    ax4 = axes[1, 1]
    mirror_shells = [n for n, data in spectrum.items() if data['is_mirror_phase']]
    critical_shells = [n for n, data in spectrum.items() if data['is_critical']]
    staircase_shells = [n for n, data in spectrum.items() if data['is_staircase']]

    # All shells
    ax4.scatter(shells, [0] * len(shells), c='lightgray', alpha=0.5, s=30, label='All shells')

    # Mirror phase
    ax4.scatter(mirror_shells, [0] * len(mirror_shells), c='orange', s=50, label='Mirror phase')

    # Critical rows
    ax4.scatter(critical_shells, [1] * len(critical_shells), c='red', s=50, label='Critical curvature')

    # Staircase (intersection)
    ax4.scatter(staircase_shells, [1] * len(staircase_shells), c='darkred', s=100, marker='*', label='Staircase')

    ax4.set_xlabel('Shell n')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Mirror Phase', 'Critical Curvature'])
    ax4.set_title('Phase Alignment: Mirror × Critical = Staircase')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('resonance_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Saved: resonance_spectrum.png")
    print("  - Phase classification across shells")
    print("  - Renormalization staircase excitation levels")
    print("  - Curvature evolution with staircase markers")
    print("  - Mirror phase × critical curvature = staircase alignment")

    # Summary statistics
    total_shells = len(spectrum)
    mirror_count = len([n for n, data in spectrum.items() if data['is_mirror_phase']])
    critical_count = len([n for n, data in spectrum.items() if data['is_critical']])
    staircase_count = len([n for n, data in spectrum.items() if data['is_staircase']])

    print("\nSpectrum statistics (shells 1-50):")
    print(f"  Total shells: {total_shells}")
    print(f"  Mirror phase: {mirror_count}")
    print(f"  Critical curvature: {critical_count}")
    print(f"  Renormalization staircase: {staircase_count}")
    print(f"  Mirror × Critical overlap: {staircase_count}")

def demo_staircase_predictions():
    """Demonstrate staircase predictions for walker evolution."""
    print("\n" + "=" * 80)
    print("🚪 STAIRCASE PREDICTIONS: Walker Evolution")
    print("=" * 80)

    staircase_computer = RenormalizationStaircase(max_shell=100)

    print("\nStaircase predictions for walker evolution:")
    test_shells = [5, 7, 10, 11, 15, 17, 20, 23, 25]

    print("Current Shell | Next Excitation | Levels to Jump")
    print("-" * 50)

    for current in test_shells:
        next_excitation = staircase_computer.get_next_excitation_shell(current)
        if next_excitation > 0:
            levels_to_jump = next_excitation - current
            print("13d")
        else:
            print("13d")

    print("\nExcitation level meanings:")
    for step in staircase_computer.get_staircase():
        level = step['excitation_level']
        shell = step['shell']
        desc = step['description'].split(':')[0]  # Short description
        print("2d")

def main():
    """Run the complete resonance staircase demonstration."""
    print("=" * 80)
    print("🧨 THE RENORMALIZATION STAIRCASE")
    print("Where Global Curvature Breaks + Local Mirror Phase Align")
    print("=" * 80)
    print("\nφ(10) = 4 creates the fundamental cycle.")
    print("e ≡ 3 mod 4 activates the mirror phase.")
    print("Row 7 is the first coincident break.")
    print("The staircase is the RG excitation chain.")

    # Demonstrate each component
    demo_mirror_phase_resonance()
    demo_curvature_critical_rows()
    demo_renormalization_staircase()

    # Visualize the spectrum
    plot_resonance_spectrum()

    # Show predictions
    demo_staircase_predictions()

    print("\n" + "=" * 80)
    print("🎯 THE STAIRCASE IS COMPLETE")
    print("=" * 80)
    print("\n✓ Mirror phase: e ≡ 3 mod 4 (7, 11, 15, 19, 23, ...)")
    print("✓ Critical rows: where Pascal curvature flips")
    print("✓ Renormalization staircase: their intersection")
    print("✓ Resonance spectrum: phase classification across shells")
    print("\n✓ Shells 1-6: Flat manifold")
    print("✓ Shell 7: First excitation")
    print("✓ Shell 11: Second excitation")
    print("✓ Shell 15: Third excitation")
    print("✓ Shell 17: Deep excitation")
    print("✓ Shell 23: Monstrous excitation")
    print("\nThis is your CE1 excitation chain.")
    print("This is your RG ladder.")
    print("This is the staircase you climb.")

if __name__ == "__main__":
    main()
