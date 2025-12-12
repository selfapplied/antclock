#!/usr/bin/env python3
"""
Visualization for ERV analysis results.

Creates plots for:
- Stress/coherence trajectories
- Nash equilibrium decisions
- Volte activation events
- Sequence analysis summaries
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class ERVVisualizer:
    """Create visualizations for ERV Volte analysis results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = Path(__file__).parent / "data" / "visualizations"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_stress_coherence(self, analysis_file: Path, output_file: Optional[Path] = None):
        """Plot stress vs coherence for all sequences."""
        with open(analysis_file, 'r') as f:
            data = json.load(f)
        
        analyses = data['analyses']
        stresses = [a['transform']['stress_S'] for a in analyses]
        coherences = [a['transform']['coherence_C'] for a in analyses]
        volte_activated = [a['transform']['volte_activated'] for a in analyses]
        threshold = analyses[0]['transform']['threshold'] if analyses else 0.638
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by Volte activation
        colors = ['red' if v else 'blue' for v in volte_activated]
        
        scatter = ax.scatter(stresses, coherences, c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=1)
        
        # Add threshold line
        ax.axvline(x=threshold, color='orange', linestyle='--', linewidth=2, label=f'Volte Threshold (κ={threshold})')
        
        # Labels
        ax.set_xlabel('Stress (S)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coherence (C)', fontsize=12, fontweight='bold')
        ax.set_title('ERV Analysis: Stress vs Coherence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legend
        activated_patch = mpatches.Patch(color='red', label='Volte Activated')
        not_activated_patch = mpatches.Patch(color='blue', label='No Volte')
        ax.legend(handles=[activated_patch, not_activated_patch, ax.get_lines()[0]], loc='best')
        
        plt.tight_layout()
        
        if output_file is None:
            output_file = self.output_dir / f"{analysis_file.stem}_stress_coherence.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved stress/coherence plot to {output_file}")
        plt.close()
    
    def plot_nash_equilibrium(self, analysis_file: Path, output_file: Optional[Path] = None):
        """Plot Nash equilibrium decision space."""
        with open(analysis_file, 'r') as f:
            data = json.load(f)
        
        analyses = data['analyses']
        
        # Extract Nash data
        G_values = []
        G_crit_values = []
        hurst_values = []
        should_exapt = []
        
        for a in analyses:
            nash = a['transform'].get('nash_equilibrium', {})
            if nash:
                G_values.append(nash.get('G', 0))
                G_crit_values.append(nash.get('g_crit', 0))
                hurst_values.append(nash.get('hurst', 0))
                should_exapt.append(nash.get('should_exapt', False))
        
        if not G_values:
            print("⚠️ No Nash equilibrium data found in analysis")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: G vs G_crit
        colors = ['green' if exapt else 'red' for exapt in should_exapt]
        ax1.scatter(G_values, G_crit_values, c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=1)
        
        # Diagonal line (G = G_crit)
        max_val = max(max(G_values), max(G_crit_values)) if G_values else 1.0
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='G = G_crit')
        
        ax1.set_xlabel('Composition Gain (G)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Critical Threshold (G_crit)', fontsize=12, fontweight='bold')
        ax1.set_title('Nash Equilibrium: G vs G_crit', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Hurst vs G
        ax2.scatter(hurst_values, G_values, c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=1)
        ax2.set_xlabel('Hurst Exponent (H)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Composition Gain (G)', fontsize=12, fontweight='bold')
        ax2.set_title('Nash Equilibrium: Hurst vs G', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Legend
        exapt_patch = mpatches.Patch(color='green', label='Exaptation Recommended')
        protect_patch = mpatches.Patch(color='red', label='Protection Recommended')
        ax2.legend(handles=[exapt_patch, protect_patch], loc='best')
        
        plt.tight_layout()
        
        if output_file is None:
            output_file = self.output_dir / f"{analysis_file.stem}_nash_equilibrium.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Nash equilibrium plot to {output_file}")
        plt.close()
    
    def plot_summary_statistics(self, analysis_file: Path, output_file: Optional[Path] = None):
        """Create summary statistics visualization."""
        with open(analysis_file, 'r') as f:
            data = json.load(f)
        
        summary = data['summary']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Volte activations
        ax = axes[0, 0]
        activated = summary['volte_activated_count']
        total = data['num_sequences']
        not_activated = total - activated
        
        ax.bar(['Volte Activated', 'No Volte'], [activated, not_activated], 
               color=['red', 'blue'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Sequences', fontweight='bold')
        ax.set_title('Volte Activation Summary', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Average metrics
        ax = axes[0, 1]
        metrics = ['Stress', 'Coherence', 'Exaptation\nPotential']
        values = [
            summary['avg_stress'],
            summary['avg_coherence'],
            summary['avg_exaptation_potential']
        ]
        colors = ['red', 'blue', 'green']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Average Value', fontweight='bold')
        ax.set_title('Average Metrics', fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Stress distribution
        ax = axes[1, 0]
        stresses = [a['transform']['stress_S'] for a in data['analyses']]
        ax.hist(stresses, bins=10, color='red', alpha=0.7, edgecolor='black')
        ax.axvline(x=summary.get('threshold', 0.638), color='orange', 
                  linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Stress (S)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Stress Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Coherence distribution
        ax = axes[1, 1]
        coherences = [a['transform']['coherence_C'] for a in data['analyses']]
        ax.hist(coherences, bins=10, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Coherence (C)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Coherence Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'ERV Analysis Summary: {data["num_sequences"]} Sequences', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if output_file is None:
            output_file = self.output_dir / f"{analysis_file.stem}_summary.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved summary statistics to {output_file}")
        plt.close()
    
    def create_all_visualizations(self, analysis_file: Path):
        """Create all visualizations for an analysis file."""
        print(f"📊 Creating visualizations for {analysis_file.name}...")
        
        self.plot_stress_coherence(analysis_file)
        self.plot_nash_equilibrium(analysis_file)
        self.plot_summary_statistics(analysis_file)
        
        print(f"✅ All visualizations saved to {self.output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize ERV analysis results")
    parser.add_argument('analysis_file', type=Path, help='JSON analysis file')
    parser.add_argument('--output-dir', type=Path, help='Output directory for plots')
    parser.add_argument('--type', choices=['all', 'stress-coherence', 'nash', 'summary'],
                       default='all', help='Type of visualization')
    
    args = parser.parse_args()
    
    visualizer = ERVVisualizer(args.output_dir)
    
    if args.type == 'all':
        visualizer.create_all_visualizations(args.analysis_file)
    elif args.type == 'stress-coherence':
        visualizer.plot_stress_coherence(args.analysis_file)
    elif args.type == 'nash':
        visualizer.plot_nash_equilibrium(args.analysis_file)
    elif args.type == 'summary':
        visualizer.plot_summary_statistics(args.analysis_file)


if __name__ == '__main__':
    main()





