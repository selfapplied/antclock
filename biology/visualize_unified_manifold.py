#!/usr/bin/env python3
"""
Visualize Unified Guardian-Mirror Manifold

Shows the cross-layer mapping between:
- Genetic Guardian Vectors (genome-space)
- Mirror Neuron FEG (behavior-space)

Same topology. Different scales.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class UnifiedManifoldVisualizer:
    """Visualize the unified Guardian-Mirror manifold."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = Path(__file__).parent / "data" / "visualizations"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_cross_layer_mapping(self, result_file: Path, output_file: Optional[Path] = None):
        """Plot centroid mapping across genetic and behavioral layers."""
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        mappings = data['ce3_evolution']['centroid_mapping']['mappings']
        
        if not mappings:
            print("⚠️ No centroid mappings found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Genetic centroids
        genetic_curvs = [m['genetic_centroid'].get('curvature', 0.0) for m in mappings]
        genetic_ids = [m['genetic_centroid'].get('id', 'g') for m in mappings]
        
        ax1.scatter(range(len(genetic_curvs)), genetic_curvs, 
                   s=200, c='blue', alpha=0.7, edgecolors='black', linewidth=2)
        ax1.set_xlabel('Genetic Centroid Index', fontweight='bold')
        ax1.set_ylabel('Curvature', fontweight='bold')
        ax1.set_title('Genetic Guardian Centroids\n(Genome-Space Stability)', 
                     fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Behavioral centroids with mapping lines
        behavioral_curvs = [m['behavioral_centroid'].get('curvature', 0.0) for m in mappings]
        behavioral_ids = [m['behavioral_centroid'].get('id', 'b') for m in mappings]
        distances = [m['distance'] for m in mappings]
        
        ax2.scatter(range(len(behavioral_curvs)), behavioral_curvs,
                   s=200, c='red', alpha=0.7, edgecolors='black', linewidth=2)
        ax2.set_xlabel('Behavioral Centroid Index', fontweight='bold')
        ax2.set_ylabel('Curvature', fontweight='bold')
        ax2.set_title('Mirror Neuron Centroids\n(Behavior-Space Stability)',
                     fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Draw mapping lines
        for i, (g_curv, b_curv, dist) in enumerate(zip(genetic_curvs, behavioral_curvs, distances)):
            ax1.plot([i, i], [g_curv, g_curv + dist * 0.5], 'g--', alpha=0.5, linewidth=1)
            ax2.plot([i, i], [b_curv, b_curv - dist * 0.5], 'g--', alpha=0.5, linewidth=1)
        
        plt.suptitle('Cross-Layer Centroid Mapping\nGenetic ↔ Behavioral', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_file is None:
            output_file = self.output_dir / f"{result_file.stem}_cross_layer_mapping.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved cross-layer mapping to {output_file}")
        plt.close()
    
    def plot_unified_topology(self, result_file: Path, output_file: Optional[Path] = None):
        """Plot unified topology showing same structure at different scales."""
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data
        genetic_curv = data['ce1_static']['domain']['genetic_manifold']['curvature']
        behavioral_curv = data['ce1_static']['domain']['behavioral_manifold']['curvature']
        
        genetic_centroids = len(data['ce1_static']['domain']['genetic_manifold']['centroids'])
        behavioral_centroids = len(data['ce1_static']['domain']['behavioral_manifold']['centroids'])
        
        # Create layered visualization
        scales = ['Genome-Space\n(Guardian Vectors)', 
                 'Behavior-Space\n(Mirror Neurons)',
                 'Intention-Space\n(Cognitive FEG)']
        
        curvatures = [genetic_curv, behavioral_curv, (genetic_curv + behavioral_curv) / 2]
        centroid_counts = [genetic_centroids, behavioral_centroids, 
                          (genetic_centroids + behavioral_centroids) / 2]
        
        # Plot as layered manifolds
        y_positions = [2, 1, 0]
        colors = ['blue', 'red', 'green']
        
        for i, (scale, curv, centroids, y_pos, color) in enumerate(
            zip(scales, curvatures, centroid_counts, y_positions, colors)
        ):
            # Draw manifold as curved line
            x = np.linspace(0, 10, 100)
            y = y_pos + 0.3 * np.sin(x * curv * 2) * centroids / 2
            ax.plot(x, y, color=color, linewidth=3, alpha=0.7, label=scale)
            
            # Mark centroids
            centroid_x = np.linspace(2, 8, int(centroids))
            centroid_y = y_pos + 0.3 * np.sin(centroid_x * curv * 2) * centroids / 2
            ax.scatter(centroid_x, centroid_y, s=150, c=color, 
                      edgecolors='black', linewidth=2, zorder=5)
        
        # Draw connecting lines (same topology)
        ax.plot([5, 5], [2, 0], 'k--', alpha=0.3, linewidth=2, label='Same Topology')
        
        ax.set_ylabel('Manifold Layer', fontweight='bold', fontsize=12)
        ax.set_xlabel('Manifold Coordinate', fontweight='bold', fontsize=12)
        ax.set_title('Unified Guardian-Mirror Topology\nSame CE Geometry at Different Scales',
                    fontsize=14, fontweight='bold')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(scales)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file is None:
            output_file = self.output_dir / f"{result_file.stem}_unified_topology.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved unified topology to {output_file}")
        plt.close()
    
    def plot_antclock_phase_axis(self, result_file: Path, output_file: Optional[Path] = None):
        """Plot antclock as phase axis of selfhood."""
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        phase_axis = data['ce2_dynamics']['phase_axis']
        tau = phase_axis['tau']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create phase axis visualization
        time_points = np.linspace(0, 10, 100)
        genetic_phase = np.sin(time_points * 0.5 + tau)  # Genetic recursion
        behavioral_phase = np.sin(time_points * 0.5 + tau + np.pi/4)  # Behavioral recursion
        
        # Antclock provides driftless synchronization
        antclock_axis = np.zeros_like(time_points) + tau
        
        ax.plot(time_points, genetic_phase, 'b-', linewidth=2, 
               label='Genetic Recursion (Guardian)', alpha=0.7)
        ax.plot(time_points, behavioral_phase, 'r-', linewidth=2,
               label='Behavioral Recursion (Mirror)', alpha=0.7)
        ax.axhline(y=tau, color='orange', linestyle='--', linewidth=3,
                  label=f'Antclock Phase Axis (τ={tau:.2f})')
        
        ax.fill_between(time_points, genetic_phase, behavioral_phase, 
                       alpha=0.2, color='green', label='Synchronized Region')
        
        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel('Phase', fontweight='bold')
        ax.set_title('Antclock as Phase Axis of Selfhood\nSynchronizing Genetic & Behavioral Manifolds',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file is None:
            output_file = self.output_dir / f"{result_file.stem}_antclock_phase_axis.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved antclock phase axis to {output_file}")
        plt.close()
    
    def create_all_visualizations(self, result_file: Path):
        """Create all unified manifold visualizations."""
        print(f"📊 Creating unified manifold visualizations from {result_file.name}...")
        
        self.plot_cross_layer_mapping(result_file)
        self.plot_unified_topology(result_file)
        self.plot_antclock_phase_axis(result_file)
        
        print(f"✅ All visualizations saved to {self.output_dir}")


def main():
    """Create visualizations for unified Guardian-Mirror operator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize Unified Guardian-Mirror Manifold"
    )
    parser.add_argument('result_file', type=Path, help='Unified operator result JSON')
    parser.add_argument('--output-dir', type=Path, help='Output directory')
    parser.add_argument('--type', choices=['all', 'mapping', 'topology', 'phase'],
                       default='all', help='Visualization type')
    
    args = parser.parse_args()
    
    visualizer = UnifiedManifoldVisualizer(args.output_dir)
    
    if args.type == 'all':
        visualizer.create_all_visualizations(args.result_file)
    elif args.type == 'mapping':
        visualizer.plot_cross_layer_mapping(args.result_file)
    elif args.type == 'topology':
        visualizer.plot_unified_topology(args.result_file)
    elif args.type == 'phase':
        visualizer.plot_antclock_phase_axis(args.result_file)


if __name__ == '__main__':
    main()




