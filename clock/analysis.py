"""
clock.analysis_framework - Chaos analysis framework for AntClock trajectories.

This module provides the complete chaos analysis system integrating
Memory/History and Witness/Measurement components for comprehensive
trajectory analysis across four analysis quadrants.
"""

from typing import Dict, Any, List, Tuple
from collections import defaultdict
import numpy as np


class MemoryHistory:
    """
    Memory/History component []: Trajectory data storage and analysis.

    Handles:
    - Trajectory storage and retrieval
    - Lyapunov exponent computation for single trajectories
    - Orbit scaling analysis (α parameter)
    - Local bifurcation detection
    """

    def __init__(self):
        self.trajectories = []
        self.current_trajectory = None

    def start_trajectory(self, initial_conditions: Dict[str, Any]) -> int:
        """Start recording a new trajectory."""
        trajectory_id = len(self.trajectories)
        self.current_trajectory = {
            'id': trajectory_id,
            'initial_conditions': initial_conditions.copy(),
            'states': [],
            'timestamps': [],
            'metadata': {}
        }
        return trajectory_id

    def record_state(self, state: Dict[str, Any], timestamp: float = None):
        """Record a state in the current trajectory."""
        if self.current_trajectory is None:
            raise ValueError("No active trajectory. Call start_trajectory() first.")

        if timestamp is None:
            timestamp = len(self.current_trajectory['states'])

        self.current_trajectory['states'].append(state.copy())
        self.current_trajectory['timestamps'].append(timestamp)

    def end_trajectory(self) -> Dict[str, Any]:
        """End the current trajectory and store it."""
        if self.current_trajectory is None:
            raise ValueError("No active trajectory to end.")

        trajectory = self.current_trajectory.copy()
        self.trajectories.append(trajectory)
        self.current_trajectory = None

        # Compute trajectory statistics
        trajectory['statistics'] = self.compute_trajectory_statistics(trajectory)
        return trajectory

    def compute_trajectory_statistics(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistical properties of a trajectory."""
        states = trajectory['states']
        if len(states) < 2:
            return {}

        # Extract key quantities
        x_values = [s.get('x', 0) for s in states]
        bifurcation_indices = [s.get('B_t', 0) for s in states]
        clock_rates = [s.get('R', 0) for s in states]

        # Lyapunov exponent estimation (simplified)
        lyapunov_exp = self.estimate_lyapunov_exponent(states)

        # Orbit scaling parameter α
        scaling_param = self.compute_scaling_parameter(states)

        # Local bifurcation analysis
        bifurcation_events = self.detect_bifurcation_events(states)

        return {
            'length': len(states),
            'lyapunov_exponent': lyapunov_exp,
            'scaling_parameter': scaling_param,
            'bifurcation_events': bifurcation_events,
            'x_range': (min(x_values), max(x_values)),
            'clock_rate_stats': {
                'mean': np.mean(clock_rates),
                'std': np.std(clock_rates),
                'range': (min(clock_rates), max(clock_rates))
            }
        }

    def estimate_lyapunov_exponent(self, states: List[Dict[str, Any]]) -> float:
        """Estimate Lyapunov exponent from trajectory data."""
        if len(states) < 3:
            return 0.0

        # For AntClock, use curvature observables for Lyapunov calculation
        # Try different observables in order of preference
        observables = ['curvature', 'bifurcation_index', 'clock_rate', 'x']

        for obs_name in observables:
            if obs_name in states[0]:
                observable_values = [s.get(obs_name, 0) for s in states]
                break
        else:
            return 0.0

        if len(set(observable_values)) < 2:
            return 0.0

        # Simplified Lyapunov estimation using local divergence
        # λ ≈ (1/N) Σ log|d(obs_{i+1} - obs_i)/d(obs_i - obs_{i-1})|
        log_ratios = []
        for i in range(1, len(observable_values) - 1):
            prev_diff = abs(observable_values[i] - observable_values[i-1])
            next_diff = abs(observable_values[i+1] - observable_values[i])

            if prev_diff > 1e-12 and next_diff > 1e-12:
                ratio = next_diff / prev_diff
                if ratio > 0:
                    log_ratios.append(np.log(ratio))

        return np.mean(log_ratios) if log_ratios else 0.0

    def compute_scaling_parameter(self, states: List[Dict[str, Any]]) -> float:
        """Compute orbit scaling parameter α from trajectory."""
        if len(states) < 3:
            return 0.0

        # Use bifurcation index scaling as proxy for α
        bifurcation_indices = [s.get('B_t', 0) for s in states]

        # Look for power-law scaling in the trajectory
        # α ≈ d log(L)/d log(t) where L is some length scale
        lengths = []
        for i in range(1, len(states)):
            # Use range of x values as length scale
            x_subset = [s['x'] for s in states[:i+1]]
            length = max(x_subset) - min(x_subset) if x_subset else 1
            lengths.append(max(length, 1))

        if len(lengths) < 2:
            return 0.0

        # Fit power law: L ~ t^α
        times = np.arange(1, len(lengths) + 1)
        log_times = np.log(times)
        log_lengths = np.log(lengths)

        # Linear regression for slope
        if len(set(log_lengths)) > 1:  # Avoid constant values
            slope = np.polyfit(log_times, log_lengths, 1)[0]
            return slope
        else:
            return 0.0

    def detect_bifurcation_events(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect local bifurcation events in trajectory."""
        if len(states) < 3:
            return []

        bifurcation_events = []
        bifurcation_indices = [s.get('B_t', 0) for s in states]

        # Detect significant changes in bifurcation index
        for i in range(1, len(bifurcation_indices) - 1):
            prev_b = bifurcation_indices[i-1]
            curr_b = bifurcation_indices[i]
            next_b = bifurcation_indices[i+1]

            # Look for discontinuous jumps
            change = abs(curr_b - prev_b)
            if change > 1:  # Significant change threshold
                bifurcation_events.append({
                    'step': i,
                    'bifurcation_change': change,
                    'b_before': prev_b,
                    'b_after': curr_b,
                    'x_value': states[i].get('x', 0)
                })

        return bifurcation_events

    def get_trajectory(self, trajectory_id: int) -> Dict[str, Any]:
        """Retrieve a stored trajectory by ID."""
        if 0 <= trajectory_id < len(self.trajectories):
            return self.trajectories[trajectory_id]
        raise ValueError(f"Trajectory {trajectory_id} not found")

    def list_trajectories(self) -> List[Dict[str, Any]]:
        """Get summary of all stored trajectories."""
        return [{
            'id': i,
            'length': len(traj['states']),
            'initial_conditions': traj['initial_conditions'],
            'statistics': traj.get('statistics', {})
        } for i, traj in enumerate(self.trajectories)]


class WitnessMeasurement:
    """
    Witness/Measurement component <>: Observable computation and analysis.

    Handles:
    - Kolmogorov-Sinai entropy (h_KS) computation
    - Invariant measure estimation
    - Local observable statistics
    - Universal chaotic statistics across parameter families
    """

    def __init__(self):
        self.measurements = defaultdict(list)
        self.invariant_measures = {}

    def compute_kolmogorov_sinai_entropy(self, trajectory: Dict[str, Any],
                                        partition_count: int = 10) -> float:
        """
        Compute Kolmogorov-Sinai entropy from trajectory data.

        h_KS = lim_{n→∞} (1/n) H(ξ_0^n)
        where H is the entropy of the partition.
        """
        states = trajectory['states']
        if len(states) < 2:
            return 0.0

        # For AntClock, use curvature observables for entropy calculation
        # Try different observables in order of preference
        observables = ['curvature', 'bifurcation_index', 'clock_rate']

        for obs_name in observables:
            if obs_name in states[0]:
                observable_values = [s.get(obs_name, 0) for s in states]
                break
        else:
            # Fallback to bifurcation index
            observable_values = [s.get('B_t', 0) for s in states]

        # Create partition based on the observable
        if len(set(observable_values)) < 2:  # All same value
            return 0.0

        partition_edges = np.linspace(min(observable_values), max(observable_values),
                                    partition_count + 1)

        # Compute partition entropy for different block lengths
        max_block_length = min(20, len(states) // 2)
        entropies = []

        for block_length in range(1, max_block_length + 1):
            # Create blocks of observable values
            blocks = []
            for i in range(len(states) - block_length + 1):
                block_values = observable_values[i:i+block_length]
                # Quantize to partition
                quantized = []
                for val in block_values:
                    bin_idx = np.digitize(val, partition_edges) - 1
                    quantized.append(min(bin_idx, partition_count - 1))
                blocks.append(tuple(quantized))

            # Count block frequencies
            block_counts = defaultdict(int)
            for block in blocks:
                block_counts[block] += 1

            # Compute entropy
            total_blocks = len(blocks)
            entropy = 0.0
            for count in block_counts.values():
                p = count / total_blocks
                entropy -= p * np.log2(p) if p > 0 else 0

            # KS entropy contribution
            ks_contribution = entropy / block_length
            entropies.append(ks_contribution)

        # Return average KS entropy estimate
        return np.mean(entropies) if entropies else 0.0

    def estimate_invariant_measure(self, trajectory: Dict[str, Any],
                                 bins: int = 20) -> Dict[str, Any]:
        """
        Estimate the invariant measure from trajectory data.

        Returns density estimation for key observables.
        """
        states = trajectory['states']
        if len(states) < 2:
            return {}

        # Extract observables
        bifurcation_indices = [s.get('B_t', 0) for s in states]
        clock_rates = [s.get('R', 0) for s in states]
        x_values = [s.get('x', 0) for s in states]

        # Compute histograms/densities
        measure = {
            'bifurcation_density': self._compute_density(bifurcation_indices, bins),
            'clock_rate_density': self._compute_density(clock_rates, bins),
            'x_density': self._compute_density(x_values, bins),
            'support': {
                'bifurcation_range': (min(bifurcation_indices), max(bifurcation_indices)),
                'clock_rate_range': (min(clock_rates), max(clock_rates)),
                'x_range': (min(x_values), max(x_values))
            }
        }

        return measure

    def _compute_density(self, values: List[float], bins: int) -> Dict[str, Any]:
        """Compute probability density from values."""
        if not values:
            return {'bins': [], 'densities': [], 'bin_centers': []}

        hist, bin_edges = np.histogram(values, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            'bins': bin_edges.tolist(),
            'densities': hist.tolist(),
            'bin_centers': bin_centers.tolist()
        }

    def measure_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Compute all measurements for a trajectory."""
        measurement = {
            'trajectory_id': trajectory['id'],
            'kolmogorov_sinai_entropy': self.compute_kolmogorov_sinai_entropy(trajectory),
            'invariant_measure': self.estimate_invariant_measure(trajectory),
            'length': len(trajectory['states']),
            'scaling_parameter': trajectory['statistics'].get('scaling_parameter', 0),
            'lyapunov_exponent': trajectory['statistics'].get('lyapunov_exponent', 0)
        }

        # Store measurement
        trajectory_id = trajectory['id']
        self.measurements[trajectory_id].append(measurement)

        return measurement

    def analyze_parameter_family(self, trajectories: List[Dict[str, Any]],
                               parameter_name: str) -> Dict[str, Any]:
        """
        Analyze measurements across a family of trajectories parameterized by some value.

        This implements <> + {} = (δ, h_KS across families)
        """
        if not trajectories:
            return {}

        # Extract parameter values and measurements
        parameters = []
        h_ks_values = []
        lyapunov_values = []
        scaling_params = []

        for traj in trajectories:
            param_value = traj['initial_conditions'].get(parameter_name, 0)
            measurement = self.measure_trajectory(traj)

            parameters.append(param_value)
            h_ks_values.append(measurement['kolmogorov_sinai_entropy'])
            lyapunov_values.append(measurement['lyapunov_exponent'])
            scaling_params.append(measurement['scaling_parameter'])

        # Compute statistics across parameter family
        analysis = {
            'parameter_name': parameter_name,
            'parameter_range': (min(parameters), max(parameters)),
            'kolmogorov_sinai_stats': {
                'mean': np.mean(h_ks_values),
                'std': np.std(h_ks_values),
                'max': max(h_ks_values),
                'min': min(h_ks_values)
            },
            'lyapunov_stats': {
                'mean': np.mean(lyapunov_values),
                'std': np.std(lyapunov_values),
                'max': max(lyapunov_values),
                'min': min(lyapunov_values)
            },
            'scaling_stats': {
                'mean': np.mean(scaling_params),
                'std': np.std(scaling_params),
                'correlation_with_param': np.corrcoef(parameters, scaling_params)[0,1] if len(parameters) > 1 else 0
            },
            'trajectory_count': len(trajectories)
        }

        return analysis


class ChaosAnalysisFramework:
    """
    Complete chaos analysis framework integrating Memory/History and Witness/Measurement.

    Four analysis quadrants:
    1. [] + () = (α, λ): Local trajectory analysis
    2. [] + {} = (δ, λ): Global bifurcation analysis
    3. <> + () = (α, h_KS, invariant measure): Local measurement analysis
    4. <> + {} = (δ, h_KS): Global measurement analysis
    """

    def __init__(self):
        self.memory = MemoryHistory()
        self.witness = WitnessMeasurement()
        self.analysis_results = {}

    def analyze_single_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Local trajectory analysis: [] + () = (α, λ)

        Returns scaling parameter α and Lyapunov exponent λ for one trajectory.
        """
        # Memory analysis
        stats = trajectory.get('statistics', self.memory.compute_trajectory_statistics(trajectory))

        # Measurement analysis
        measurement = self.witness.measure_trajectory(trajectory)

        result = {
            'quadrant': 'local_trajectory',
            'scaling_parameter': stats.get('scaling_parameter', 0),
            'lyapunov_exponent': stats.get('lyapunov_exponent', 0),
            'kolmogorov_sinai_entropy': measurement['kolmogorov_sinai_entropy'],
            'invariant_measure': measurement['invariant_measure'],
            'trajectory_length': len(trajectory['states'])
        }

        self.analysis_results[f"trajectory_{trajectory['id']}"] = result
        return result

    def analyze_bifurcation_family(self, trajectories: List[Dict[str, Any]],
                                 parameter_name: str = 'control_parameter') -> Dict[str, Any]:
        """
        Global bifurcation analysis: [] + {} = (δ, λ)

        Returns Lyapunov exponents λ over parameter family with bifurcation parameter δ.
        """
        if not trajectories:
            return {}

        # Extract bifurcation data across family
        parameters = []
        lyapunov_values = []
        bifurcation_events = []

        for traj in trajectories:
            param = traj['initial_conditions'].get(parameter_name, 0)
            stats = traj.get('statistics', self.memory.compute_trajectory_statistics(traj))

            parameters.append(param)
            lyapunov_values.append(stats.get('lyapunov_exponent', 0))
            bifurcation_events.extend(stats.get('bifurcation_events', []))

        # Find bifurcation points (parameter values where λ changes significantly)
        bifurcation_points = []
        if len(parameters) > 1 and len(lyapunov_values) > 1:
            sorted_indices = np.argsort(parameters)
            sorted_params = np.array(parameters)[sorted_indices]
            sorted_lyapunov = np.array(lyapunov_values)[sorted_indices]

            # Detect significant changes in Lyapunov exponent
            lyapunov_changes = np.abs(np.diff(sorted_lyapunov))
            threshold = np.mean(lyapunov_changes) + 2 * np.std(lyapunov_changes)

            for i, change in enumerate(lyapunov_changes):
                if change > threshold:
                    bifurcation_points.append({
                        'parameter': sorted_params[i],
                        'lyapunov_change': change,
                        'lyapunov_before': sorted_lyapunov[i],
                        'lyapunov_after': sorted_lyapunov[i+1]
                    })

        result = {
            'quadrant': 'global_bifurcation',
            'parameter_name': parameter_name,
            'parameter_range': (min(parameters), max(parameters)) if parameters else (0, 0),
            'lyapunov_over_family': {
                'values': lyapunov_values,
                'parameters': parameters,
                'mean': np.mean(lyapunov_values),
                'std': np.std(lyapunov_values)
            },
            'bifurcation_points': bifurcation_points,
            'total_bifurcations': len(bifurcation_points)
        }

        self.analysis_results[f"bifurcation_family_{parameter_name}"] = result
        return result

    def analyze_measurement_family(self, trajectories: List[Dict[str, Any]],
                                 parameter_name: str = 'control_parameter') -> Dict[str, Any]:
        """
        Local measurement analysis: <> + () = (α, h_KS, invariant measure)

        Returns entropy h_KS and invariant measures over parameter family.
        """
        analysis = self.witness.analyze_parameter_family(trajectories, parameter_name)
        analysis['quadrant'] = 'local_measurement'

        self.analysis_results[f"measurement_family_{parameter_name}"] = analysis
        return analysis

    def correlation_analysis(self, x: List[float], y: List[float]) -> float:
        """
        Compute correlation coefficient between two observables.

        Returns Pearson correlation coefficient.
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        return np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0.0

    def run_complete_analysis(self, trajectories: List[Dict[str, Any]],
                            parameter_name: str = 'control_parameter') -> Dict[str, Any]:
        """
        Run complete four-quadrant analysis on trajectory family.
        """
        results = {
            'individual_trajectories': [self.analyze_single_trajectory(traj) for traj in trajectories],
            'bifurcation_analysis': self.analyze_bifurcation_family(trajectories, parameter_name),
            'measurement_analysis': self.analyze_measurement_family(trajectories, parameter_name),
            'summary': self._create_analysis_summary(trajectories, parameter_name)
        }

        return results

    def _create_analysis_summary(self, trajectories: List[Dict[str, Any]],
                               parameter_name: str) -> Dict[str, Any]:
        """Create summary of the complete analysis."""
        if not trajectories:
            return {}

        # Extract key metrics across all trajectories
        all_lyapunov = []
        all_entropy = []
        all_scaling = []

        for traj in trajectories:
            stats = traj.get('statistics', {})
            measurement = self.witness.measure_trajectory(traj)

            all_lyapunov.append(stats.get('lyapunov_exponent', 0))
            all_entropy.append(measurement['kolmogorov_sinai_entropy'])
            all_scaling.append(stats.get('scaling_parameter', 0))

        return {
            'total_trajectories': len(trajectories),
            'lyapunov_range': (min(all_lyapunov), max(all_lyapunov)) if all_lyapunov else (0, 0),
            'entropy_range': (min(all_entropy), max(all_entropy)) if all_entropy else (0, 0),
            'scaling_range': (min(all_scaling), max(all_scaling)) if all_scaling else (0, 0),
            'chaos_classification': self._classify_chaos_regime(all_lyapunov, all_entropy)
        }

    def _classify_chaos_regime(self, lyapunov_values: List[float],
                             entropy_values: List[float]) -> str:
        """Classify the overall chaotic regime based on Lyapunov and entropy."""
        if not lyapunov_values or not entropy_values:
            return "insufficient_data"

        mean_lyapunov = np.mean(lyapunov_values)
        mean_entropy = np.mean(entropy_values)

        if mean_lyapunov < -0.1:
            return "stable"
        elif mean_lyapunov > 0.1 and mean_entropy < 0.5:
            return "weak_chaos"
        elif mean_lyapunov > 0.5 and mean_entropy > 1.0:
            return "strong_chaos"
        else:
            return "transition_regime"
