#!/usr/bin/env python3
"""
Visual Tracer Demonstration
Shows how to visualize spectral trajectories from the Zero-image μVM
"""

import sys
import json
import csv

def load_csv(filename):
    """Load trace data from CSV file"""
    data = {'rho_real': [], 'rho_imag': [], 'depth': [], 'monodromy': [], 'timestamp': []}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['rho_real'].append(float(row['rho_real']))
            data['rho_imag'].append(float(row['rho_imag']))
            data['depth'].append(int(row['depth']))
            data['monodromy'].append(float(row['monodromy']))
            data['timestamp'].append(int(row['timestamp']))
    return data

def print_data(data):
    """Print trace data"""
    print("\n=== Spectral Trajectory Data ===\n")
    print(f"Total frames: {len(data['rho_real'])}")
    for i in range(min(10, len(data['rho_real']))):
        print(f"Frame {i}: ρ={data['rho_real'][i]:.6f}+{data['rho_imag'][i]:.6f}i, depth={data['depth'][i]}, t={data['timestamp'][i]}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 trace_visualization.py trace.csv")
        sys.exit(1)
    data = load_csv(sys.argv[1])
    print_data(data)
