#!/usr/bin/env python3
"""
3DDFA_V2 Pitch Zero-Mean Plotter (No Pandas Required)
====================================================

Visualize zero-mean pitch data from a CSV file.
Uses only matplotlib and numpy (no pandas dependency).

Usage:
    python3 plot_pitch_zero_mean.py [csv_file]
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import sys
import glob

def load_csv_data(csv_file):
    """Load CSV data using only built-in csv module"""
    data = {
        'frame': [],
        'timestamp': [],
        'pitch': []
    }
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['frame'].append(int(row['frame']))
            data['timestamp'].append(float(row['timestamp']))
            data['pitch'].append(float(row['pitch']))
    return data

def plot_pitch_zero_mean(csv_file):
    print(f"📊 Loading data from: {csv_file}")
    data = load_csv_data(csv_file)
    timestamps = np.array(data['timestamp'])
    pitch = np.array(data['pitch'])
    pitch_zero_mean = pitch - pitch.mean()

    print(f"📈 Data summary:")
    print(f"   Total frames: {len(timestamps)}")
    print(f"   Duration: {timestamps[-1]:.2f} seconds")
    print(f"   Pitch range: {pitch.min():.1f}° to {pitch.max():.1f}°")
    print(f"   Zero-mean pitch range: {pitch_zero_mean.min():.1f}° to {pitch_zero_mean.max():.1f}°")
    print(f"   Mean pitch: {pitch.mean():.2f}°")

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, pitch_zero_mean, 'r-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (deg)')
    plt.title(f'Pitch Detection, Zero-Mean (up is negative)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    plt.text(0.02, 0.98, f'Range: {pitch_zero_mean.min():.1f}° to {pitch_zero_mean.max():.1f}°\nMean: {pitch_zero_mean.mean():.1f}°', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    # Save plot in the same directory as the CSV file
    csv_dir = os.path.dirname(os.path.abspath(csv_file))
    csv_base = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = os.path.join(csv_dir, f'{csv_base}_pitch_zero_mean_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved: {output_file}")
    plt.show()

def main():
    print("🎨 3DDFA_V2 Pitch Zero-Mean Plotter")
    print("=" * 50)
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        possible_files = []
        possible_files.extend(glob.glob("*.csv"))
        if os.path.exists("results"):
            possible_files.extend(glob.glob("results/*.csv"))
        if os.path.exists("examples/results"):
            possible_files.extend(glob.glob("examples/results/*.csv"))
        if not possible_files:
            print("❌ No CSV files found. Please specify a file:")
            print("   python3 plot_pitch_zero_mean.py your_file.csv")
            return
        csv_file = possible_files[0]
        print(f"📁 Found CSV files: {len(possible_files)}")
        print(f"🎯 Using: {csv_file}")
        print()
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    try:
        print("📊 Creating pitch zero-mean plot...")
        plot_pitch_zero_mean(csv_file)
        print("🎉 Plotting complete!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main() 