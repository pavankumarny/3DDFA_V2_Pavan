#!/usr/bin/env python3
"""
3DDFA_V2 Pose Data Plotter (No Pandas Required)
===============================================

Visualize time series pose data (pitch, yaw, roll) from CSV files
Uses only matplotlib and numpy (no pandas dependency).

Usage:
    python3 plot_pose_simple.py [csv_file]
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
        'pitch': [],
        'yaw': [],
        'roll': []
    }
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['frame'].append(int(row['frame']))
            data['timestamp'].append(float(row['timestamp']))
            data['pitch'].append(float(row['pitch']))
            data['yaw'].append(float(row['yaw']))
            data['roll'].append(float(row['roll']))
    
    return data

def plot_pose_data(csv_file):
    """Create comprehensive pose data visualization"""
    print(f"📊 Loading data from: {csv_file}")
    
    # Load data
    data = load_csv_data(csv_file)
    
    # Convert to numpy arrays for easier manipulation
    timestamps = np.array(data['timestamp'])
    pitch = np.array(data['pitch'])
    yaw = np.array(data['yaw'])
    roll = np.array(data['roll'])
    
    # Print statistics
    print(f"📈 Data summary:")
    print(f"   Total frames: {len(timestamps)}")
    print(f"   Duration: {timestamps[-1]:.2f} seconds")
    print(f"   Pitch range: {pitch.min():.1f}° to {pitch.max():.1f}°")
    print(f"   Yaw range: {yaw.min():.1f}° to {yaw.max():.1f}°")
    print(f"   Roll range: {roll.min():.1f}° to {roll.max():.1f}°")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'3DDFA_V2 Pose Analysis: {os.path.basename(csv_file)}', fontsize=16, fontweight='bold')
    
    # Plot 1: All angles over time
    ax1 = axes[0]
    ax1.plot(timestamps, pitch, 'r-', label='Pitch (Up/Down)', linewidth=2)
    ax1.plot(timestamps, yaw, 'g-', label='Yaw (Left/Right)', linewidth=2)
    ax1.plot(timestamps, roll, 'b-', label='Roll (Tilt)', linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Pose Angles Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: Pitch over time (zero-mean)
    ax2 = axes[1]
    pitch_zero_mean = pitch - pitch.mean()
    ax2.plot(timestamps, pitch_zero_mean, 'r-', linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Pitch (degrees) - Zero Mean')
    ax2.set_title('Pitch: Zero-Mean (Up/Down Movement)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    ax2.text(0.02, 0.98, f'Range: {pitch_zero_mean.min():.1f}° to {pitch_zero_mean.max():.1f}°\nMean: {pitch_zero_mean.mean():.1f}°', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = csv_file.replace('.csv', '_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved: {output_file}")
    
    # Show plot
    plt.show()


def main():
    print("🎨 3DDFA_V2 Pose Data Plotter")
    print("=" * 50)
    
    # Get CSV file
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Look for CSV files
        possible_files = []
        
        # Check current directory
        possible_files.extend(glob.glob("*.csv"))
        
        # Check results directory
        if os.path.exists("results"):
            possible_files.extend(glob.glob("results/*.csv"))
        
        # Check examples/results directory
        if os.path.exists("examples/results"):
            possible_files.extend(glob.glob("examples/results/*.csv"))
        
        if not possible_files:
            print("❌ No CSV files found. Please specify a file:")
            print("   python3 plot_pose_simple.py your_file.csv")
            return
        
        csv_file = possible_files[0]
        print(f"📁 Found CSV files: {len(possible_files)}")
        print(f"🎯 Using: {csv_file}")
        print()
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    try:
        # Create time series plot
        print("📊 Creating time series plot...")
        plot_pose_data(csv_file)
        
        print("🎉 Plotting complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
