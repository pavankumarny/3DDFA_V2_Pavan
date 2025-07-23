#!/usr/bin/env python3
"""
3DDFA_V2 Experiment 2 Pitch vs Time Plotter (PyTorch Data)
==========================================================

Plot pitch vs time for three videos (75°, 90°, 115°) processed with PyTorch mode.
Focuses specifically on pitch stability at angles closer to 90°.

Usage:
    python3 plot_experiment_2_pitch_vs_time_pytorch.py
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import time
from pathlib import Path

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

def plot_experiment_2_pitch_vs_time():
    """Create pitch vs time plots for three videos (75°, 90°, 115°)"""
    
    # Define the CSV files for the three angles (PyTorch processed)
    csv_files = [
        'examples/experiment/csv/75_pose_data_pytorch.csv',
        'examples/experiment/csv/90_pose_data_pytorch.csv', 
        'examples/experiment/csv/115_pose_data_pytorch.csv'
    ]
    
    # Define angle labels for each video
    angle_labels = ['75°', '90°', '115°']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Add supertitle
    plt.suptitle('Experiment 2: Pitch Stability at Angles Closer to 90° (PyTorch Mode)\nPitch vs Time Analysis', 
                fontsize=16, y=0.98)
    
    # Plot each CSV file
    for i, (csv_file, angle_label) in enumerate(zip(csv_files, angle_labels)):
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found: {csv_file}")
            continue
            
        data = load_csv_data(csv_file)
        timestamps = np.array(data['timestamp'])
        pitch = np.array(data['pitch'])
        
        # Skip if no data
        if len(pitch) == 0:
            print(f"Warning: No data in CSV file: {csv_file}")
            continue
        
        # Calculate zero-mean pitch
        pitch_zero_mean = pitch - pitch.mean()
        
        # Plot pitch vs time
        axes[i].plot(timestamps, pitch_zero_mean, 'b-', linewidth=2, alpha=0.8)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Pitch (deg)')
        axes[i].set_title(f'{angle_label} - Pitch Detection, Zero-Mean (up is negative)')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.7)
        
        # Add statistics text
        pitch_std = np.std(pitch_zero_mean)
        pitch_range = pitch_zero_mean.max() - pitch_zero_mean.min()
        mean_pitch = pitch.mean()
        stats_text = f'Mean: {mean_pitch:.1f}° | Std Dev: {pitch_std:.1f}° | Range: {pitch_range:.1f}° | Frames: {len(pitch)}'
        axes[i].text(0.02, 0.98, stats_text, 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set y-axis limits for better comparison
        y_max = max(abs(pitch_zero_mean.min()), abs(pitch_zero_mean.max()))
        axes[i].set_ylim(-y_max*1.1, y_max*1.1)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save the plot
    save_dir = os.path.join('examples', 'experiment', 'plot')
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = int(time.time())
    save_path = os.path.join(save_dir, f'experiment_2_pitch_vs_time_pytorch_{timestamp}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Experiment 2 pitch vs time plot (PyTorch) saved to: {save_path}")
    
    # Print analysis to console
    print("\n" + "="*60)
    print("PITCH ANALYSIS RESULTS (PyTorch Mode)")
    print("="*60)
    
    for csv_file, angle_label in zip(csv_files, angle_labels):
        if os.path.exists(csv_file):
            data = load_csv_data(csv_file)
            pitch = np.array(data['pitch'])
            
            if len(pitch) > 0:
                print(f"\n{angle_label} Analysis:")
                print(f"  Frames processed: {len(pitch)}")
                print(f"  Mean pitch: {pitch.mean():.1f}°")
                print(f"  Pitch std dev: {pitch.std():.1f}°")
                print(f"  Pitch range: [{pitch.min():.1f}°, {pitch.max():.1f}°]")
                
                # Calculate frame-to-frame stability
                pitch_diff = np.abs(np.diff(pitch))
                print(f"  Max frame jump: {pitch_diff.max():.1f}°")
                print(f"  Large jumps (>5°): {np.sum(pitch_diff > 5)}")
    
    plt.show()

if __name__ == "__main__":
    plot_experiment_2_pitch_vs_time() 