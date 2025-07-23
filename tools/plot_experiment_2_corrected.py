#!/usr/bin/env python3
"""
3DDFA_V2 Experiment 2 Pitch vs Time Plotter (CORRECTED)
======================================================

Plot pitch vs time for three videos (75°, 90°, 115°) with CORRECTED pose extraction.
Shows how pitch varies dramatically when moving head up/down.

Usage:
    python3 plot_experiment_2_corrected.py
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

def plot_experiment_2_corrected():
    """Create corrected pitch vs time plots for three videos (75°, 90°, 115°)"""
    
    # Define the CSV files for the three angles (FINAL ALGORITHM data)
    csv_files = [
        'examples/experiment/final-csv/75_pose_data.csv',
        'examples/experiment/final-csv/90_pose_data.csv', 
        'examples/experiment/final-csv/115_pose_data.csv'
    ]
    
    # Define angle labels for each video
    angle_labels = ['75°', '90°', '115°']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Add supertitle
    plt.suptitle('Pitch ROM acquisition at different side view angles', 
                 fontsize=24, y=0.94)
    
    # Define angle labels for cleaner titles
    angle_mapping = {
        '75': '75 deg rotation in Yaw from frontal',
        '90': '90 deg rotation in Yaw from frontal', 
        '115': '115 deg rotation in Yaw from frontal'
    }
    
    # Plot each CSV file in its own subplot
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
        
        # REVERSE PITCH SIGN: Make positive = up, negative = down (more intuitive)
        pitch = -pitch  # Flip the sign so positive = looking up, negative = looking down
        
        # Calculate zero-mean pitch
        pitch_zero_mean = pitch - pitch.mean()
        
        # Plot in subplot
        axes[i].plot(timestamps, pitch_zero_mean, 'r-', linewidth=2)
        axes[i].set_xlabel('Time (s)', fontsize=16)
        axes[i].set_ylabel('Pitch (deg)', fontsize=16)
        
        # Get angle from filename and use mapping for clean title
        angle_num = angle_label.replace('°', '')
        clean_title = angle_mapping.get(angle_num, f'{angle_label} rotation in Yaw from frontal')
        axes[i].set_title(clean_title, fontsize=18)
        
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.7)
        
        # Increase tick label font sizes and use broader intervals
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        
        # Set broader y-axis tick intervals to avoid crowding
        y_min, y_max = axes[i].get_ylim()
        y_range = y_max - y_min
        if y_range > 50:
            tick_interval = 20
        elif y_range > 25:
            tick_interval = 10
        else:
            tick_interval = 5
        axes[i].yaxis.set_major_locator(plt.MultipleLocator(tick_interval))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save the plot
    save_dir = os.path.join('examples', 'experiment', 'plot')
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = int(time.time())
    save_path = os.path.join(save_dir, f'experiment_2_ORIGINAL_algorithm_pitch_analysis_{timestamp}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ ORIGINAL Algorithm Experiment 2 plot saved to: {save_path}")
    
    # Print analysis to console
    print("\n" + "="*60)
    print("ORIGINAL ALGORITHM PITCH ANALYSIS RESULTS")
    print("="*60)
    print("✅ USING PURE 3DDFA_V2 ALGORITHM (NO MODIFICATIONS)!")
    print("📊 PITCH SIGN REVERSED: Positive = Up, Negative = Down")
    
    for csv_file, angle_label in zip(csv_files, angle_labels):
        if os.path.exists(csv_file):
            data = load_csv_data(csv_file)
            pitch = np.array(data['pitch'])
            
            if len(pitch) > 0:
                # REVERSE PITCH SIGN: Make positive = up, negative = down (more intuitive)
                pitch = -pitch  # Flip the sign so positive = looking up, negative = looking down
                
                print(f"\n{angle_label} Analysis:")
                print(f"  Frames processed: {len(pitch)}")
                print(f"  Mean pitch: {pitch.mean():.1f}°")
                print(f"  Pitch std dev: {pitch.std():.1f}°")
                print(f"  Pitch range: [{pitch.min():.1f}°, {pitch.max():.1f}°]")
                print(f"  Total movement: {pitch.max() - pitch.min():.1f}° ✅")
                
                # Movement analysis
                if pitch.max() - pitch.min() > 30:
                    print(f"  📈 Excellent pitch movement detected!")
                elif pitch.max() - pitch.min() > 15:
                    print(f"  📊 Good pitch movement detected!")
                else:
                    print(f"  ⚠️  Limited pitch movement")
    
    plt.show()

if __name__ == "__main__":
    plot_experiment_2_corrected() 