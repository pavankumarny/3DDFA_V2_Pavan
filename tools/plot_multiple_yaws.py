#!/usr/bin/env python3
"""
3DDFA_V2 Multi-CSV Yaw Zero-Mean Plotter (No Pandas Required)
============================================================

Visualize zero-mean yaw data from multiple CSV files in separate subplots.
Uses only matplotlib and numpy (no pandas dependency).

Usage:
    python3 plot_multiple_yaws.py [csv_file1] [csv_file2] ...
    python3 plot_multiple_yaws.py --name "custom_name" [csv_file1] [csv_file2] ...
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import sys
import time
from pathlib import Path

def load_csv_data(csv_file):
    """Load CSV data using only built-in csv module"""
    data = {
        'frame': [],
        'timestamp': [],
        'yaw': []
    }
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['frame'].append(int(row['frame']))
            data['timestamp'].append(float(row['timestamp']))
            data['yaw'].append(float(row['yaw']))
    return data

def plot_multiple_yaws(csv_files, save_dir, plot_name=None, supertitle=None):
    """Create separate subplots for each CSV file"""
    # Filter out empty CSV files
    valid_csv_files = []
    for csv_file in csv_files:
        data = load_csv_data(csv_file)
        if len(data['yaw']) > 0:  # Check if there's actual data
            valid_csv_files.append(csv_file)
        else:
            print(f"Warning: Skipping empty CSV file: {csv_file}")
    
    if not valid_csv_files:
        print("Error: No valid CSV files with data found!")
        return
    
    n_files = len(valid_csv_files)
    
    # Calculate subplot layout
    if n_files <= 2:
        rows, cols = 1, n_files
    elif n_files <= 4:
        rows, cols = 2, 2
    elif n_files <= 6:
        rows, cols = 2, 3
    elif n_files <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    
    # Add supertitle if provided
    if supertitle:
        plt.suptitle(supertitle, fontsize=18, y=0.98)
    
    # Handle single subplot case
    if n_files == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each CSV file in its own subplot
    for i, csv_file in enumerate(valid_csv_files):
        if i >= len(axes):
            break
            
        data = load_csv_data(csv_file)
        timestamps = np.array(data['timestamp'])
        yaw = np.array(data['yaw'])
        
        # Skip if no data
        if len(yaw) == 0:
            continue
            
        yaw_zero_mean = yaw - yaw.mean()
        
        # Get filename for title
        filename = Path(csv_file).stem
        
        # Plot in subplot
        axes[i].plot(timestamps, yaw_zero_mean, 'b-', linewidth=2)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Yaw (deg)')
        axes[i].set_title(f'{filename} - Yaw Detection, Zero-Mean')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.7)
        
        # Add statistics text
        stats_text = f'Range: {yaw_zero_mean.min():.1f}° to {yaw_zero_mean.max():.1f}°\nMean: {yaw_zero_mean.mean():.1f}°'
        axes[i].text(0.02, 0.98, stats_text, 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_files, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout to make room for supertitle
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(save_dir, exist_ok=True)
    
    # Create unique filename
    if plot_name:
        save_path = os.path.join(save_dir, f'{plot_name}_yaw_zero_mean_plot.png')
    else:
        timestamp = int(time.time())
        save_path = os.path.join(save_dir, f'yaw_zero_mean_plot_{timestamp}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_multiple_yaws.py [--name plot_name] [--supertitle supertitle] <csv1> <csv2> ...")
        sys.exit(1)
    
    # Parse arguments
    csv_files = []
    plot_name = None
    supertitle = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--name' and i + 1 < len(sys.argv):
            plot_name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--supertitle' and i + 1 < len(sys.argv):
            supertitle = sys.argv[i + 1]
            i += 2
        else:
            csv_files.append(sys.argv[i])
            i += 1
    
    if not csv_files:
        print("Error: No CSV files specified!")
        sys.exit(1)
    
    save_dir = os.path.join('examples', 'experiment', 'plot')
    plot_multiple_yaws(csv_files, save_dir, plot_name, supertitle)

if __name__ == "__main__":
    main() 