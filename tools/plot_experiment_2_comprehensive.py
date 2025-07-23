#!/usr/bin/env python3
"""
3DDFA_V2 Experiment 2 Comprehensive Analysis
============================================

Analyze pitch stability at angles close to 90° (75°, 90°, 115°)
Creates detailed plots with statistics and stability analysis.
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

def calculate_stability_metrics(values, window_size=10):
    """Calculate stability metrics for angle data"""
    # Rolling standard deviation
    rolling_std = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(values), i + window_size // 2 + 1)
        window = values[start_idx:end_idx]
        rolling_std.append(np.std(window))
    
    # Calculate jumps (frame-to-frame differences)
    jumps = np.abs(np.diff(values))
    
    return {
        'rolling_std': np.array(rolling_std),
        'jumps': jumps,
        'mean_stability': np.mean(rolling_std),
        'max_jump': np.max(jumps) if len(jumps) > 0 else 0,
        'jump_count': np.sum(jumps > 10)  # Count jumps > 10 degrees
    }

def plot_comprehensive_analysis():
    """Create comprehensive analysis plots for Experiment 2"""
    
    # Define the CSV files for the three angles
    csv_files = [
        'examples/experiment/csv/75_pose_data_fixed.csv',
        'examples/experiment/csv/90_pose_data_fixed.csv', 
        'examples/experiment/csv/115_pose_data_fixed.csv'
    ]
    
    # Define angle labels
    angle_labels = ['75°', '90°', '115°']
    colors = ['blue', 'red', 'green']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Main title
    fig.suptitle('Experiment 2: Comprehensive Pitch Analysis at Angles Close to 90°\n' + 
                 'Effect of Yaw Angle on Pitch Estimation Stability', fontsize=16, y=0.98)
    
    # Subplot 1: Pitch vs Time (all three angles)
    ax1 = plt.subplot(3, 2, (1, 2))
    
    all_data = []
    for i, (csv_file, angle_label, color) in enumerate(zip(csv_files, angle_labels, colors)):
        if not os.path.exists(csv_file):
            print(f"Warning: CSV file not found: {csv_file}")
            continue
            
        data = load_csv_data(csv_file)
        all_data.append(data)
        
        timestamps = np.array(data['timestamp'])
        pitch = np.array(data['pitch'])
        
        # Calculate zero-mean pitch
        pitch_zero_mean = pitch - pitch.mean()
        
        # Plot
        ax1.plot(timestamps, pitch_zero_mean, color=color, linewidth=2, 
                alpha=0.8, label=f'{angle_label} (μ={pitch.mean():.1f}°)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pitch - Mean (degrees)')
    ax1.set_title('Pitch Deviation Over Time (Zero-Mean)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Subplots 2-4: Individual angle analysis
    for idx, (data, angle_label, color) in enumerate(zip(all_data, angle_labels, colors)):
        ax = plt.subplot(3, 2, 3 + idx)
        
        timestamps = np.array(data['timestamp'])
        pitch = np.array(data['pitch'])
        yaw = np.array(data['yaw'])
        roll = np.array(data['roll'])
        
        # Plot all three angles
        ax.plot(timestamps, pitch, 'r-', linewidth=2, alpha=0.8, label='Pitch')
        ax.plot(timestamps, yaw, 'g-', linewidth=1, alpha=0.6, label='Yaw')
        ax.plot(timestamps, roll, 'b-', linewidth=1, alpha=0.6, label='Roll')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'{angle_label} - All Angles')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate and display statistics
        pitch_metrics = calculate_stability_metrics(pitch)
        stats_text = (f'Pitch Stats:\n'
                     f'Mean: {pitch.mean():.1f}°\n'
                     f'Std: {pitch.std():.1f}°\n'
                     f'Range: {pitch.max()-pitch.min():.1f}°\n'
                     f'Stability: {pitch_metrics["mean_stability"]:.1f}°\n'
                     f'Max Jump: {pitch_metrics["max_jump"]:.1f}°\n'
                     f'Jumps >10°: {pitch_metrics["jump_count"]}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Subplot 5: Stability comparison
    ax5 = plt.subplot(3, 2, 5)
    
    stability_means = []
    stability_stds = []
    max_jumps = []
    
    for data, angle_label in zip(all_data, angle_labels):
        pitch = np.array(data['pitch'])
        metrics = calculate_stability_metrics(pitch)
        stability_means.append(metrics['mean_stability'])
        stability_stds.append(pitch.std())
        max_jumps.append(metrics['max_jump'])
    
    x = np.arange(len(angle_labels))
    width = 0.25
    
    ax5.bar(x - width, stability_means, width, label='Rolling Std', color='blue', alpha=0.7)
    ax5.bar(x, stability_stds, width, label='Overall Std', color='red', alpha=0.7)
    ax5.bar(x + width, max_jumps, width, label='Max Jump', color='green', alpha=0.7)
    
    ax5.set_xlabel('Yaw Angle')
    ax5.set_ylabel('Degrees')
    ax5.set_title('Stability Metrics Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(angle_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Summary text
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    summary_text = "Key Findings:\n\n"
    
    # Analyze the data
    if len(all_data) == 3:
        # Find most stable angle
        most_stable_idx = np.argmin(stability_means)
        least_stable_idx = np.argmax(stability_means)
        
        summary_text += f"• Most stable: {angle_labels[most_stable_idx]} " \
                       f"(stability={stability_means[most_stable_idx]:.1f}°)\n"
        summary_text += f"• Least stable: {angle_labels[least_stable_idx]} " \
                       f"(stability={stability_means[least_stable_idx]:.1f}°)\n\n"
        
        # Check for gimbal lock effects
        if stability_means[1] > stability_means[0] and stability_means[1] > stability_means[2]:
            summary_text += "• 90° shows highest instability\n  (possible gimbal lock effects)\n\n"
        
        # Analyze pitch ranges
        for data, angle, metrics in zip(all_data, angle_labels, stability_means):
            pitch = np.array(data['pitch'])
            summary_text += f"• {angle}: Pitch range {pitch.min():.0f}° to {pitch.max():.0f}°\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             verticalalignment='top', fontsize=11)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    save_dir = os.path.join('examples', 'experiment', 'plot')
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = int(time.time())
    save_path = os.path.join(save_dir, f'experiment_2_comprehensive_analysis_{timestamp}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comprehensive analysis plot saved to: {save_path}")
    
    # Print detailed analysis to console
    print("\n" + "="*60)
    print("DETAILED ANALYSIS RESULTS")
    print("="*60)
    
    for data, angle in zip(all_data, angle_labels):
        pitch = np.array(data['pitch'])
        yaw = np.array(data['yaw'])
        roll = np.array(data['roll'])
        
        print(f"\n{angle} Analysis:")
        print(f"  Frames processed: {len(pitch)}")
        print(f"  Pitch: mean={pitch.mean():.1f}°, std={pitch.std():.1f}°, " \
              f"range=[{pitch.min():.1f}°, {pitch.max():.1f}°]")
        print(f"  Yaw:   mean={yaw.mean():.1f}°, std={yaw.std():.1f}°")
        print(f"  Roll:  mean={roll.mean():.1f}°, std={roll.std():.1f}°")
        
        metrics = calculate_stability_metrics(pitch)
        print(f"  Stability: {metrics['mean_stability']:.2f}° (lower is better)")
        print(f"  Max jump: {metrics['max_jump']:.1f}°")
        print(f"  Large jumps (>10°): {metrics['jump_count']}")
    
    plt.show()

if __name__ == "__main__":
    plot_comprehensive_analysis() 