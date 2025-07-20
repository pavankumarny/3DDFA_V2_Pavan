#!/usr/bin/env python3
"""
3DDFA_V2 Pitch vs Yaw Analysis for Experiment 1
===============================================

Plot pitch vs yaw to identify the yaw threshold where pitch estimates become unstable.
This is specifically designed for Experiment 1 where P (pitch) is kept static while Y (yaw) varies.
Creates one figure with 3 subplots - one for each scenario.

Usage:
    python3 plot_pitch_vs_yaw.py [csv_file1] [csv_file2] [csv_file3]
    python3 plot_pitch_vs_yaw.py --threshold 15 [csv_file1] [csv_file2] [csv_file3]
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import sys
import argparse
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

def detect_instability_breakpoint(pitch_values, window_size=10, threshold_std=5.0):
    """Detect where pitch becomes unstable (break point)"""
    if len(pitch_values) < window_size:
        return len(pitch_values)
    
    # Calculate rolling standard deviation
    rolling_std = []
    for i in range(len(pitch_values)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(pitch_values), i + window_size // 2 + 1)
        window = pitch_values[start_idx:end_idx]
        rolling_std.append(np.std(window))
    
    rolling_std = np.array(rolling_std)
    
    # Find first point where stability exceeds threshold
    unstable_points = rolling_std > threshold_std
    
    if not np.any(unstable_points):
        return len(pitch_values)
    
    # Find the first break point (with some tolerance for noise)
    # Look for 3 consecutive unstable points to confirm it's a real break
    for i in range(len(unstable_points) - 2):
        if unstable_points[i] and unstable_points[i+1] and unstable_points[i+2]:
            return i
    
    return len(pitch_values)

def find_clean_plot_range(pitch_values, yaw_values, break_point, stable_window=50, unstable_window=20):
    """Find the range for clean plotting"""
    total_frames = len(pitch_values)
    
    # Start from beginning
    start_frame = 0
    
    # End point: include stable portion + small unstable window
    if break_point < total_frames:
        end_frame = min(break_point + unstable_window, total_frames)
    else:
        # If no break detected, show all data
        end_frame = total_frames
    
    return start_frame, end_frame, break_point

def plot_experiment_1_pitch_vs_yaw(csv_files, scenario_names=None, instability_threshold=10.0, save_dir=None):
    """Create one figure with 3 subplots - Pitch vs Yaw for each scenario"""
    
    if len(csv_files) != 3:
        print("❌ Error: Exactly 3 CSV files are required for Experiment 1!")
        return
    
    print(f"🔬 Creating Experiment 1: Pitch vs Yaw Analysis")
    print(f"   Instability threshold: {instability_threshold}° std dev")
    print()
    
    # Create figure with 3 subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Experiment 1: Pitch vs Yaw Analysis\n(Static P, Variable Y)', fontsize=16, fontweight='bold')
    
    results = []
    colors = ['blue', 'red', 'green']
    
    # Process each CSV file and create subplot
    for i, csv_file in enumerate(csv_files):
        print(f"📊 Processing scenario {i+1}/3: {csv_file}")
        
        # Load data
        data = load_csv_data(csv_file)
        yaw = np.array(data['yaw'])
        pitch = np.array(data['pitch'])
        timestamps = np.array(data['timestamp'])
        
        if len(pitch) == 0:
            print(f"❌ No data found in CSV file: {csv_file}")
            continue
        
        # Detect break point where pitch becomes unstable
        break_point = detect_instability_breakpoint(pitch, window_size=10, threshold_std=5.0)
        
        # Interactive break point input
        auto_break_yaw = yaw[break_point] if break_point < len(yaw) else None
        print(f"   ✓ Auto-detected break point: Frame {break_point+1} at {auto_break_yaw:.1f}° yaw")
        
        try:
            manual_input = input(f"   Enter new frame number for break point (or press Enter to keep current): ").strip()
            if manual_input:
                manual_frame = int(manual_input) - 1  # Convert to 0-based index
                if 0 <= manual_frame < len(yaw):
                    break_point = manual_frame
                    print(f"   ✓ Manual break point: Frame {manual_frame+1} at {yaw[manual_frame]:.1f}° yaw")
                else:
                    print(f"   ⚠️  Invalid frame number, keeping auto-detected break point")
        except (ValueError, KeyboardInterrupt):
            print(f"   ⚠️  Keeping auto-detected break point")
        
        # Find clean plot range
        start_frame, end_frame, actual_break = find_clean_plot_range(pitch, yaw, break_point)
        
        # Get clean data ranges
        yaw_clean = yaw[start_frame:end_frame]
        pitch_clean = pitch[start_frame:end_frame]
        
        # Calculate pitch stability for the clean range
        pitch_stability = calculate_pitch_stability(pitch_clean)
        
        # Find instability thresholds
        pos_threshold, neg_threshold = find_instability_threshold(yaw_clean, pitch_stability, instability_threshold)
        
        # Get scenario name
        if scenario_names and i < len(scenario_names):
            scenario_name = scenario_names[i]
        else:
            scenario_name = Path(csv_file).stem
        
        # Create subplot
        ax = axes[i]
        
        # Plot Pitch vs Yaw (clean data)
        ax.plot(yaw_clean, pitch_clean, color=colors[i], linewidth=2, alpha=0.8, 
                label=f'Pitch (mean: {pitch_clean.mean():.1f}°)')
        
        # Mark the break point if detected
        if actual_break < len(yaw_clean):
            break_yaw = yaw_clean[actual_break - start_frame]
            break_pitch = pitch_clean[actual_break - start_frame]
            ax.scatter(break_yaw, break_pitch, color='red', s=100, zorder=5, 
                      label=f'Break point: {break_yaw:.1f}° yaw')
        
        # Add instability threshold lines if found
        if pos_threshold is not None:
            ax.axvline(x=pos_threshold, color='orange', linestyle='--', alpha=0.8, linewidth=2,
                      label=f'Unstable threshold: {pos_threshold:.1f}°')
        if neg_threshold is not None:
            ax.axvline(x=neg_threshold, color='purple', linestyle='--', alpha=0.8, linewidth=2,
                      label=f'Unstable threshold: {neg_threshold:.1f}°')
        
        # Customize subplot
        ax.set_xlabel('Yaw (degrees)')
        ax.set_ylabel('Pitch (degrees)')
        ax.set_title(f'{scenario_name}\nMean P: {pitch_clean.mean():.1f}°')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.legend(fontsize=9)
        
        # Add statistics text
        stable_frames = actual_break - start_frame if actual_break < len(pitch) else len(pitch_clean)
        stats_text = f"""Data (Clean Range):
• Frames: {start_frame+1}-{end_frame} ({len(pitch_clean)} total)
• Stable frames: {stable_frames}
• P range: {pitch_clean.min():.1f}° to {pitch_clean.max():.1f}°
• Y range: {yaw_clean.min():.1f}° to {yaw_clean.max():.1f}°
• Break at yaw: {break_yaw:.1f}° (if detected)"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
        
        # Store results
        results.append({
            'scenario': scenario_name,
            'positive_threshold': pos_threshold,
            'negative_threshold': neg_threshold,
            'unstable_frames': np.sum(pitch_stability > instability_threshold),
            'total_frames': len(pitch_clean),
            'mean_pitch': pitch_clean.mean(),
            'yaw_range': (yaw_clean.min(), yaw_clean.max()),
            'break_point_yaw': break_yaw if actual_break < len(yaw_clean) else None,
            'stable_frames': stable_frames
        })
        
        print(f"   ✓ Mean pitch: {pitch_clean.mean():.1f}°")
        print(f"   ✓ Yaw range: {yaw_clean.min():.1f}° to {yaw_clean.max():.1f}°")
        print(f"   ✓ Break point: Frame {actual_break+1} at {break_yaw:.1f}° yaw" if actual_break < len(yaw_clean) else "   ✓ No clear break point detected")
        if pos_threshold is not None:
            print(f"   ✓ Positive threshold: {pos_threshold:.1f}°")
        if neg_threshold is not None:
            print(f"   ✓ Negative threshold: {neg_threshold:.1f}°")
        print()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    if save_dir is None:
        save_dir = 'examples/experiment/plot'
    os.makedirs(save_dir, exist_ok=True)
    
    output_file = os.path.join(save_dir, 'experiment_1_pitch_vs_yaw_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved: {output_file}")
    
    # Print overall summary
    if results:
        print(f"\n🎯 EXPERIMENT 1 SUMMARY:")
        print(f"   Total scenarios analyzed: {len(results)}")
        print(f"   Instability threshold used: {instability_threshold}° std dev")
        print()
        
        for result in results:
            print(f"📋 {result['scenario']}:")
            print(f"   • Mean pitch: {result['mean_pitch']:.1f}°")
            print(f"   • Yaw range: {result['yaw_range'][0]:.1f}° to {result['yaw_range'][1]:.1f}°")
            if result['break_point_yaw'] is not None:
                print(f"   • Break point: {result['break_point_yaw']:.1f}° yaw")
            if result['positive_threshold'] is not None:
                print(f"   • Positive yaw threshold: {result['positive_threshold']:.1f}°")
            if result['negative_threshold'] is not None:
                print(f"   • Negative yaw threshold: {result['negative_threshold']:.1f}°")
            print(f"   • Stable frames: {result['stable_frames']}/{result['total_frames']}")
            print()
        
        # Find overall threshold
        all_pos_thresholds = [r['positive_threshold'] for r in results if r['positive_threshold'] is not None]
        all_neg_thresholds = [r['negative_threshold'] for r in results if r['negative_threshold'] is not None]
        
        if all_pos_thresholds:
            print(f"🎯 OVERALL POSITIVE YAW THRESHOLD: {min(all_pos_thresholds):.1f}°")
        if all_neg_thresholds:
            print(f"🎯 OVERALL NEGATIVE YAW THRESHOLD: {max(all_neg_thresholds):.1f}°")
    
    plt.show()

def calculate_pitch_stability(pitch_values, window_size=5):
    """Calculate pitch stability using rolling standard deviation"""
    if len(pitch_values) < window_size:
        return np.zeros_like(pitch_values)
    
    stability = []
    for i in range(len(pitch_values)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(pitch_values), i + window_size // 2 + 1)
        window = pitch_values[start_idx:end_idx]
        stability.append(np.std(window))
    
    return np.array(stability)

def find_instability_threshold(yaw_values, pitch_stability, threshold_std=10.0):
    """Find yaw threshold where pitch becomes unstable"""
    # Find frames where pitch is unstable
    unstable_frames = pitch_stability > threshold_std
    
    if not np.any(unstable_frames):
        return None, None
    
    # Get yaw values at unstable frames
    unstable_yaws = yaw_values[unstable_frames]
    
    # Find the actual yaw value (with sign)
    if np.any(unstable_yaws >= 0):
        positive_threshold = np.min(unstable_yaws[unstable_yaws >= 0])
    else:
        positive_threshold = None
        
    if np.any(unstable_yaws < 0):
        negative_threshold = np.max(unstable_yaws[unstable_yaws < 0])
    else:
        negative_threshold = None
    
    return positive_threshold, negative_threshold

def main():
    parser = argparse.ArgumentParser(description='Plot pitch vs yaw analysis for Experiment 1 (3 scenarios in one figure)')
    parser.add_argument('csv_files', nargs=3, help='Exactly 3 CSV files with pose data (3 different scenarios)')
    parser.add_argument('--threshold', type=float, default=10.0, 
                       help='Instability threshold (std dev of pitch, default: 10.0)')
    parser.add_argument('--save-dir', help='Directory to save plot (default: examples/experiment/plot)')
    parser.add_argument('--scenario-names', nargs=3, help='Names for each scenario (optional)')
    
    args = parser.parse_args()
    
    # Validate input files
    for csv_file in args.csv_files:
        if not os.path.exists(csv_file):
            print(f"❌ Error: CSV file '{csv_file}' not found!")
            sys.exit(1)
    
    # Validate scenario names
    if args.scenario_names and len(args.scenario_names) != 3:
        print(f"❌ Error: Number of scenario names ({len(args.scenario_names)}) must be exactly 3!")
        sys.exit(1)
    
    plot_experiment_1_pitch_vs_yaw(args.csv_files, args.scenario_names, args.threshold, args.save_dir)

if __name__ == "__main__":
    main() 