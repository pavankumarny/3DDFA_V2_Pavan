#!/usr/bin/env python3
"""
Debug script to understand why 90° video shows incorrect pitch
"""

import numpy as np
import csv
from math import degrees, radians, sin, cos, atan2, asin

def analyze_90_degree_video():
    """Analyze the 90 degree video CSV data"""
    print("Analyzing 90° Video Pose Data")
    print("="*60)
    
    # Load CSV data
    with open('examples/results/videos/90_2d_sparse_hpe.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Extract angles
    pitches = [float(row['pitch']) for row in data]
    yaws = [float(row['yaw']) for row in data]
    rolls = [float(row['roll']) for row in data]
    
    print(f"Total frames: {len(data)}")
    print(f"\nYaw statistics:")
    print(f"  Min: {min(yaws):.1f}°")
    print(f"  Max: {max(yaws):.1f}°")
    print(f"  Mean: {sum(yaws)/len(yaws):.1f}°")
    print(f"  Std: {np.std(yaws):.1f}°")
    
    print(f"\nPitch statistics (ORIGINAL VALUES):")
    print(f"  Min: {min(pitches):.1f}°")
    print(f"  Max: {max(pitches):.1f}°")
    print(f"  Mean: {sum(pitches)/len(pitches):.1f}°")
    print(f"  Std: {np.std(pitches):.1f}°")
    
    print(f"\nRoll statistics:")
    print(f"  Min: {min(rolls):.1f}°")
    print(f"  Max: {max(rolls):.1f}°")
    print(f"  Mean: {sum(rolls)/len(rolls):.1f}°")
    print(f"  Std: {np.std(rolls):.1f}°")
    
    # Check for gimbal lock conditions
    print("\n" + "="*60)
    print("GIMBAL LOCK ANALYSIS:")
    print("="*60)
    
    # Count frames near gimbal lock
    near_90_yaw = sum(1 for y in yaws if abs(abs(y) - 90) < 5)
    print(f"Frames with yaw near ±90° (within 5°): {near_90_yaw}/{len(yaws)} ({100*near_90_yaw/len(yaws):.1f}%)")
    
    # The real issue
    print("\n🔴 THE REAL PROBLEM:")
    print("When the head is viewed from the side (90° yaw), the 3DDFA_V2 algorithm")
    print("has difficulty distinguishing between pitch and yaw rotations because:")
    print("1. The face is in profile view - many landmarks are occluded")
    print("2. The rotation matrix decomposition becomes ambiguous")
    print("3. The algorithm 'sees' the head tilt as a combination of pitch and yaw")
    print("\nThe mean pitch of -100° actually means:")
    print("  • The algorithm thinks the head is tilted back 100°")
    print("  • This is because it's confusing the side view with a tilted view")
    print("  • The landmarks are correct, but the angle interpretation is wrong")
    
    print("\n💡 SOLUTION:")
    print("For accurate pose estimation at extreme yaw angles (near ±90°):")
    print("1. Use quaternions instead of Euler angles")
    print("2. Use a different pose estimation method (e.g., 6D rotation representation)")
    print("3. Apply constraints based on expected head pose range")
    print("4. Use temporal smoothing to reduce jumps")

if __name__ == "__main__":
    analyze_90_degree_video() 