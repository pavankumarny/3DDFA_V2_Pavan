#!/usr/bin/env python3
"""
Direct Algorithm Comparison: demo_video_original.py vs pose_extractor.py
========================================================================

This script processes the same video with both algorithms to identify differences.
"""

import numpy as np
import cv2
import yaml
import os
import csv
import time
from pathlib import Path

# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'

def test_original_algorithm(video_path, output_csv):
    """Test using demo_video_original.py algorithm (simplified)"""
    print("🔬 Testing ORIGINAL demo_video.py algorithm...")
    
    # Initialize exactly like demo_video_original.py
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    
    from FaceBoxes import FaceBoxes
    from TDDFA import TDDFA
    from utils.pose import calc_pose
    
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **cfg)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    results = []
    dense_flag = False  # Use sparse landmarks like in original
    pre_ver = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= 50:  # Test first 50 frames only
            break
            
        frame_bgr = frame
        
        try:
            if frame_idx == 0:
                # EXACT original algorithm: first frame
                boxes = face_boxes(frame_bgr)
                if len(boxes) == 0:
                    frame_idx += 1
                    continue
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                
                # EXACT original refinement step
                param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            else:
                # EXACT original algorithm: subsequent frames
                param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')
                
                roi_box = roi_box_lst[0]
                # EXACT original tracking failure detection
                if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                    boxes = face_boxes(frame_bgr)
                    if len(boxes) == 0:
                        frame_idx += 1
                        continue
                    boxes = [boxes[0]]
                    param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
                
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            
            pre_ver = ver  # EXACT original tracking variable
            
            # EXACT original pose extraction
            P, pose = calc_pose(param_lst[0])
            yaw = pose[0]    # Original order
            pitch = pose[1] 
            roll = pose[2]
            
            timestamp = frame_idx / fps
            results.append([frame_idx, timestamp, pitch, yaw, roll])
            
        except Exception as e:
            print(f"Frame {frame_idx} error: {e}")
        
        frame_idx += 1
    
    cap.release()
    
    # Save results
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'timestamp', 'pitch', 'yaw', 'roll'])
        writer.writerows(results)
    
    print(f"   ✅ Original algorithm: {len(results)} frames processed")
    return results

def test_pose_extractor_algorithm(video_path, output_csv):
    """Test using pose_extractor.py algorithm"""
    print("🔬 Testing POSE_EXTRACTOR algorithm...")
    
    # Import pose_extractor
    import sys
    sys.path.append('tools')
    from pose_extractor import PoseExtractor
    
    # Initialize pose extractor (no ONNX for fair comparison)
    extractor = PoseExtractor(use_onnx=False)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    results = []
    previous_landmarks = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= 50:  # Test first 50 frames only
            break
        
        try:
            # Use pose_extractor algorithm
            result = extractor.process_frame(frame, previous_landmarks)
            
            if result['success']:
                previous_landmarks = result['landmarks']
                timestamp = frame_idx / fps
                results.append([frame_idx, timestamp, result['pitch'], result['yaw'], result['roll']])
            else:
                previous_landmarks = None
                
        except Exception as e:
            print(f"Frame {frame_idx} error: {e}")
        
        frame_idx += 1
    
    cap.release()
    
    # Save results
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'timestamp', 'pitch', 'yaw', 'roll'])
        writer.writerows(results)
    
    print(f"   ✅ Pose extractor: {len(results)} frames processed")
    return results

def compare_results(results1, results2):
    """Compare the two sets of results"""
    print("\n📊 ALGORITHM COMPARISON RESULTS:")
    print("="*60)
    
    if len(results1) == 0 or len(results2) == 0:
        print("❌ One or both algorithms failed to process frames")
        return
    
    # Extract angles for comparison
    pitches1 = [r[2] for r in results1]
    yaws1 = [r[3] for r in results1]
    rolls1 = [r[4] for r in results1]
    
    pitches2 = [r[2] for r in results2]
    yaws2 = [r[3] for r in results2]
    rolls2 = [r[4] for r in results2]
    
    # Ensure same length for comparison
    min_len = min(len(pitches1), len(pitches2))
    pitches1 = pitches1[:min_len]
    pitches2 = pitches2[:min_len]
    yaws1 = yaws1[:min_len]
    yaws2 = yaws2[:min_len]
    rolls1 = rolls1[:min_len]
    rolls2 = rolls2[:min_len]
    
    # Calculate differences
    pitch_diff = np.array(pitches1) - np.array(pitches2)
    yaw_diff = np.array(yaws1) - np.array(yaws2)
    roll_diff = np.array(rolls1) - np.array(rolls2)
    
    print(f"Frames compared: {min_len}")
    print(f"\nPITCH COMPARISON:")
    print(f"  Original mean:      {np.mean(pitches1):.3f}°")
    print(f"  Pose extractor mean: {np.mean(pitches2):.3f}°")
    print(f"  Mean difference:     {np.mean(pitch_diff):.3f}°")
    print(f"  Max difference:      {np.max(np.abs(pitch_diff)):.3f}°")
    print(f"  RMS difference:      {np.sqrt(np.mean(pitch_diff**2)):.3f}°")
    
    print(f"\nYAW COMPARISON:")
    print(f"  Original mean:      {np.mean(yaws1):.3f}°")
    print(f"  Pose extractor mean: {np.mean(yaws2):.3f}°")
    print(f"  Mean difference:     {np.mean(yaw_diff):.3f}°")
    print(f"  Max difference:      {np.max(np.abs(yaw_diff)):.3f}°")
    print(f"  RMS difference:      {np.sqrt(np.mean(yaw_diff**2)):.3f}°")
    
    print(f"\nROLL COMPARISON:")
    print(f"  Original mean:      {np.mean(rolls1):.3f}°")
    print(f"  Pose extractor mean: {np.mean(rolls2):.3f}°")
    print(f"  Mean difference:     {np.mean(roll_diff):.3f}°")
    print(f"  Max difference:      {np.max(np.abs(roll_diff)):.3f}°")
    print(f"  RMS difference:      {np.sqrt(np.mean(roll_diff**2)):.3f}°")
    
    # Overall assessment
    total_rms = np.sqrt(np.mean(pitch_diff**2 + yaw_diff**2 + roll_diff**2))
    print(f"\n🎯 OVERALL RMS DIFFERENCE: {total_rms:.3f}°")
    
    if total_rms < 0.1:
        print("✅ ALGORITHMS ARE ESSENTIALLY IDENTICAL")
    elif total_rms < 1.0:
        print("⚠️  ALGORITHMS HAVE MINOR DIFFERENCES")
    else:
        print("❌ ALGORITHMS HAVE SIGNIFICANT DIFFERENCES")

def main():
    print("🔍 ALGORITHM COMPARISON TEST")
    print("="*60)
    
    # Test video
    video_path = "examples/experiment/90.MOV"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"Testing video: {video_path}")
    
    # Test both algorithms
    results1 = test_original_algorithm(video_path, "algorithm_test_original.csv")
    results2 = test_pose_extractor_algorithm(video_path, "algorithm_test_extractor.csv")
    
    # Compare results
    compare_results(results1, results2)

if __name__ == "__main__":
    main() 