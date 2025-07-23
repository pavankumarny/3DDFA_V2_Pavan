#!/usr/bin/env python3
"""
Comprehensive Algorithm Diagnostic Test
=======================================

This test will identify ANY remaining differences between pose_extractor.py
and demo_video_original.py algorithms.
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

def test_detailed_comparison():
    """Detailed frame-by-frame comparison of both algorithms"""
    print("🔍 COMPREHENSIVE ALGORITHM DIAGNOSTIC TEST")
    print("="*70)
    
    video_path = "examples/experiment/90.MOV"
    test_frames = 10  # Test first 10 frames in detail
    
    print(f"Testing video: {video_path}")
    print(f"Detailed analysis of first {test_frames} frames")
    print()
    
    # ===============================================
    # Test 1: Original algorithm (cv2.VideoCapture)
    # ===============================================
    print("🔬 TEST 1: Original algorithm with cv2.VideoCapture")
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    
    from FaceBoxes import FaceBoxes
    from TDDFA import TDDFA
    from utils.pose import calc_pose
    
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **cfg)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    original_results = []
    dense_flag = False
    pre_ver = None
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_bgr = frame  # cv2 already reads as BGR
        
        if i == 0:
            print(f"  Frame {i}: Shape={frame_bgr.shape}, dtype={frame_bgr.dtype}")
            boxes = face_boxes(frame_bgr)
            print(f"  Frame {i}: Detected {len(boxes)} faces")
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            
            # Refinement
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')
            roi_box = roi_box_lst[0]
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        
        pre_ver = ver
        P, pose = calc_pose(param_lst[0])
        yaw, pitch, roll = pose[0], pose[1], pose[2]
        
        original_results.append([i, i/fps, pitch, yaw, roll])
        print(f"  Frame {i}: P={pitch:.3f}°, Y={yaw:.3f}°, R={roll:.3f}°")
    
    cap.release()
    
    # ===============================================
    # Test 2: pose_extractor algorithm
    # ===============================================
    print(f"\n🔬 TEST 2: pose_extractor algorithm")
    
    import sys
    sys.path.append('tools')
    from pose_extractor import PoseExtractor
    
    extractor = PoseExtractor(use_onnx=False)
    
    cap2 = cv2.VideoCapture(video_path)
    extractor_results = []
    previous_landmarks = None
    
    for i in range(test_frames):
        ret, frame = cap2.read()
        if not ret:
            break
            
        frame_bgr = frame  # cv2 already reads as BGR
        
        if i == 0:
            print(f"  Frame {i}: Shape={frame_bgr.shape}, dtype={frame_bgr.dtype}")
        
        result = extractor.process_frame(frame_bgr, previous_landmarks)
        
        if result['success']:
            previous_landmarks = result['landmarks']
            extractor_results.append([i, i/fps, result['pitch'], result['yaw'], result['roll']])
            print(f"  Frame {i}: P={result['pitch']:.3f}°, Y={result['yaw']:.3f}°, R={result['roll']:.3f}°")
        else:
            print(f"  Frame {i}: FAILED")
            previous_landmarks = None
    
    cap2.release()
    
    # ===============================================
    # Test 3: Detailed comparison
    # ===============================================
    print(f"\n📊 DETAILED FRAME-BY-FRAME COMPARISON:")
    print("-" * 70)
    
    max_frames = min(len(original_results), len(extractor_results))
    total_diff = 0
    
    for i in range(max_frames):
        orig = original_results[i]
        extr = extractor_results[i]
        
        p_diff = abs(orig[2] - extr[2])  # pitch
        y_diff = abs(orig[3] - extr[3])  # yaw  
        r_diff = abs(orig[4] - extr[4])  # roll
        frame_diff = np.sqrt(p_diff**2 + y_diff**2 + r_diff**2)
        total_diff += frame_diff
        
        print(f"Frame {i:2d}: P_diff={p_diff:8.5f}°, Y_diff={y_diff:8.5f}°, R_diff={r_diff:8.5f}°, Total={frame_diff:8.5f}°")
    
    avg_diff = total_diff / max_frames if max_frames > 0 else 0
    print(f"\nAverage difference per frame: {avg_diff:.6f}°")
    
    # ===============================================
    # Test 4: Parameter inspection
    # ===============================================
    print(f"\n🔍 ALGORITHM PARAMETER INSPECTION:")
    print("-" * 70)
    
    # Test with single frame to compare intermediate results
    cap3 = cv2.VideoCapture(video_path)
    ret, test_frame = cap3.read()
    cap3.release()
    
    if ret:
        print("Testing first frame processing in detail...")
        
        # Original algorithm step-by-step
        print("\nOriginal algorithm:")
        boxes_orig = face_boxes(test_frame)
        print(f"  Face detection: {len(boxes_orig)} faces")
        print(f"  First box: {boxes_orig[0]}")
        
        param_lst_orig, roi_box_lst_orig = tddfa(test_frame, [boxes_orig[0]])
        print(f"  Initial params shape: {param_lst_orig[0].shape}")
        
        ver_orig = tddfa.recon_vers(param_lst_orig, roi_box_lst_orig, dense_flag=False)[0]
        print(f"  Initial vertices shape: {ver_orig.shape}")
        
        # Refinement step
        param_lst_refined, roi_box_lst_refined = tddfa(test_frame, [ver_orig], crop_policy='landmark')
        print(f"  Refined params shape: {param_lst_refined[0].shape}")
        print(f"  Refined ROI box: {roi_box_lst_refined[0]}")
        
        P_orig, pose_orig = calc_pose(param_lst_refined[0])
        print(f"  Final pose: {pose_orig}")
        
        # pose_extractor algorithm step-by-step
        print("\npose_extractor algorithm:")
        result_extr = extractor.process_frame(test_frame, None)
        print(f"  Success: {result_extr['success']}")
        print(f"  Final pose: [{result_extr['yaw']}, {result_extr['pitch']}, {result_extr['roll']}]")
        print(f"  Landmarks shape: {result_extr['landmarks'].shape if result_extr['landmarks'] is not None else None}")
        print(f"  BBox: {result_extr['bbox']}")
        
        # Compare final results
        print("\nDirect comparison:")
        print(f"  Pitch: {pose_orig[1]:.6f} vs {result_extr['pitch']:.6f} (diff: {abs(pose_orig[1] - result_extr['pitch']):.6f})")
        print(f"  Yaw:   {pose_orig[0]:.6f} vs {result_extr['yaw']:.6f} (diff: {abs(pose_orig[0] - result_extr['yaw']):.6f})")
        print(f"  Roll:  {pose_orig[2]:.6f} vs {result_extr['roll']:.6f} (diff: {abs(pose_orig[2] - result_extr['roll']):.6f})")
    
    # ===============================================
    # Final verdict
    # ===============================================
    print(f"\n🎯 FINAL DIAGNOSTIC VERDICT:")
    print("="*70)
    
    if avg_diff < 0.000001:
        print("✅ PERFECT: Algorithms are ABSOLUTELY IDENTICAL")
    elif avg_diff < 0.001:
        print("✅ EXCELLENT: Algorithms are essentially identical")
    elif avg_diff < 0.1:
        print("⚠️  GOOD: Very minor differences (likely numerical precision)")
    elif avg_diff < 1.0:
        print("⚠️  MINOR ISSUES: Small algorithmic differences")
    else:
        print("❌ MAJOR ISSUES: Significant algorithmic differences")
        print("   Check: frame reading, face detection, refinement step, pose extraction")
    
    return avg_diff

if __name__ == "__main__":
    test_detailed_comparison() 