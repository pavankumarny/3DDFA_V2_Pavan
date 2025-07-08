#!/usr/bin/env python3
# coding: utf-8

"""
Video/Webcam Pose Estimation Demo
=================================

Extract and display pitch, yaw, roll angles from video files or webcam.
Based on the 3DDFA_V2 system with real-time pose angle visualization.

Usage:
    # Webcam
    python demo_pose_angles.py --mode webcam --config configs/mb1_120x120.yml --onnx
    
    # Video file  
    python demo_pose_angles.py --mode video -f input_video.mp4 --config configs/mb1_120x120.yml --onnx
"""

import argparse
import cv2
import numpy as np
import yaml
import time
import imageio

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import cv_draw_landmark
from utils.pose import calc_pose
from utils.tddfa_util import _parse_param


def extract_pose_angles(param):
    """Extract pitch, yaw, roll from 3DMM parameters using the existing tested method."""
    # Use the existing calc_pose function which is already tested and used in demo.py
    P, pose = calc_pose(param)
    
    # From utils/pose.py viz_pose function, we know the order is:
    # pose[0] = yaw, pose[1] = pitch, pose[2] = roll
    yaw = pose[0]    # Left/right head rotation
    pitch = pose[1]  # Up/down head rotation  
    roll = pose[2]   # Head tilt rotation
    
    return pitch, yaw, roll


def draw_pose_overlay(img, pitch, yaw, roll):
    """Draw pose angles on image with directional indicators."""
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    color = (0, 255, 0)
    
    # Draw angles with directional indicators
    y_start = 30
    cv2.putText(img, f"Pitch: {pitch:6.1f}° {'UP' if pitch > 5 else 'DOWN' if pitch < -5 else 'LEVEL'}", 
                (20, y_start), font, scale, color, thickness)
    
    cv2.putText(img, f"Yaw:   {yaw:6.1f}° {'RIGHT' if yaw > 5 else 'LEFT' if yaw < -5 else 'CENTER'}", 
                (20, y_start + 25), font, scale, color, thickness)
    
    cv2.putText(img, f"Roll:  {roll:6.1f}° {'RIGHT' if roll > 5 else 'LEFT' if roll < -5 else 'LEVEL'}", 
                (20, y_start + 50), font, scale, color, thickness)
    
    # Add instruction text
    cv2.putText(img, "Test: Move head UP/DOWN (pitch should change)", 
                (20, y_start + 80), font, 0.4, (255, 255, 0), 1)
    cv2.putText(img, "Test: Turn head LEFT/RIGHT (yaw should change)", 
                (20, y_start + 95), font, 0.4, (255, 255, 0), 1)
    cv2.putText(img, "Test: Tilt head LEFT/RIGHT (roll should change)", 
                (20, y_start + 110), font, 0.4, (255, 255, 0), 1)
    
    # Add note about the tested implementation
    cv2.putText(img, "Using tested calc_pose() from original demo", 
                (20, y_start + 130), font, 0.35, (200, 200, 200), 1)


def process_webcam(args, tddfa, face_boxes):
    """Process webcam input."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("Webcam started. Press 'q' to quit")
    pre_ver = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)  # Mirror effect
        
        try:
            if pre_ver is None:
                # First frame: detect
                boxes = face_boxes(frame)
                if len(boxes) == 0:
                    cv2.putText(frame, "No face detected", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Pose Angles', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                param_lst, roi_box_lst = tddfa(frame, [boxes[0]])
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
                pre_ver = ver
            else:
                # Track
                param_lst, roi_box_lst = tddfa(frame, [pre_ver], crop_policy='landmark')
                roi_box = roi_box_lst[0]
                
                # Re-detect if tracking failed
                if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                    boxes = face_boxes(frame)
                    if len(boxes) > 0:
                        param_lst, roi_box_lst = tddfa(frame, [boxes[0]])
                
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
                pre_ver = ver
            
            # Extract pose
            param = param_lst[0]
            pitch, yaw, roll = extract_pose_angles(param)
            
            # Visualize
            if args.show_landmarks:
                frame = cv_draw_landmark(frame, ver)
            
            draw_pose_overlay(frame, pitch, yaw, roll)
            
            cv2.imshow('Pose Angles', frame)
            
        except Exception as e:
            print(f"Frame processing error: {e}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def process_video(args, tddfa, face_boxes):
    """Process video file."""
    # Open video
    if args.input_file.endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(args.input_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        print(f"Unsupported video format: {args.input_file}")
        return
    
    # Setup output if requested
    writer = None
    if args.output_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output_file, fourcc, fps, (width, height))
    
    print(f"Processing video: {args.input_file}")
    print(f"Frames: {frame_count}, FPS: {fps}")
    
    pre_ver = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            if frame_idx == 0:
                # First frame
                boxes = face_boxes(frame)
                if len(boxes) == 0:
                    print(f"No face detected in frame {frame_idx}")
                    frame_idx += 1
                    continue
                
                param_lst, roi_box_lst = tddfa(frame, [boxes[0]])
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
                pre_ver = ver
            else:
                # Track
                param_lst, roi_box_lst = tddfa(frame, [pre_ver], crop_policy='landmark')
                roi_box = roi_box_lst[0]
                
                # Re-detect if needed
                if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                    boxes = face_boxes(frame)
                    if len(boxes) > 0:
                        param_lst, roi_box_lst = tddfa(frame, [boxes[0]])
                
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
                pre_ver = ver
            
            # Extract and display pose
            param = param_lst[0]
            pitch, yaw, roll = extract_pose_angles(param)
            
            if args.show_landmarks:
                frame = cv_draw_landmark(frame, ver)
            
            draw_pose_overlay(frame, pitch, yaw, roll)
            
            # Output frame
            if writer:
                writer.write(frame)
            
            if args.display:
                cv2.imshow('Video Pose Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_idx % 30 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_idx}/{frame_count}")
                print(f"  Pitch: {pitch:6.1f}°, Yaw: {yaw:6.1f}°, Roll: {roll:6.1f}°")
        
        except Exception as e:
            print(f"Error in frame {frame_idx}: {e}")
        
        frame_idx += 1
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete! Processed {frame_idx} frames")
    if args.output_file:
        print(f"Output saved to: {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description='3DDFA_V2 Pose Angle Extraction')
    parser.add_argument('--mode', choices=['webcam', 'video'], required=True,
                       help='Input mode: webcam or video file')
    parser.add_argument('-f', '--input_file', type=str,
                       help='Input video file (required for video mode)')
    parser.add_argument('-o', '--output_file', type=str,
                       help='Output video file with pose overlay')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml',
                       help='Config file path')
    parser.add_argument('--onnx', action='store_true',
                       help='Use ONNX models for speed')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--show_landmarks', action='store_true',
                       help='Show facial landmarks')
    parser.add_argument('--display', action='store_true', default=True,
                       help='Display video during processing')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'video' and not args.input_file:
        print("Error: --input_file required for video mode")
        return
    
    # Load config
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    
    # Initialize models
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX
        
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
        print("Using ONNX models for faster inference")
    else:
        face_boxes = FaceBoxes()
        tddfa = TDDFA(gpu_mode=args.gpu, **cfg)
        print(f"Using {'GPU' if args.gpu else 'CPU'} mode")
    
    # Process input
    if args.mode == 'webcam':
        process_webcam(args, tddfa, face_boxes)
    else:
        process_video(args, tddfa, face_boxes)


if __name__ == '__main__':
    main()
