#!/usr/bin/env python3
# coding: utf-8

"""
Real-time Webcam Demo with Pitch, Yaw, Roll Display
===================================================

This demo shows how to extract and display head pose angles (pitch, yaw, roll)
in real-time from webcam input using the 3DDFA_V2 system.

Usage:
    python demo_webcam_pose.py --config configs/mb1_120x120.yml --onnx

Features:
    - Real-time face detection and 3D reconstruction
    - Live pitch, yaw, roll angle display
    - Smooth tracking across frames
    - Configurable visualization options
"""

__author__ = 'cleardusk'

import argparse
import cv2
import numpy as np
import yaml
from collections import deque
import time

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import cv_draw_landmark
from utils.pose import matrix2angle
from utils.tddfa_util import _parse_param


def draw_pose_info(img, pitch, yaw, roll, confidence=None):
    """
    Draw pose information on the image.
    
    Args:
        img: Input image
        pitch: Pitch angle in degrees
        yaw: Yaw angle in degrees  
        roll: Roll angle in degrees
        confidence: Optional confidence score
    """
    # Convert to degrees and format
    pitch_deg = int(pitch)
    yaw_deg = int(yaw)
    roll_deg = int(roll)
    
    # Set up text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0)  # Green color
    
    # Background rectangle for better readability
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Draw pose information
    y_offset = 40
    cv2.putText(img, f"Pitch: {pitch_deg:3d}°", (20, y_offset), font, font_scale, color, thickness)
    cv2.putText(img, f"Yaw:   {yaw_deg:3d}°", (20, y_offset + 30), font, font_scale, color, thickness)
    cv2.putText(img, f"Roll:  {roll_deg:3d}°", (20, y_offset + 60), font, font_scale, color, thickness)
    
    # Add confidence if available
    if confidence is not None:
        cv2.putText(img, f"Conf:  {confidence:.2f}", (20, y_offset + 90), font, font_scale, (255, 255, 0), thickness)
    
    # Add pose interpretation
    pose_text = get_pose_description(pitch_deg, yaw_deg, roll_deg)
    cv2.putText(img, pose_text, (20, y_offset + 120), font, 0.6, (255, 255, 255), 1)


def get_pose_description(pitch, yaw, roll):
    """
    Get human-readable description of head pose.
    
    Args:
        pitch: Pitch angle in degrees
        yaw: Yaw angle in degrees
        roll: Roll angle in degrees
        
    Returns:
        String description of pose
    """
    descriptions = []
    
    # Pitch (up/down)
    if pitch > 15:
        descriptions.append("Looking up")
    elif pitch < -15:
        descriptions.append("Looking down")
    else:
        descriptions.append("Level")
    
    # Yaw (left/right)
    if yaw > 15:
        descriptions.append("Turned left")
    elif yaw < -15:
        descriptions.append("Turned right")
    else:
        descriptions.append("Facing forward")
    
    # Roll (tilt)
    if roll > 15:
        descriptions.append("Tilted left")
    elif roll < -15:
        descriptions.append("Tilted right")
    
    return " | ".join(descriptions)


def extract_pose_angles(param):
    """
    Extract pitch, yaw, roll angles from 3DMM parameters.
    
    Args:
        param: 62-dimensional parameter vector
        
    Returns:
        pitch, yaw, roll angles in degrees
    """
    # Parse parameters to get rotation matrix
    R, offset, alpha_shp, alpha_exp = _parse_param(param)
    
    # Convert rotation matrix to Euler angles
    pitch, yaw, roll = matrix2angle(R)
    
    # Convert from radians to degrees
    pitch_deg = pitch * 180 / np.pi
    yaw_deg = yaw * 180 / np.pi
    roll_deg = roll * 180 / np.pi
    
    return pitch_deg, yaw_deg, roll_deg


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Smoothing parameters
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_param = deque()

    # Performance tracking
    fps_counter = 0
    start_time = time.time()
    
    print("Starting webcam... Press 'q' to quit")
    
    dense_flag = args.opt in ('2d_dense', '3d')
    pre_ver = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        try:
            if pre_ver is None:
                # First frame: detect face
                boxes = face_boxes(frame)
                if len(boxes) == 0:
                    cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('3DDFA_V2 Pose Estimation', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                    
                boxes = [boxes[0]]  # Use first detected face
                param_lst, roi_box_lst = tddfa(frame, boxes)
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                
                # Refine with landmark-based cropping
                param_lst, roi_box_lst = tddfa(frame, [ver], crop_policy='landmark')
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                
                # Initialize queues
                param = param_lst[0]
                for _ in range(n_pre):
                    queue_ver.append(ver.copy())
                    queue_param.append(param.copy())
                queue_ver.append(ver.copy())
                queue_param.append(param.copy())
                
            else:
                # Subsequent frames: track face
                param_lst, roi_box_lst = tddfa(frame, [pre_ver], crop_policy='landmark')
                
                # Check if tracking failed (face too small)
                roi_box = roi_box_lst[0]
                if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                    boxes = face_boxes(frame)
                    if len(boxes) > 0:
                        boxes = [boxes[0]]
                        param_lst, roi_box_lst = tddfa(frame, boxes)
                
                ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                param = param_lst[0]
                
                queue_ver.append(ver.copy())
                queue_param.append(param.copy())
            
            pre_ver = ver  # Update for tracking
            
            # Apply smoothing if queue is full
            if len(queue_ver) >= n:
                ver_ave = np.mean(queue_ver, axis=0)
                param_ave = np.mean(queue_param, axis=0)
                
                # Extract pose angles
                pitch, yaw, roll = extract_pose_angles(param_ave)
                
                # Calculate confidence based on face size
                roi_box = roi_box_lst[0]
                face_area = abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1])
                confidence = min(1.0, face_area / 10000.0)  # Normalize to 0-1
                
                # Draw visualization
                if args.opt == '2d_sparse':
                    img_draw = cv_draw_landmark(frame, ver_ave)
                elif args.opt == '2d_dense':
                    img_draw = cv_draw_landmark(frame, ver_ave, size=1)
                elif args.opt == '3d':
                    img_draw = render(frame, [ver_ave], tddfa.tri, alpha=0.7)
                else:
                    img_draw = frame.copy()
                
                # Draw pose information
                draw_pose_info(img_draw, pitch, yaw, roll, confidence)
                
                # Remove oldest elements from queues
                queue_ver.popleft()
                queue_param.popleft()
                
            else:
                # Not enough frames for smoothing yet
                pitch, yaw, roll = extract_pose_angles(param)
                img_draw = frame.copy()
                draw_pose_info(img_draw, pitch, yaw, roll)
            
            # Calculate and display FPS
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:  # Update FPS every second
                fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()
                cv2.putText(img_draw, f"FPS: {fps:.1f}", (img_draw.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display result
            cv2.imshow('3DDFA_V2 Pose Estimation', img_draw)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            cv2.putText(frame, f"Error: {str(e)[:50]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('3DDFA_V2 Pose Estimation', frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time webcam pose estimation with 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml', help='Path to config file')
    parser.add_argument('-m', '--mode', type=str, default='cpu', choices=['cpu', 'gpu'], help='Processing mode')
    parser.add_argument('--onnx', action='store_true', help='Use ONNX models for faster inference')
    parser.add_argument('--opt', type=str, default='2d_sparse', 
                       choices=['2d_sparse', '2d_dense', '3d', 'pose_only'], 
                       help='Visualization option')
    parser.add_argument('--n_pre', type=int, default=1, help='Number of frames for pre-smoothing')
    parser.add_argument('--n_next', type=int, default=1, help='Number of frames for post-smoothing')
    
    args = parser.parse_args()
    main(args)
