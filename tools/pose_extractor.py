#!/usr/bin/env python3
# coding: utf-8

"""
3DDFA_V2 Pose Extractor
=======================

Professional-grade tool for extracting pitch, yaw, and roll angles from:
- Images (single or batch)
- Videos (with frame-by-frame analysis) 
- Live webcam (real-time processing)

This tool provides high-accuracy head pose estimation using the 3DDFA_V2 system
with optimized performance and comprehensive output options.

Usage:
    # Single image
    python3 pose_extractor.py --mode image -f path/to/image.jpg
    
    # Video with landmarks and CSV export
    python3 pose_extractor.py --mode video -f video.mp4 --landmarks -o output.mp4 --csv data.csv
    
    # Real-time webcam
    python3 pose_extractor.py --mode webcam --landmarks

Requirements:
    - Python 3.8+
    - All packages from requirements.txt
    - Pre-built 3DDFA_V2 models (run build.sh first)

Author: Based on 3DDFA_V2 by cleardusk (https://github.com/cleardusk/3DDFA_V2)
"""

import argparse
import cv2
import numpy as np
import yaml
import time
import os
import csv
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment for optimal performance
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'


class PoseExtractor:
    """High-accuracy pose extraction using 3DDFA_V2"""
    
    def __init__(self, config_path='configs/mb1_120x120.yml', use_onnx=True):
        """Initialize pose extractor with specified configuration"""
        self.config_path = config_path
        self.use_onnx = use_onnx
        
        # Store the base directory (3DDFA_V2 root) for all path operations
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        print(f"🔧 Initializing 3DDFA_V2 Pose Extractor")
        print(f"   • Base directory: {self.base_dir}")
        print(f"   • Configuration: {config_path}")
        print(f"   • ONNX Optimization: {'Enabled' if use_onnx else 'Disabled'}")
        
        # Load model configuration with proper path resolution
        if os.path.isabs(config_path):
            config_full_path = config_path
        else:
            config_full_path = os.path.join(self.base_dir, config_path)
        
        if not os.path.exists(config_full_path):
            raise FileNotFoundError(f"Config file not found: {config_full_path}")
            
        self.cfg = yaml.load(open(config_full_path), Loader=yaml.SafeLoader)
        
        # Initialize models
        self._initialize_models()
        print(f"✅ Pose extractor ready!")
    
    def _initialize_models(self):
        """Initialize face detection and pose estimation models"""
        # Change to base directory for model loading (models expect to be run from 3DDFA_V2 root)
        original_cwd = os.getcwd()
        os.chdir(self.base_dir)
        
        try:
            if self.use_onnx:
                from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
                from TDDFA_ONNX import TDDFA_ONNX
                
                self.face_detector = FaceBoxes_ONNX()
                self.pose_estimator = TDDFA_ONNX(**self.cfg)
                print(f"   • Using ONNX models (fastest performance)")
            else:
                from FaceBoxes import FaceBoxes
                from TDDFA import TDDFA
                
                self.face_detector = FaceBoxes()
                self.pose_estimator = TDDFA(gpu_mode=False, **self.cfg)
                print(f"   • Using PyTorch models")
                
        except ImportError as e:
            print(f"   ⚠️  ONNX import failed, falling back to PyTorch: {e}")
            from FaceBoxes import FaceBoxes
            from TDDFA import TDDFA
            
            self.face_detector = FaceBoxes()
            self.pose_estimator = TDDFA(gpu_mode=False, **self.cfg)
            self.use_onnx = False
        finally:
            # Return to original directory
            os.chdir(original_cwd)
    
    def extract_pose_angles(self, param):
        """Extract pitch, yaw, roll from 3DMM parameters"""
        from utils.pose import calc_pose
        
        # Use the proven calc_pose function from original 3DDFA_V2
        P, pose = calc_pose(param)
        
        # Return angles in degrees
        yaw = pose[0]    # Left(-)/Right(+) head turn
        pitch = pose[1]  # Down(-)/Up(+) head movement  
        roll = pose[2]   # Left(-)/Right(+) head tilt
        
        return pitch, yaw, roll
    
    def process_frame(self, frame, previous_landmarks=None):
        """
        Process single frame for pose estimation
        
        Returns:
            dict: {
                'success': bool,
                'pitch': float,
                'yaw': float,
                'roll': float,
                'landmarks': ndarray,
                'bbox': list
            }
        """
        result = {
            'success': False,
            'pitch': None,
            'yaw': None,
            'roll': None,
            'landmarks': None,
            'bbox': None
        }
        
        try:
            # Face detection or tracking
            if previous_landmarks is None:
                # First frame: detect face
                boxes = self.face_detector(frame)
                if len(boxes) == 0:
                    return result
                
                # Use largest detected face
                largest_box = max(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                param_lst, roi_box_lst = self.pose_estimator(frame, [largest_box])
                
            else:
                # Subsequent frames: track existing face
                param_lst, roi_box_lst = self.pose_estimator(
                    frame, [previous_landmarks], crop_policy='landmark'
                )
                
                # Check if tracking failed (small bounding box indicates failure)
                roi_box = roi_box_lst[0]
                box_area = abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1])
                
                if box_area < 2020:  # Re-detect if tracking lost
                    boxes = self.face_detector(frame)
                    if len(boxes) > 0:
                        largest_box = max(boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                        param_lst, roi_box_lst = self.pose_estimator(frame, [largest_box])
                    else:
                        return result
            
            # Extract landmarks and pose
            landmarks = self.pose_estimator.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
            pitch, yaw, roll = self.extract_pose_angles(param_lst[0])
            
            # Update result
            result.update({
                'success': True,
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'landmarks': landmarks,
                'bbox': roi_box_lst[0]
            })
            
        except Exception as e:
            print(f"⚠️  Frame processing error: {e}")
        
        return result
    
    def draw_visualization(self, frame, pitch, yaw, roll, landmarks=None, show_landmarks=False):
        """Draw pose information and landmarks on frame"""
        
        # Create semi-transparent overlay for text
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (10, 10), (min(450, w-10), 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Helper function for status colors
        def get_status_color(angle, threshold=15):
            if abs(angle) > threshold:
                return (0, 255, 255)  # Yellow for significant angle
            else:
                return (0, 255, 0)    # Green for neutral
        
        # Draw pose angles
        y = 40
        cv2.putText(frame, f"PITCH: {pitch:6.1f}°", (20, y), font, 0.6, get_status_color(pitch), 2)
        cv2.putText(frame, "(Up/Down)", (250, y), font, 0.4, (200, 200, 200), 1)
        
        y += 35
        cv2.putText(frame, f"YAW:   {yaw:6.1f}°", (20, y), font, 0.6, get_status_color(yaw), 2)
        cv2.putText(frame, "(Left/Right)", (250, y), font, 0.4, (200, 200, 200), 1)
        
        y += 35
        cv2.putText(frame, f"ROLL:  {roll:6.1f}°", (20, y), font, 0.6, get_status_color(roll), 2)
        cv2.putText(frame, "(Tilt)", (250, y), font, 0.4, (200, 200, 200), 1)
        
        # Draw landmarks if requested
        if show_landmarks and landmarks is not None:
            try:
                from utils.functions import cv_draw_landmark, GREEN
                
                # Handle different landmark formats
                if landmarks.shape[0] == 3:  # 3D landmarks
                    landmarks_2d = landmarks[:2, :]
                else:
                    landmarks_2d = landmarks
                
                frame = cv_draw_landmark(frame, landmarks_2d, color=GREEN, size=1)
                
                # Show landmark count
                cv2.putText(frame, f"Landmarks: {landmarks_2d.shape[1]}", 
                           (20, h-20), font, 0.4, GREEN, 1)
                           
            except Exception as e:
                cv2.putText(frame, f"Landmark error: {str(e)[:20]}", 
                           (20, h-20), font, 0.4, (0, 255, 255), 1)
        
        return frame


def create_output_dirs(file_path):
    """Create output directories if they don't exist"""
    if file_path:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description='3DDFA_V2 Pose Extractor - Extract pitch, yaw, roll from images/videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python3 pose_extractor.py --mode image -f ../examples/inputs/emma.jpg
  
  # Process video with landmarks and export data
  python3 pose_extractor.py --mode video -f ../examples/inputs/videos/Lit.mp4 \\
                            --landmarks -o ../results/Lit_processed.mp4 \\
                            --csv ../results/Lit_pose_data.csv
  
  # Real-time webcam with landmarks
  python3 pose_extractor.py --mode webcam --landmarks
        """
    )
    
    # Required arguments
    parser.add_argument('--mode', choices=['image', 'video', 'webcam'], required=True,
                       help='Processing mode')
    
    # Input/Output
    parser.add_argument('-f', '--file', type=str,
                       help='Input file path (required for image/video modes)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output file path')
    parser.add_argument('--csv', type=str,
                       help='Export pose data to CSV file')
    
    # Model options
    parser.add_argument('--config', type=str, default='configs/mb1_120x120.yml',
                       help='Model configuration file')
    parser.add_argument('--no-onnx', action='store_true',
                       help='Disable ONNX optimization (use PyTorch)')
    
    # Visualization
    parser.add_argument('--landmarks', action='store_true',
                       help='Show facial landmarks')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['image', 'video'] and not args.file:
        parser.error(f"--file is required for {args.mode} mode")
    
    if args.file and not os.path.exists(args.file):
        print(f"❌ Error: Input file not found: {args.file}")
        return 1
    
    # Initialize extractor
    try:
        extractor = PoseExtractor(
            config_path=args.config,
            use_onnx=not args.no_onnx
        )
        base_dir = extractor.base_dir  # Get base directory for path resolution
    except Exception as e:
        print(f"❌ Failed to initialize pose extractor: {e}")
        return 1
    
    # Helper function to resolve output paths relative to base directory
    def resolve_output_path(path):
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(base_dir, path)
    
    # Process based on mode
    if args.mode == 'image':
        # Single image processing
        print(f"📷 Processing image: {args.file}")
        
        frame = cv2.imread(args.file)
        if frame is None:
            print(f"❌ Could not load image: {args.file}")
            return 1
        
        result = extractor.process_frame(frame)
        
        if result['success']:
            print(f"✅ Pose extracted:")
            print(f"   Pitch: {result['pitch']:6.1f}° (up/down)")
            print(f"   Yaw:   {result['yaw']:6.1f}° (left/right)")
            print(f"   Roll:  {result['roll']:6.1f}° (tilt)")
            
            # Create visualization
            vis_frame = extractor.draw_visualization(
                frame, result['pitch'], result['yaw'], result['roll'],
                result['landmarks'], args.landmarks
            )
            
            # Save output
            if args.output:
                output_path = resolve_output_path(args.output)
                create_output_dirs(output_path)
                cv2.imwrite(output_path, vis_frame)
                print(f"💾 Saved result: {output_path}")
            else:
                output_path = args.file.replace('.jpg', '_pose.jpg').replace('.png', '_pose.png')
                cv2.imwrite(output_path, vis_frame)
                print(f"💾 Saved result: {output_path}")
            
            # Export CSV
            if args.csv:
                csv_path = resolve_output_path(args.csv)
                create_output_dirs(csv_path)
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['file', 'pitch', 'yaw', 'roll', 'timestamp'])
                    writer.writerow([args.file, result['pitch'], result['yaw'], result['roll'], time.time()])
                print(f"📊 Exported data: {csv_path}")
        else:
            print("❌ No face detected in image")
            return 1
    
    elif args.mode == 'video':
        # Video processing
        print(f"🎬 Processing video: {args.file}")
        
        cap = cv2.VideoCapture(args.file)
        if not cap.isOpened():
            print(f"❌ Could not open video: {args.file}")
            return 1
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        
        # Setup output video
        writer = None
        if args.output:
            output_path = resolve_output_path(args.output)
            create_output_dirs(output_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Setup CSV export
        csv_file = None
        csv_writer = None
        if args.csv:
            csv_path = resolve_output_path(args.csv)
            create_output_dirs(csv_path)
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame', 'timestamp', 'pitch', 'yaw', 'roll'])
        
        # Process frames
        previous_landmarks = None
        frame_idx = 0
        successful_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = extractor.process_frame(frame, previous_landmarks)
                
                if result['success']:
                    previous_landmarks = result['landmarks']
                    successful_frames += 1
                    
                    # Create visualization
                    frame = extractor.draw_visualization(
                        frame, result['pitch'], result['yaw'], result['roll'],
                        result['landmarks'], args.landmarks
                    )
                    
                    # Export to CSV
                    if csv_writer:
                        timestamp = frame_idx / fps
                        csv_writer.writerow([frame_idx, timestamp, result['pitch'], result['yaw'], result['roll']])
                    
                    # Progress update
                    if frame_idx % 30 == 0:
                        progress = (frame_idx / total_frames) * 100
                        print(f"Progress: {progress:5.1f}% | Frame {frame_idx:4d}/{total_frames} | "
                              f"P={result['pitch']:5.1f}° Y={result['yaw']:5.1f}° R={result['roll']:5.1f}°")
                else:
                    previous_landmarks = None
                
                # Write frame
                if writer:
                    writer.write(frame)
                
                frame_idx += 1
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if csv_file:
                csv_file.close()
        
        # Summary
        success_rate = (successful_frames / frame_idx) * 100 if frame_idx > 0 else 0
        print(f"✅ Video processing complete!")
        print(f"   Processed: {frame_idx} frames")
        print(f"   Success rate: {success_rate:.1f}%")
        if args.output:
            print(f"   Output video: {args.output}")
        if args.csv:
            print(f"   Pose data: {args.csv}")
    
    elif args.mode == 'webcam':
        # Real-time webcam processing
        print("🎥 Starting webcam pose estimation...")
        print("   Controls: 'q' to quit, 's' to save current frame")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return 1
        
        previous_landmarks = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            frame_count += 1
            
            result = extractor.process_frame(frame, previous_landmarks)
            
            if result['success']:
                previous_landmarks = result['landmarks']
                
                # Draw pose visualization
                frame = extractor.draw_visualization(
                    frame, result['pitch'], result['yaw'], result['roll'],
                    result['landmarks'], args.landmarks
                )
                
                # Console output every 30 frames
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count:4d}: P={result['pitch']:5.1f}° Y={result['yaw']:5.1f}° R={result['roll']:5.1f}°")
            else:
                cv2.putText(frame, "NO FACE DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                previous_landmarks = None
            
            cv2.imshow('3DDFA_V2 Real-time Pose Estimation', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and result['success']:
                save_path = resolve_output_path(f"results/webcam_snapshot_{int(time.time())}.jpg")
                create_output_dirs(save_path)
                cv2.imwrite(save_path, frame)
                print(f"📸 Snapshot saved: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam session ended")
    
    return 0


if __name__ == '__main__':
    print("🚀 3DDFA_V2 Pose Extractor")
    print("=" * 50)
    exit_code = main()
    sys.exit(exit_code)
