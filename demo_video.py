# coding: utf-8
"""
3DDFA_V2 Video Demo Script
==========================

This script processes videos to perform 3D Dense Face Alignment (3DDFA) on faces.
It can output either:
1. 2D sparse landmarks (dots on facial features)
2. 3D rendered face meshes overlaid on the original video

The system works by:
1. Detecting faces in each frame using FaceBoxes
2. Reconstructing 3D face models using TDDFA (Three-D Dense Face Alignment)
3. Tracking faces across frames for better performance
4. Rendering the results back onto the video frames
"""

__author__ = 'cleardusk'

# Standard library imports for command line arguments, video I/O, progress bars, and config files
import argparse          # Parse command line arguments (like --config, --video_fp, etc.)
import imageio           # Read and write video files frame by frame
from tqdm import tqdm    # Show progress bars during video processing
import yaml              # Load configuration files (YAML format)

# Core 3DDFA_V2 modules
from FaceBoxes import FaceBoxes          # Face detection module - finds rectangular boxes around faces
from TDDFA import TDDFA                  # Main 3D face alignment module - builds 3D face models from 2D images
from utils.render import render          # 3D rendering module - draws 3D face meshes on images
# from utils.render_ctypes import render  # Alternative C-based renderer (faster but requires compilation)
from utils.functions import cv_draw_landmark, get_suffix  # Helper functions for drawing and file handling


def main(args):
    """
    Main function that processes a video file to perform 3D face alignment.
    
    Args:
        args: Command line arguments containing:
            - config: Path to YAML configuration file
            - video_fp: Path to input video file
            - mode: 'cpu' or 'gpu' processing mode
            - opt: Output type ('2d_sparse' for landmarks, '3d' for mesh)
            - onnx: Whether to use ONNX models for faster inference
    """
    # Load configuration settings from YAML file (model paths, parameters, etc.)
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Initialize face detection and 3D face alignment models
    # Two modes available: ONNX (faster) or PyTorch (standard)
    if args.onnx:
        # ONNX mode: Optimized models for faster inference
        import os
        # Set environment variables for better CPU performance
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Avoid library conflicts
        os.environ['OMP_NUM_THREADS'] = '4'          # Use 4 CPU threads

        # Import ONNX-optimized versions of the models
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        # Initialize the optimized models
        face_boxes = FaceBoxes_ONNX()    # Fast face detector
        tddfa = TDDFA_ONNX(**cfg)        # Fast 3D face alignment
    else:
        # Standard PyTorch mode
        gpu_mode = args.mode == 'gpu'    # Check if GPU mode is requested
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)  # Standard 3D face alignment
        face_boxes = FaceBoxes()                 # Standard face detector

    # Setup video input/output
    fn = args.video_fp.split('/')[-1]           # Extract filename from full path
    reader = imageio.get_reader(args.video_fp)  # Create video reader object

    # Get video metadata (frames per second) to maintain same speed in output
    fps = reader.get_meta_data()['fps']

    # Create output video path and writer
    suffix = get_suffix(args.video_fp)  # Get file extension (.mp4, .avi, etc.)
    video_wfp = f'examples/results/videos/{fn.replace(suffix, "")}_{args.opt}.mp4'
    writer = imageio.get_writer(video_wfp, fps=fps)  # Create video writer with same FPS

    # Main processing loop - analyze each frame
    dense_flag = args.opt in ('3d',)  # Use dense reconstruction for 3D mesh rendering
    pre_ver = None                    # Store previous frame's face vertices for tracking
    # Process each frame of the video
    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # Convert from RGB to BGR color format (OpenCV uses BGR)

        if i == 0:
            # FIRST FRAME: Full face detection and 3D reconstruction
            # We need to detect faces from scratch since we have no prior information
            
            # Step 1: Detect all faces in the frame
            boxes = face_boxes(frame_bgr)  # Returns list of face bounding boxes [x1, y1, x2, y2, confidence]
            boxes = [boxes[0]]             # Use only the first detected face (you can modify this for multiple faces)
            
            # Step 2: Perform 3D face alignment using the detected face box
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)  # Returns 62 3DMM parameters and refined face boxes
            
            # Step 3: Reconstruct 3D vertices from parameters
            # dense_flag=True gives ~38k vertices, False gives ~68 landmarks
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # Step 4: Refine the result for better accuracy
            # Use the reconstructed vertices to get better crop and more precise parameters
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
        else:
            # SUBSEQUENT FRAMES: Use tracking for efficiency
            # Instead of detecting faces from scratch, track from previous frame
            
            # Step 1: Try to track face using previous frame's vertices
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            # Step 2: Check if tracking failed (face became too small/lost)
            roi_box = roi_box_lst[0]  # Get the tracked face box
            face_area = abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1])  # Calculate face area
            
            # If face area is too small (< 2020 pixels), tracking probably failed
            if face_area < 2020:
                # Fallback to full face detection
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            # Step 3: Reconstruct 3D vertices for this frame
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        # Store current frame's vertices for tracking in next frame
        pre_ver = ver

        # Generate output based on selected visualization mode
        if args.opt == '2d_sparse':
            # Draw 2D landmarks (dots) on the original image
            res = cv_draw_landmark(frame_bgr, ver)
        elif args.opt == '3d':
            # Render 3D face mesh overlay on the original image
            res = render(frame_bgr, [ver], tddfa.tri)  # tddfa.tri contains triangle faces for mesh
        else:
            raise ValueError(f'Unknown visualization option: {args.opt}')

        # Save the processed frame to output video
        writer.append_data(res[..., ::-1])  # Convert BGR back to RGB for video writer

    # Close video writer and finish processing
    writer.close()
    print(f'Processed video saved to: {video_wfp}')


if __name__ == '__main__':
    # Command line argument parser setup
    parser = argparse.ArgumentParser(description='3DDFA_V2 Video Processing Demo - Analyze faces in videos')
    
    # Configuration file (contains model paths, parameters, etc.)
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml',
                       help='Path to YAML configuration file')
    
    # Input video file path
    parser.add_argument('-f', '--video_fp', type=str, required=True,
                       help='Path to input video file to process')
    
    # Processing mode (CPU vs GPU)
    parser.add_argument('-m', '--mode', default='cpu', type=str, choices=['cpu', 'gpu'],
                       help='Processing mode: "cpu" for CPU-only, "gpu" for GPU acceleration')
    
    # Output visualization type
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', 
                       choices=['2d_sparse', '3d'],
                       help='Output type: "2d_sparse" for landmark dots, "3d" for mesh overlay')
    
    # Use optimized ONNX models for faster inference
    parser.add_argument('--onnx', action='store_true', default=False,
                       help='Use ONNX optimized models for faster processing')

    # Parse command line arguments and run main function
    args = parser.parse_args()
    main(args)
