# coding: utf-8
"""
TDDFA: Three-D Dense Face Alignment
===================================

This is the core module for 3D face reconstruction from 2D images.
The TDDFA class performs the following key functions:

1. Face Alignment: Takes detected face boxes and reconstructs 3D face models
2. Parameter Regression: Uses deep neural networks to predict 3DMM parameters
3. 3D Reconstruction: Converts parameters to actual 3D face vertices
4. Tracking: Can track faces across frames for better performance

The system uses a 3D Morphable Model (3DMM) based on the Basel Face Model (BFM)
to represent faces as linear combinations of shape and expression basis vectors.
"""

__author__ = 'cleardusk'

# Core system imports
import os.path as osp        # File path operations
import time                  # Performance timing
import numpy as np           # Numerical computations for 3D operations
import cv2                   # Computer vision operations (image processing)
import torch                 # Deep learning framework
from torchvision.transforms import Compose  # Image preprocessing pipeline
import torch.backends.cudnn as cudnn        # GPU optimization

# 3DDFA_V2 specific modules
import models                # Neural network architectures (ResNet, MobileNet, etc.)
from bfm import BFMModel     # Basel Face Model - the 3D morphable model
from utils.io import _load   # File I/O utilities for loading model parameters
from utils.functions import (
    crop_img,                     # Crop face regions from full images
    parse_roi_box_from_bbox,      # Convert face bounding boxes to regions of interest
    parse_roi_box_from_landmark,  # Convert face landmarks to regions of interest
)
from utils.tddfa_util import (
    load_model,       # Load pre-trained neural network weights
    _parse_param,     # Parse 62-dimensional 3DMM parameters into components
    similar_transform, # Apply similarity transformations to 3D points
    ToTensorGjz,      # Convert images to PyTorch tensors
    NormalizeGjz      # Normalize image pixel values for neural network input
)

# Helper function to create absolute file paths relative to this script
make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class TDDFA(object):
    """
    TDDFA: Three-D Dense Face Alignment
    
    This is the main class that handles 3D face reconstruction from 2D images.
    
    Key Components:
    1. Neural Network: Predicts 62 3DMM parameters from face images
    2. 3DMM Model: Basel Face Model for representing 3D faces
    3. Image Processing: Handles cropping, resizing, and normalization
    4. 3D Reconstruction: Converts parameters to 3D vertices
    
    The 62 parameters consist of:
    - 12 pose parameters (rotation + translation)
    - 40 shape parameters (face geometry)
    - 10 expression parameters (facial expressions)
    """

    def __init__(self, **kvs):
        """
        Initialize the TDDFA model with configuration parameters.
        
        Args:
            **kvs: Keyword arguments containing configuration:
                - bfm_fp: Path to Basel Face Model file
                - shape_dim: Number of shape basis vectors (default: 40)
                - exp_dim: Number of expression basis vectors (default: 10)
                - gpu_mode: Whether to use GPU acceleration
                - size: Input image size for neural network (default: 120x120)
                - arch: Neural network architecture ('mobilenet_v1', 'resnet18', etc.)
                - checkpoint_fp: Path to pre-trained model weights
        """
        # Disable gradient computation for inference (saves memory and speeds up)
        torch.set_grad_enabled(False)

        # Load the 3D Morphable Model (Basel Face Model)
        # This model contains the mean face shape and basis vectors for variation
        self.bfm = BFMModel(
            bfm_fp=kvs.get('bfm_fp', make_abs_path('configs/bfm_noneck_v3.pkl')),
            shape_dim=kvs.get('shape_dim', 40),    # 40 principal components for shape variation
            exp_dim=kvs.get('exp_dim', 10)         # 10 principal components for expression variation
        )
        # Triangle indices for 3D mesh rendering (defines which vertices form triangles)
        self.tri = self.bfm.tri

        # Hardware and processing configuration
        self.gpu_mode = kvs.get('gpu_mode', False)  # Use GPU if available and requested
        self.gpu_id = kvs.get('gpu_id', 0)          # GPU device ID
        self.size = kvs.get('size', 120)            # Neural network input size (120x120 pixels)

        # Load parameter normalization statistics
        # The neural network outputs normalized parameters that need to be rescaled
        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', make_abs_path(f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl')
        )

        # Initialize and load the neural network model
        # The model architecture can be ResNet, MobileNet, etc.
        model = getattr(models, kvs.get('arch'))(
            num_classes=kvs.get('num_params', 62),     # Output 62 3DMM parameters
            widen_factor=kvs.get('widen_factor', 1),   # Model width multiplier
            size=self.size,                            # Input image size
            mode=kvs.get('mode', 'small')              # Model complexity mode
        )
        
        # Load pre-trained weights from checkpoint file
        model = load_model(model, kvs.get('checkpoint_fp'))

        # Move model to GPU if requested and available
        if self.gpu_mode:
            cudnn.benchmark = True                    # Optimize GPU performance
            model = model.cuda(device=self.gpu_id)

        self.model = model
        self.model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)

        # Setup image preprocessing pipeline
        # Images need to be normalized for the neural network
        transform_normalize = NormalizeGjz(mean=127.5, std=128)  # Normalize to [-1, 1] range
        transform_to_tensor = ToTensorGjz()                      # Convert to PyTorch tensor
        transform = Compose([transform_to_tensor, transform_normalize])
        self.transform = transform

        # Load parameter normalization statistics
        # Neural network outputs need to be denormalized using training statistics
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')  # Mean values for each of the 62 parameters
        self.param_std = r.get('std')    # Standard deviation for each parameter

    def __call__(self, img_ori, objs, **kvs):
        """
        Main inference function - the "brain" of 3D face reconstruction.
        
        This function takes an image and face locations (boxes or landmarks) and returns
        the 3DMM parameters that describe each face in 3D.
        
        Args:
            img_ori: Original input image (BGR format, from cv2.imread)
            objs: List of face locations - can be either:
                  - Bounding boxes: [x1, y1, x2, y2, confidence]
                  - Landmarks: Array of 2D facial landmark points
            **kvs: Optional keyword arguments:
                   - crop_policy: 'box' (use bounding boxes) or 'landmark' (use landmarks)
                   - timer_flag: Whether to measure inference time
                   
        Returns:
            param_lst: List of 62-dimensional parameter arrays (one per face)
                      Format: [pose(12) + shape(40) + expression(10)]
            roi_box_lst: List of refined face bounding boxes after processing
        """
        # Initialize lists to store results for each detected face
        param_lst = []     # Will contain 62 3DMM parameters for each face
        roi_box_lst = []   # Will contain refined bounding boxes for each face

        # Determine how to crop face regions from the full image
        crop_policy = kvs.get('crop_policy', 'box')
        
        # Process each detected face object
        for obj in objs:
            if crop_policy == 'box':
                # Method 1: Use face detection bounding box to define region of interest
                # obj is expected to be [x1, y1, x2, y2, confidence]
                roi_box = parse_roi_box_from_bbox(obj)
            elif crop_policy == 'landmark':
                # Method 2: Use facial landmarks to define region of interest
                # obj is expected to be an array of 2D facial landmark coordinates
                # This is more accurate but requires prior landmark detection
                roi_box = parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            # Store the region of interest box for this face
            roi_box_lst.append(roi_box)
            
            # Step 1: Crop the face region from the full image
            # This extracts just the face area based on the ROI box
            img = crop_img(img_ori, roi_box)
            
            # Step 2: Resize to network input size (120x120 pixels)
            # Neural networks require fixed input sizes
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            
            # Step 3: Preprocess image for neural network
            # Convert to tensor and normalize pixel values
            inp = self.transform(img).unsqueeze(0)  # Add batch dimension [1, 3, 120, 120]

            # Step 4: Move to GPU if using GPU mode
            if self.gpu_mode:
                inp = inp.cuda(device=self.gpu_id)

            # Step 5: Neural network inference
            if kvs.get('timer_flag', False):
                # Measure inference time if requested
                end = time.time()
                param = self.model(inp)  # Forward pass through neural network
                elapse = f'Inference: {(time.time() - end) * 1000:.1f}ms'
                print(elapse)
            else:
                # Standard inference without timing
                param = self.model(inp)  # Get 62 normalized 3DMM parameters

            # Step 6: Post-process the neural network output
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)  # Convert to numpy array
            param = param * self.param_std + self.param_mean  # Denormalize using training statistics
            
            # Store parameters for this face
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        """
        Reconstruct 3D face vertices from 3DMM parameters.
        
        This function converts the abstract 62-dimensional parameters into actual
        3D coordinates of face vertices in the image coordinate system.
        
        Args:
            param_lst: List of 62-dimensional parameter arrays from neural network
            roi_box_lst: List of face bounding boxes corresponding to each parameter set
            **kvs: Optional arguments:
                   - dense_flag: If True, reconstruct ~38k vertices (dense mesh)
                                If False, reconstruct ~68 landmarks (sparse points)
                                
        Returns:
            ver_lst: List of 3D vertex arrays, each shaped as [3, N] where:
                    - 3 rows represent X, Y, Z coordinates
                    - N columns represent different vertices
                    - N ≈ 38,000 for dense reconstruction
                    - N ≈ 68 for sparse landmark reconstruction
        """
        dense_flag = kvs.get('dense_flag', False)  # Choose between dense mesh or sparse landmarks
        size = self.size  # Neural network input size (120x120)

        ver_lst = []  # List to store reconstructed vertices for each face
        
        # Process each face's parameters
        for param, roi_box in zip(param_lst, roi_box_lst):
            # Step 1: Parse the 62 parameters into meaningful components
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            # R: 3x3 rotation matrix (head pose)
            # offset: 3x1 translation vector (head position)
            # alpha_shp: 40x1 shape coefficients (face geometry variation)
            # alpha_exp: 10x1 expression coefficients (facial expression variation)
            
            if dense_flag:
                # Dense reconstruction: Generate full 3D face mesh (~38,000 vertices)
                # Formula: 3D_face = R * (mean_face + shape_basis * shape_coeffs + expr_basis * expr_coeffs) + translation
                pts3d = R @ (self.bfm.u + self.bfm.w_shp @ alpha_shp + self.bfm.w_exp @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                # self.bfm.u: Mean face shape (38k vertices)
                # self.bfm.w_shp: Shape basis vectors (40 components)
                # self.bfm.w_exp: Expression basis vectors (10 components)
            else:
                # Sparse reconstruction: Generate only facial landmarks (~68 points)
                # Uses subset of vertices corresponding to important facial features
                pts3d = R @ (self.bfm.u_base + self.bfm.w_shp_base @ alpha_shp + self.bfm.w_exp_base @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                # self.bfm.u_base: Mean landmark positions (68 points)
                # self.bfm.w_shp_base: Shape basis for landmarks only
                # self.bfm.w_exp_base: Expression basis for landmarks only

            # Step 2: Transform 3D points to image coordinate system
            # The neural network works in normalized coordinates, but we need pixel coordinates
            pts3d = similar_transform(pts3d, roi_box, size)
            
            # Store the reconstructed vertices for this face
            ver_lst.append(pts3d)

        return ver_lst
