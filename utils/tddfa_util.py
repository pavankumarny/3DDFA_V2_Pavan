# coding: utf-8
"""
TDDFA Utility Functions
======================

This module contains essential utility functions for the 3DDFA_V2 system:
1. Model loading and parameter parsing
2. Image preprocessing transformations  
3. Coordinate system transformations
4. 3DMM parameter interpretation

These utilities handle the mathematical operations needed to convert between
different coordinate systems and parameter representations.
"""

__author__ = 'cleardusk'

import sys
sys.path.append('..')

import argparse        # Command line argument parsing
import numpy as np     # Numerical operations
import torch          # PyTorch tensor operations


def _to_ctype(arr):
    """
    Ensure numpy array is C-contiguous for efficient memory access.
    
    Args:
        arr: Input numpy array
        
    Returns:
        C-contiguous copy of the array if needed, otherwise original array
    """
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def str2bool(v):
    """
    Convert string representations to boolean values.
    Useful for command line argument parsing.
    
    Args:
        v: String value to convert
        
    Returns:
        Boolean value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def load_model(model, checkpoint_fp):
    """
    Load pre-trained weights into a PyTorch model.
    
    This function handles loading weights that may have been trained on multiple GPUs
    (which adds 'module.' prefix) and maps them correctly to single GPU/CPU models.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_fp: Path to the checkpoint file (.pth)
        
    Returns:
        Model with loaded weights
    """
    # Load checkpoint file
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model_dict = model.state_dict()
    
    # Handle multi-GPU training artifacts: remove 'module.' prefix
    for k in checkpoint.keys():
        kc = k.replace('module.', '')  # Remove DataParallel prefix
        if kc in model_dict.keys():
            model_dict[kc] = checkpoint[k]
        # Handle special parameter renaming if needed
        if kc in ['fc_param.bias', 'fc_param.weight']:
            model_dict[kc.replace('_param', '')] = checkpoint[k]

    # Load the processed weights into the model
    model.load_state_dict(model_dict)
    return model


class ToTensorGjz(object):
    """
    Convert numpy arrays to PyTorch tensors with proper channel ordering.
    
    This transform converts images from HWC (Height-Width-Channels) format 
    to CHW (Channels-Height-Width) format expected by PyTorch models.
    """
    
    def __call__(self, pic):
        """
        Convert numpy array image to PyTorch tensor.
        
        Args:
            pic: Numpy array in HWC format (e.g., 120x120x3)
            
        Returns:
            PyTorch tensor in CHW format (e.g., 3x120x120)
        """
        if isinstance(pic, np.ndarray):
            # Transpose from HWC to CHW and convert to tensor
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    """
    Normalize tensor values for neural network input.
    
    This normalization converts pixel values from [0, 255] range to a normalized
    range suitable for neural network training/inference.
    """
    
    def __init__(self, mean, std):
        """
        Initialize normalization parameters.
        
        Args:
            mean: Mean value to subtract (typically 127.5 for [0,255] -> [-1,1])
            std: Standard deviation to divide by (typically 128 for [0,255] -> [-1,1])
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Apply normalization: (tensor - mean) / std
        
        Args:
            tensor: Input tensor to normalize
            
        Returns:
            Normalized tensor
        """
        tensor.sub_(self.mean).div_(self.std)  # In-place operations for efficiency
        return tensor


def similar_transform(pts3d, roi_box, size):
    """
    Transform 3D points from normalized coordinates to image pixel coordinates.
    
    The neural network works with normalized coordinates, but we need to transform
    the reconstructed 3D points back to the original image coordinate system.
    
    Args:
        pts3d: 3D points in normalized coordinates, shape (3, N)
               Row 0: X coordinates
               Row 1: Y coordinates  
               Row 2: Z coordinates (depth)
        roi_box: Face region bounding box [x1, y1, x2, y2]
        size: Neural network input size (typically 120)
        
    Returns:
        Transformed 3D points in image pixel coordinates
    """
    # Adjust for coordinate system differences
    pts3d[0, :] -= 1  # X-coordinate adjustment for Python/OpenCV compatibility
    pts3d[2, :] -= 1  # Z-coordinate adjustment
    pts3d[1, :] = size - pts3d[1, :]  # Flip Y-axis (neural network vs image coordinates)

    # Extract bounding box parameters
    sx, sy, ex, ey = roi_box  # Start X, Start Y, End X, End Y
    
    # Calculate scaling factors to map from network size to actual face region size
    scale_x = (ex - sx) / size  # How much wider the actual face region is vs network input
    scale_y = (ey - sy) / size  # How much taller the actual face region is vs network input
    
    # Transform X and Y coordinates from normalized space to image pixel space
    pts3d[0, :] = pts3d[0, :] * scale_x + sx  # Scale and translate X coordinates
    pts3d[1, :] = pts3d[1, :] * scale_y + sy  # Scale and translate Y coordinates
    
    # Handle Z coordinates (depth) - use average scaling to maintain proportions
    s = (scale_x + scale_y) / 2  # Average scale factor
    pts3d[2, :] *= s             # Scale depth values
    pts3d[2, :] -= np.min(pts3d[2, :])  # Normalize depth to start from 0
    
    return np.array(pts3d, dtype=np.float32)


def _parse_param(param):
    """
    Parse the 62-dimensional 3DMM parameter vector into meaningful components.
    
    The neural network outputs a single vector of 62 parameters that encodes
    all information needed to reconstruct a 3D face. This function splits
    these parameters into their semantic components.
    
    Args:
        param: 1D array of 3DMM parameters
               Standard format: 62 = 12 (pose) + 40 (shape) + 10 (expression)
    
    Returns:
        R: 3x3 rotation matrix (head pose rotation)
        offset: 3x1 translation vector (head position)
        alpha_shp: 40x1 shape coefficients (face geometry: wide/narrow, etc.)
        alpha_exp: 10x1 expression coefficients (facial expression: smile, etc.)
    """
    # Determine parameter layout based on total length
    n = param.shape[0]
    if n == 62:
        # Standard configuration: 12 + 40 + 10
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        # Extended expression model: 12 + 40 + 20
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        # High-resolution model: 12 + 100 + 29
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined parameter parsing rule for {n} parameters')

    # Extract and reshape pose parameters (first 12 values)
    # These 12 values encode both rotation (9 values for 3x3 matrix) and translation (3 values)
    R_ = param[:trans_dim].reshape(3, -1)  # Reshape to 3x4 matrix
    R = R_[:, :3]                          # First 3 columns: 3x3 rotation matrix
    offset = R_[:, -1].reshape(3, 1)       # Last column: 3x1 translation vector
    
    # Extract shape parameters (next 40 values)
    # These control face geometry: face width, nose size, chin shape, etc.
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    
    # Extract expression parameters (remaining 10 values)  
    # These control facial expressions: smile, frown, raised eyebrows, etc.
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp
