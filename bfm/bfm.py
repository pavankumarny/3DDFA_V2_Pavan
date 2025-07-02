# coding: utf-8
"""
Basel Face Model (BFM) Implementation
====================================

The Basel Face Model is a 3D Morphable Model (3DMM) that represents human faces
as linear combinations of shape and expression basis vectors.

Key Concepts:
- Mean Face (u): The average 3D face shape
- Shape Basis (w_shp): Principal components for identity variation (wide vs narrow face, etc.)
- Expression Basis (w_exp): Principal components for facial expressions (smile, frown, etc.)
- Linear Model: Any face = mean_face + shape_coeffs * shape_basis + expr_coeffs * expr_basis

This mathematical representation allows us to describe any face with just:
- 40 shape parameters (identity/geometry)
- 10 expression parameters (emotions/expressions)
- 12 pose parameters (head rotation/translation)
"""

__author__ = 'cleardusk'

import sys
sys.path.append('..')

import os.path as osp       # File path operations
import numpy as np          # Numerical operations for 3D mathematics
from utils.io import _load  # Utility to load model data files

# Helper function to create absolute paths relative to this file
make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def _to_ctype(arr):
    """
    Ensure array is C-contiguous for efficient memory access.
    
    C-contiguous arrays have elements stored in row-major order,
    which is more efficient for many operations.
    """
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class BFMModel(object):
    """
    Basel Face Model - 3D Morphable Model for Face Representation
    
    This class loads and manages the 3D morphable face model that enables
    reconstruction of any human face from a small set of parameters.
    
    The model contains:
    1. Mean face shape (u): Average 3D coordinates of ~38,000 face vertices
    2. Shape basis (w_shp): 40 principal components for identity variation
    3. Expression basis (w_exp): 10 principal components for expression variation
    4. Triangulation (tri): How vertices connect to form 3D mesh surfaces
    5. Keypoints: Subset of vertices corresponding to facial landmarks
    
    Mathematical Formula:
    face_3d = u + w_shp @ alpha_shape + w_exp @ alpha_expression
    
    Where:
    - u: mean face (3N x 1)
    - w_shp: shape basis (3N x 40)  
    - w_exp: expression basis (3N x 10)
    - alpha_shape: 40 shape coefficients
    - alpha_expression: 10 expression coefficients
    - N: number of vertices (~38,000 for dense, ~68 for sparse)
    """
    
    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        """
        Initialize the Basel Face Model.
        
        Args:
            bfm_fp: Path to the BFM data file (.pkl format)
            shape_dim: Number of shape basis vectors to use (default: 40)
            exp_dim: Number of expression basis vectors to use (default: 10)
        """
        # Load the pre-computed BFM data
        bfm = _load(bfm_fp)
        
        # Mean face shape: 3D coordinates of all vertices in neutral expression
        # Shape: (3*N, 1) where N is number of vertices (~38k)
        # Data is flattened: [x1,y1,z1, x2,y2,z2, ...]
        self.u = bfm.get('u').astype(np.float32)
        
        # Shape basis vectors: Principal components for identity variation
        # Shape: (3*N, 40) - each column is a shape component
        # These control things like: face width, nose size, chin shape, etc.
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        
        # Expression basis vectors: Principal components for facial expressions  
        # Shape: (3*N, 10) - each column is an expression component
        # These control things like: smile, frown, raised eyebrows, etc.
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        
        # Load triangulation data (how vertices connect to form mesh)
        if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
            # Special case: re-built triangulation for neck-less model
            self.tri = _load(make_abs_path('../configs/tri.pkl'))
        else:
            # Standard triangulation from BFM file
            self.tri = bfm.get('tri')

        # Convert triangulation to C-contiguous format for efficiency
        # Shape: (num_triangles, 3) - each row defines one triangle face
        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        
        # Keypoint indices: subset of vertices corresponding to facial landmarks
        # These are the ~68 important points (eye corners, nose tip, mouth, etc.)
        self.keypoints = bfm.get('keypoints').astype(np.long)

        # Compute normalization factors for the basis vectors
        # This helps with numerical stability during reconstruction
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        # Extract sparse versions (landmarks only) for faster processing
        # These are used when dense_flag=False in reconstruction
        self.u_base = self.u[self.keypoints].reshape(-1, 1)      # Mean landmark positions
        self.w_shp_base = self.w_shp[self.keypoints]             # Shape basis for landmarks
        self.w_exp_base = self.w_exp[self.keypoints]             # Expression basis for landmarks
