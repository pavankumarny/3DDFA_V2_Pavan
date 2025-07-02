# coding: utf-8
"""
FaceBoxes: Lightweight and Fast Face Detection
==============================================

FaceBoxes is a CNN-based face detector optimized for speed and accuracy.
It's designed to quickly find face bounding boxes in images, which are then
passed to TDDFA for 3D face reconstruction.

Key Features:
- Real-time face detection
- High accuracy across different face sizes
- Optimized for both CPU and GPU inference
- Returns face bounding boxes with confidence scores

This module serves as the first stage in the 3DDFA_V2 pipeline:
Image → FaceBoxes (detect faces) → TDDFA (3D reconstruction)
"""

import os.path as osp    # File path operations

# Deep learning and computer vision libraries
import torch             # PyTorch for neural network inference
import numpy as np       # Numerical operations
import cv2              # Computer vision operations

# FaceBoxes-specific modules
from .utils.prior_box import PriorBox        # Generate anchor boxes for detection
from .utils.nms_wrapper import nms          # Non-Maximum Suppression to remove duplicate detections
from .utils.box_utils import decode         # Decode network predictions to bounding boxes
from .utils.timer import Timer              # Performance timing utilities
from .utils.functions import check_keys, remove_prefix, load_model  # Model loading utilities
from .utils.config import cfg               # Configuration parameters
from .models.faceboxes import FaceBoxesNet  # Neural network architecture

# Global detection parameters - these control detection behavior
confidence_threshold = 0.05   # Minimum confidence score to consider a detection
top_k = 5000                  # Maximum detections to consider before NMS
keep_top_k = 750             # Maximum detections to keep after NMS
nms_threshold = 0.3          # Non-Maximum Suppression threshold (lower = more aggressive)
vis_thres = 0.5              # Minimum confidence for final output (higher = more conservative)
resize = 1                   # Image resize factor (not used in current implementation)

# Image scaling parameters for performance optimization
scale_flag = True            # Whether to scale large images for faster processing
HEIGHT, WIDTH = 720, 1080    # Maximum dimensions before scaling kicks in

# Helper function to create absolute paths relative to this file
make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
pretrained_path = make_abs_path('weights/FaceBoxesProd.pth')  # Path to pre-trained model weights


def viz_bbox(img, dets, wfp='out.jpg'):
    """
    Visualize detected face bounding boxes on an image.
    
    Args:
        img: Input image (BGR format)
        dets: List of detections, each as [x1, y1, x2, y2, confidence]
        wfp: Output file path for the visualization
    """
    # Draw each detection on the image
    for b in dets:
        if b[4] < vis_thres:  # Skip low-confidence detections
            continue
        text = "{:.4f}".format(b[4])  # Format confidence score
        b = list(map(int, b))         # Convert to integers for drawing
        
        # Draw rectangle around face
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        
        # Add confidence score text
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    
    cv2.imwrite(wfp, img)
    print(f'Visualization saved to {wfp}')


class FaceBoxes:
    """
    FaceBoxes face detector class.
    
    This class implements a fast and lightweight face detection system
    based on the FaceBoxes architecture. It processes images and returns
    bounding boxes around detected faces with confidence scores.
    
    The detection pipeline:
    1. Preprocess image (resize if needed, normalize)
    2. Run neural network to get face predictions
    3. Decode predictions to bounding boxes
    4. Apply Non-Maximum Suppression to remove duplicates
    5. Filter by confidence threshold
    """
    
    def __init__(self, timer_flag=False):
        """
        Initialize the FaceBoxes detector.
        
        Args:
            timer_flag: Whether to print timing information during inference
        """
        # Disable gradient computation for inference (saves memory and speeds up)
        torch.set_grad_enabled(False)

        # Load the FaceBoxes neural network
        # phase='test' means inference mode (vs training mode)
        # num_classes=2 means background + face classes
        net = FaceBoxesNet(phase='test', size=None, num_classes=2)
        
        # Load pre-trained weights and set to CPU mode
        self.net = load_model(net, pretrained_path=pretrained_path, load_to_cpu=True)
        self.net.eval()  # Set to evaluation mode (disables dropout, etc.)

        self.timer_flag = timer_flag  # Whether to show timing information

    def __call__(self, img_):
        """
        Detect faces in an input image.
        
        Args:
            img_: Input image in BGR format (from cv2.imread)
            
        Returns:
            det_bboxes: List of detected face bounding boxes
                       Each detection: [x1, y1, x2, y2, confidence]
                       where (x1,y1) is top-left, (x2,y2) is bottom-right
        """
        img_raw = img_.copy()  # Make a copy to avoid modifying original

        # STEP 1: Image scaling for performance optimization
        # Large images are scaled down to speed up detection
        scale = 1
        if scale_flag:
            h, w = img_raw.shape[:2]
            # Scale down if image is too large
            if h > HEIGHT:
                scale = HEIGHT / h
            if w * scale > WIDTH:
                scale *= WIDTH / (w * scale)
            
            if scale == 1:
                img_raw_scale = img_raw  # No scaling needed
            else:
                # Resize image for faster processing
                h_s = int(scale * h)
                w_s = int(scale * w)
                img_raw_scale = cv2.resize(img_raw, dsize=(w_s, h_s))

            img = np.float32(img_raw_scale)  # Convert to float32 for neural network
        else:
            img = np.float32(img_raw)

        # STEP 2: Neural network inference
        _t = {'forward_pass': Timer(), 'misc': Timer()}  # Timing objects
        im_height, im_width, _ = img.shape
        
        # Scale factor for converting normalized coordinates back to pixel coordinates
        scale_bbox = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        
        # STEP 3: Image preprocessing for neural network
        img -= (104, 117, 123)      # Subtract mean values (BGR format)
        img = img.transpose(2, 0, 1)  # Change from HWC to CHW format
        img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension: [1, C, H, W]

        # STEP 4: Forward pass through neural network
        _t['forward_pass'].tic()
        loc, conf = self.net(img)  # Get location predictions and confidence scores
        _t['forward_pass'].toc()
        
        # STEP 5: Post-processing to get final bounding boxes
        _t['misc'].tic()
        
        # Generate anchor boxes (prior boxes) for the current image size
        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data
        
        # Decode network predictions into actual bounding box coordinates
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        
        # Scale boxes back to original image coordinates
        if scale_flag:
            boxes = boxes * scale_bbox / scale / resize
        else:
            boxes = boxes * scale_bbox / resize

        boxes = boxes.cpu().numpy()                    # Convert to numpy
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]  # Get face confidence scores

        # STEP 6: Filter out low-confidence detections
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # STEP 7: Keep only top-K highest confidence detections
        order = scores.argsort()[::-1][:top_k]  # Sort by confidence, keep top K
        boxes = boxes[order]
        scores = scores[order]

        # STEP 8: Apply Non-Maximum Suppression (NMS)
        # Remove duplicate detections of the same face
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, nms_threshold)  # Apply NMS
        dets = dets[keep, :]

        # STEP 9: Keep only the top detections after NMS
        dets = dets[:keep_top_k, :]
        _t['misc'].toc()

        # Print timing information if requested
        if self.timer_flag:
            print('Detection: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(
                1, 1, _t['forward_pass'].average_time, _t['misc'].average_time))

        # STEP 10: Final filtering by visualization threshold
        det_bboxes = []
        for b in dets:
            if b[4] > vis_thres:  # Only keep high-confidence detections
                xmin, ymin, xmax, ymax, score = b[0], b[1], b[2], b[3], b[4]
                bbox = [xmin, ymin, xmax, ymax, score]
                det_bboxes.append(bbox)

        return det_bboxes


def main():
    face_boxes = FaceBoxes(timer_flag=True)

    fn = 'trump_hillary.jpg'
    img_fp = f'../examples/inputs/{fn}'
    img = cv2.imread(img_fp)
    print(f'input shape: {img.shape}')
    dets = face_boxes(img)  # xmin, ymin, w, h
    # print(dets)

    # repeating inference for `n` times
    n = 10
    for i in range(n):
        dets = face_boxes(img)

    wfn = fn.replace('.jpg', '_det.jpg')
    wfp = osp.join('../examples/results', wfn)
    viz_bbox(img, dets, wfp)


if __name__ == '__main__':
    main()
