import sys
import os
sys.path.append('.')

from tools.pose_extractor import PoseExtractor
import cv2
import numpy as np

# Test on a simple image
extractor = PoseExtractor(use_onnx=False)
frame = cv2.imread('examples/inputs/emma.jpg')

result = extractor.process_frame(frame)

if result['success']:
    print(f"Emma's pose angles:")
    print(f"  Pitch: {result['pitch']:.1f}° (should be ~0° if looking straight)")
    print(f"  Yaw:   {result['yaw']:.1f}° (should be ~0° if facing camera)")
    print(f"  Roll:  {result['roll']:.1f}° (should be ~0° if head level)")
    print()
    print("Problem analysis:")
    if abs(result['pitch']) > 45:
        print(f"  ⚠️  Pitch {result['pitch']:.1f}° is too extreme for normal face pose")
    if abs(result['yaw']) > 45:
        print(f"  ⚠️  Yaw {result['yaw']:.1f}° is too extreme for normal face pose")
    if abs(result['roll']) > 45:
        print(f"  ⚠️  Roll {result['roll']:.1f}° is too extreme for normal face pose")
        
    print("\nThis suggests the coordinate system or angle interpretation is wrong.")
else:
    print("No face detected")
