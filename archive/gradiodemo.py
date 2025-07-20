# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
# Updated to use relative paths and proper error handling for better portability
import sys
from subprocess import call
import os
import torch

torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Solvay_conference_1927.jpg/1400px-Solvay_conference_1927.jpg', 'solvay.jpg')

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except Exception as e:
        print(f"Errorrrrr: {e}!")

def safe_chdir(path, description=""):
    """Safely change directory with validation"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path} ({description})")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path is not a directory: {path} ({description})")
    os.chdir(path)
    print(f"Changed to: {os.getcwd()} ({description})")

# Get base directory - configurable via environment variable or relative to script
BASE_DIR = os.environ.get('TDDFA_BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Using base directory: {BASE_DIR}")

# Build FaceBoxes NMS extension
facebox_utils_path = os.path.join(BASE_DIR, "FaceBoxes", "utils")
safe_chdir(facebox_utils_path, "FaceBoxes utils")
run_cmd("python3 build.py build_ext --inplace")

# Build Sim3DR extension
sim3dr_path = os.path.join(BASE_DIR, "Sim3DR")
safe_chdir(sim3dr_path, "Sim3DR")
run_cmd("python3 setup.py build_ext --inplace")

# Build render extension
utils_asset_path = os.path.join(BASE_DIR, "utils", "asset")
safe_chdir(utils_asset_path, "utils/asset")
run_cmd("gcc -shared -Wall -O3 render.c -o render.so -fPIC")

# Return to base directory
safe_chdir(BASE_DIR, "base directory")


import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix

import matplotlib.pyplot as plt
from skimage import io
import gradio as gr

# load config
config_path = os.path.join(BASE_DIR, 'configs', 'mb1_120x120.yml')
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True  # or True to use ONNX to speed up
if onnx_flag:    
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **cfg)


def inference(img):
    # face detection
    boxes = face_boxes(img)
    # regress 3DMM params
    param_lst, roi_box_lst = tddfa(img, boxes)
    # reconstruct vertices and render
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    return render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=False)


title = "3DDFA V2"
description = "demo for 3DDFA V2. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2009.09960'>Towards Fast, Accurate and Stable 3D Dense Face Alignment</a> | <a href='https://github.com/cleardusk/3DDFA_V2'>Github Repo</a></p>"
examples = [
    ['solvay.jpg']
]

gr.Interface(
    inference, 
    [gr.inputs.Image(type="numpy", label="Input")], 
    gr.outputs.Image(type="numpy", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=examples
).launch()
