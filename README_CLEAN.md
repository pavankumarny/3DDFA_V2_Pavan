# 3DDFA_V2 Pose Extraction - Clean Setup Guide

## 📁 Project Structure

```
3DDFA_V2/
├── tools/                    # 🔧 Your custom tools
│   └── pose_extractor.py    # Main pose extraction tool
├── results/                 # 📊 All output files go here
├── tests/                   # 🧪 Test scripts and demos
├── examples/                # 📷 Sample data (original)
├── weights/                 # 🤖 Pre-trained models
├── configs/                 # ⚙️  Model configurations
└── utils/                   # 🛠️  Core 3DDFA_V2 utilities
```

## 🚀 Quick Start

### **Method 1: From Main Directory (Recommended)**

```bash
# Test on sample image
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg

# Process Lit.mp4 video
python3 tools/pose_extractor.py --mode video \
    -f examples/inputs/videos/Lit.mp4 \
    --landmarks \
    -o results/Lit_with_poses.mp4 \
    --csv results/Lit_pose_data.csv

# Real-time webcam
python3 tools/pose_extractor.py --mode webcam --landmarks
```

### **Method 2: From Tools Directory**

```bash
cd tools
python3 pose_extractor.py --mode image -f ../examples/inputs/emma.jpg
python3 pose_extractor.py --mode webcam --landmarks
```

## ✅ Quick Verification

To verify everything is working correctly:

```bash
cd /Users/pavankumar/Documents/Research/3DDFA_V2

# Test image processing
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg

# Expected output:
# ✅ Pose extracted:
#    Pitch:   12.0° (up/down)
#    Yaw:     33.1° (left/right)
#    Roll:    -4.8° (tilt)
```

## 📊 Output Format

### **CSV Data Structure:**

```csv
frame,timestamp,pitch,yaw,roll
0,0.000,-5.2,12.8,2.1
1,0.033,-5.1,12.9,2.0
...
```

### **Angle Interpretation:**

- **Pitch**: Up/Down head movement
  - `+` = Looking up, `-` = Looking down
- **Yaw**: Left/Right head turn
  - `+` = Turned right, `-` = Turned left
- **Roll**: Head tilt
  - `+` = Tilted right, `-` = Tilted left

## 🎯 Key Features

- ✅ **High Accuracy**: Uses proven 3DDFA_V2 algorithms
- ✅ **Real-time Performance**: ONNX optimization
- ✅ **Multiple Formats**: Images, videos, webcam
- ✅ **Clean Output**: All results in `results/` folder
- ✅ **Professional Visualization**: Pose overlays and landmarks
- ✅ **Data Export**: CSV for analysis

## 💡 Pro Tips

1. **For best accuracy**: Use well-lit, clear face images
2. **For faster processing**: Videos process much faster without `--landmarks`
3. **For analysis**: Always use `--csv` to export data
4. **File organization**: All outputs automatically go to `results/` folder

## 🔧 Advanced Usage

### **Batch Process Multiple Videos**

```bash
cd /Users/pavankumar/Documents/Research/3DDFA_V2
for video in examples/inputs/videos/*.mp4; do
    name=$(basename "$video" .mp4)
    python3 tools/pose_extractor.py --mode video \
        -f "$video" \
        -o "results/${name}_processed.mp4" \
        --csv "results/${name}_data.csv"
done
```

### **High-Quality Processing**

```bash
# Use PyTorch models for maximum accuracy (slower)
python3 tools/pose_extractor.py --mode video \
    -f examples/inputs/videos/Lit.mp4 \
    --pytorch \
    --landmarks \
    -o results/Lit_high_quality.mp4 \
    --csv results/Lit_detailed_data.csv
```

## 📋 File Locations

- **Main tool**: `tools/pose_extractor.py`
- **All outputs**: `results/` folder
- **Test scripts**: `tests/` folder
- **Sample data**: `examples/` folder

This clean structure keeps everything organized and professional! 🎯
