# 3DDFA_V2 Perfect Pose Extraction Guide

## Complete Setup and Usage Instructions

### 🎯 What You'll Get

This guide will help you extract **perfect pitch, yaw, and roll angles** from:

- 📷 **Images** (single or batch)
- 🎥 **Videos** (with frame-by-frame tracking)
- 📹 **Live webcam** (real-time estimation)

### 📋 Prerequisites

1. **Python 3.8+** (you have Python 3.13.5 ✅)
2. **macOS** (your system ✅)
3. **Required packages** (install below)

### 🛠 Installation Steps

#### Step 1: Install Required Packages

```bash
# Core dependencies
pip install --user torch torchvision opencv-python numpy pyyaml scipy
pip install --user onnxruntime imageio scikit-image matplotlib

# Optional for video processing
pip install --user imageio-ffmpeg
```

#### Step 2: Verify Installation

```bash
cd /Users/pavankumar/Documents/Research/3DDFA_V2
python3 -c "import torch, cv2, numpy, yaml, onnxruntime; print('✅ All packages installed')"
```

#### Step 3: Test Basic Functionality

```bash
# Test on the sample Emma image
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg

# This should output:
# ✅ Pose extracted:
#    Pitch: XX.XX° (up/down)
#    Yaw:   XX.XX° (left/right)
#    Roll:  XX.XX° (tilt)
```

### 🚀 Usage Examples

#### 1. **Single Image Processing**

```bash
# Basic image processing
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg

# With landmarks visualization
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg --landmarks

# Save results to CSV
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg --csv results/pose_data.csv

# Custom output filename
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg -o results/my_result.jpg
```

#### 2. **Real-time Webcam Processing**

```bash
# Basic webcam mode
python3 tools/pose_extractor.py --mode webcam

# With facial landmarks
python3 tools/pose_extractor.py --mode webcam --landmarks

# Controls:
# - Press 'q' to quit
# - Press 's' to save current frame
```

#### 3. **Video Processing**

```bash
# Process video file
python3 tools/pose_extractor.py --mode video -f examples/inputs/videos/Lit.mp4

# Save processed video with pose overlay
python3 tools/pose_extractor.py --mode video -f examples/inputs/videos/Lit.mp4 -o results/output_with_poses.mp4

# Export pose data to CSV for analysis
python3 tools/pose_extractor.py --mode video -f examples/inputs/videos/Lit.mp4 --csv results/pose_timeline.csv

# Complete processing with all outputs
python3 tools/pose_extractor.py --mode video -f examples/inputs/videos/Lit.mp4 -o results/output.mp4 --csv results/data.csv --landmarks
```

### 📊 Understanding the Output

#### **Pose Angles Explained:**

1. **PITCH** (Up/Down movement)

   - **Positive values**: Looking up (e.g., +20° = looking up)
   - **Negative values**: Looking down (e.g., -15° = looking down)
   - **Zero**: Looking straight ahead

2. **YAW** (Left/Right turn)

   - **Positive values**: Head turned right (e.g., +30° = turned right)
   - **Negative values**: Head turned left (e.g., -25° = turned left)
   - **Zero**: Facing camera directly

3. **ROLL** (Head tilt)
   - **Positive values**: Head tilted right (e.g., +10° = right ear toward shoulder)
   - **Negative values**: Head tilted left (e.g., -12° = left ear toward shoulder)
   - **Zero**: Head level

#### **Status Indicators:**

- **NEUTRAL**: Angle between -15° and +15°
- **LOW/HIGH**: Angle beyond ±15° threshold
- **Color coding**: Green = neutral, Yellow = high, Light red = low

### 📈 Performance Optimization

#### **ONNX Mode (Default - Fastest)**

```bash
python3 tools/pose_extractor.py --mode webcam
# Uses ONNX optimization for ~1.35ms per face
```

#### **PyTorch Mode (Fallback)**

```bash
python3 tools/pose_extractor.py --mode webcam --pytorch
# Uses PyTorch models instead of ONNX
```

### 🔧 Advanced Usage

#### **Batch Processing Multiple Images**

```bash
# Process all images in a directory
find examples/inputs -name "*.jpg" -exec python3 tools/pose_extractor.py --mode image -f {} \;

# Or create a batch script
for img in examples/inputs/*.jpg; do
    python3 tools/pose_extractor.py --mode image -f "$img" --csv "results/batch_results.csv"
done
```

#### **Video Analysis Pipeline**

```bash
# Step 1: Extract poses from video
python3 tools/pose_extractor.py --mode video -f examples/inputs/videos/Lit.mp4 --csv results/poses.csv

# Step 2: Analyze the CSV data
python3 -c "
import pandas as pd
df = pd.read_csv('results/poses.csv')
print('Pose Statistics:')
print(f'Pitch range: {df.pitch.min():.1f}° to {df.pitch.max():.1f}°')
print(f'Yaw range: {df.yaw.min():.1f}° to {df.yaw.max():.1f}°')
print(f'Roll range: {df.roll.min():.1f}° to {df.roll.max():.1f}°')
"
```

### 🎯 Perfect Pose Extraction Tips

1. **For Best Accuracy:**

   - Use well-lit environments
   - Ensure face is clearly visible
   - Avoid extreme angles (>90°)
   - Keep face size reasonable in frame

2. **For Video Processing:**

   - Use stable camera position
   - Avoid rapid movements
   - Process at original resolution

3. **For Real-time Applications:**
   - Use ONNX mode for speed
   - Consider reducing frame rate if needed
   - Monitor CPU usage

### 📋 Troubleshooting

#### **Common Issues:**

1. **"No face detected"**

   - Check lighting conditions
   - Ensure face is clearly visible
   - Try different camera angles

2. **Slow performance**

   - Use ONNX mode (default)
   - Close other applications
   - Consider lower resolution input

3. **Import errors**

   - Reinstall packages: `pip install --user package_name`
   - Check Python version compatibility

4. **Model loading issues**
   - Verify all model files are present in weights/
   - Check network connection for downloads

### 🎉 You're Ready!

Your 3DDFA_V2 system is now set up for perfect pose extraction. The system provides:

- ✅ **High accuracy** pose estimation
- ✅ **Real-time performance** with ONNX
- ✅ **Multiple input modes** (image/video/webcam)
- ✅ **Data export** capabilities
- ✅ **Visual feedback** with pose visualization
- ✅ **Professional-grade** tracking and stability

### 📞 Quick Start Commands

```bash
# Test the system
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg

# Real-time webcam
python3 tools/pose_extractor.py --mode webcam

# Process your video
python3 tools/pose_extractor.py --mode video -f examples/inputs/videos/Lit.mp4 -o results/output.mp4 --csv results/data.csv
```

Now you can extract perfect pitch, yaw, and roll angles from any footage! 🎯
