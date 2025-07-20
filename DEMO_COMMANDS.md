# 🎯 3DDFA_V2 Pose Extraction Demo Commands

## Quick Demo: Landmarks + Pitch/Yaw/Roll Extraction

### 📷 **1. Image Processing with Landmarks**

```bash
cd /Users/pavankumar/Documents/Research/3DDFA_V2

# Basic image with pose angles
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg

# Image with landmarks visualization
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg --landmarks

# Image with CSV data export
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg --landmarks --csv results/emma_demo.csv

# Custom output with everything
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg \
    --landmarks \
    -o results/emma_landmarks_demo.jpg \
    --csv results/emma_pose_data.csv
```

### 🎥 **2. Video Processing with Frame-by-Frame Data**

```bash
# Process video with landmarks and full data export
python3 tools/pose_extractor.py --mode video \
    -f examples/inputs/videos/Lit.mp4 \
    --landmarks \
    -o results/Lit_landmarks_demo.mp4 \
    --csv results/Lit_frame_data.csv

# Quick video processing (no landmarks for speed)
python3 tools/pose_extractor.py --mode video \
    -f examples/inputs/videos/Lit.mp4 \
    -o results/Lit_pose_only.mp4 \
    --csv results/Lit_angles.csv
```

### 📹 **3. Real-time Webcam Demo**

```bash
# Live webcam with landmarks (press 'q' to quit, 's' to save frame)
python3 tools/pose_extractor.py --mode webcam --landmarks

# Basic webcam (faster performance)
python3 tools/pose_extractor.py --mode webcam
```

## 📊 **Expected Output Data**

### **Console Output Example:**

```
✅ Pose extracted:
   Pitch:   12.0° (up/down)
   Yaw:     33.1° (left/right)
   Roll:    -4.8° (tilt)
```

### **CSV Data Format:**

```csv
frame,timestamp,pitch,yaw,roll
0,0.000,12.0,33.1,-4.8
1,0.033,11.8,33.2,-4.7
2,0.067,11.9,33.0,-4.9
```

### **Image Output:**

- ✅ **Pose angles** displayed as text overlay
- ✅ **Facial landmarks** (68 green points when `--landmarks` used)
- ✅ **Color-coded status** (Green=neutral, Yellow=high, Cyan=low)

## 🎯 **Quick Test Sequence**

Run these commands in order for a complete demonstration:

```bash
# Step 1: Navigate to project
cd /Users/pavankumar/Documents/Research/3DDFA_V2

# Step 2: Test image processing
echo "🔹 Testing image processing..."
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg --landmarks --csv results/demo_test.csv

# Step 3: Check the results
echo "🔹 Results generated:"
ls -la results/demo_test.csv
cat results/demo_test.csv

# Step 4: View output image
echo "🔹 Output image location:"
ls -la examples/inputs/emma_pose.jpg
```

## 📋 **Data Interpretation Guide**

### **Pitch (Up/Down):**

- `+` values = Looking UP
- `-` values = Looking DOWN
- `0°` = Looking straight ahead

### **Yaw (Left/Right):**

- `+` values = Head turned RIGHT
- `-` values = Head turned LEFT
- `0°` = Facing camera directly

### **Roll (Tilt):**

- `+` values = Head tilted RIGHT
- `-` values = Head tilted LEFT
- `0°` = Head perfectly level

### **Landmarks:**

- 68 facial landmark points
- Green dots when visualized
- Follow facial contours and features

## 🚀 **Performance Options**

```bash
# Fastest (ONNX - default)
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg

# Highest accuracy (PyTorch)
python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg --pytorch
```

This demonstrates the complete pose extraction pipeline with all three data points and landmarks! 🎉
