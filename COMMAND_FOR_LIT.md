# 🎯 FINAL COMMAND FOR LIT.MP4

## Your Clean, Organized Command:

```bash
cd /Users/pavankumar/Documents/Research/3DDFA_V2

python3 tools/pose_extractor.py --mode video \
    -f examples/inputs/videos/Lit.mp4 \
    --landmarks \
    -o results/Lit_with_poses_and_landmarks.mp4 \
    --csv results/Lit_pose_data.csv
```

## What This Will Do:

✅ **Process** `Lit.mp4` with high accuracy
✅ **Show landmarks** (green facial points)  
✅ **Display pose angles** (pitch/yaw/roll) on each frame
✅ **Save video** to `results/Lit_with_poses_and_landmarks.mp4`
✅ **Export data** to `results/Lit_pose_data.csv`

## Results Location:

- 📹 **Annotated Video**: `results/Lit_with_poses_and_landmarks.mp4`
- 📊 **Pose Data**: `results/Lit_pose_data.csv`

## CSV Data Format:

```csv
frame,timestamp,pitch,yaw,roll
0,0.000,-5.2,12.8,2.1
1,0.033,-5.1,12.9,2.0
...
```

## Your Clean File Structure:

```
3DDFA_V2/
├── tools/pose_extractor.py    ← Main tool (clean & professional)
├── results/                   ← All your outputs go here
├── tests/                     ← Test scripts
├── archive/                   ← Old demo files (organized)
└── examples/                  ← Original sample data
```

🎉 **Everything is now clean, organized, and professional!**
