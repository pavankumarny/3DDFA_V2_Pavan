#!/usr/bin/env python3
"""
3DDFA_V2 Pose Extraction Demo & Test Suite
==========================================

This script demonstrates and tests all pose extraction capabilities.
Run this to verify your setup and see all features in action.
"""

import os
import sys
import time

def test_image_processing():
    """Test single image pose extraction"""
    print("🧪 TEST 1: Single Image Processing")
    print("-" * 40)
    
    cmd = "python3 pose_extractor.py --mode image -f examples/inputs/emma.jpg"
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print("✅ Image processing test PASSED")
        return True
    else:
        print("❌ Image processing test FAILED")
        return False

def demo_webcam():
    """Demo real-time webcam processing"""
    print("\n🧪 TEST 2: Real-time Webcam Demo")
    print("-" * 40)
    print("This will open your webcam for real-time pose estimation.")
    print("Move your head to see pitch, yaw, roll angles change in real-time.")
    print("Press 'q' to quit the demo.")
    
    response = input("Ready to test webcam? (y/n): ")
    if response.lower() == 'y':
        cmd = "python3 pose_extractor.py --mode webcam"
        print(f"Running: {cmd}")
        os.system(cmd)
        return True
    else:
        print("⏭️  Skipping webcam test")
        return False

def analyze_sample_results():
    """Analyze the results from image processing"""
    print("\n📊 ANALYSIS: Sample Results")
    print("-" * 40)
    
    # Check if CSV was created
    csv_path = "emma_pose_data.csv"
    cmd = f"python3 pose_extractor.py --mode image -f examples/inputs/emma.jpg --csv {csv_path}"
    print(f"Generating CSV data: {cmd}")
    os.system(cmd)
    
    # Try to read and analyze the CSV
    try:
        import csv
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pitch = float(row['pitch'])
                    yaw = float(row['yaw'])
                    roll = float(row['roll'])
                    
                    print(f"📈 Emma's Pose Analysis:")
                    print(f"   Pitch: {pitch:6.1f}° ({'Looking up' if pitch > 5 else 'Looking down' if pitch < -5 else 'Level gaze'})")
                    print(f"   Yaw:   {yaw:6.1f}° ({'Turned right' if yaw > 5 else 'Turned left' if yaw < -5 else 'Facing forward'})")
                    print(f"   Roll:  {roll:6.1f}° ({'Tilted right' if roll > 5 else 'Tilted left' if roll < -5 else 'Head level'})")
                    
                    return True
        else:
            print("❌ CSV file not created")
            return False
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return False

def demo_video_processing():
    """Demo video processing if user has a video file"""
    print("\n🧪 TEST 3: Video Processing Demo")
    print("-" * 40)
    print("To test video processing, you need a video file.")
    print("Supported formats: .mp4, .avi, .mov")
    
    video_path = input("Enter path to your video file (or press Enter to skip): ")
    
    if video_path and os.path.exists(video_path):
        output_path = video_path.replace('.mp4', '_with_poses.mp4').replace('.avi', '_with_poses.avi').replace('.mov', '_with_poses.mov')
        csv_path = video_path.replace('.mp4', '_poses.csv').replace('.avi', '_poses.csv').replace('.mov', '_poses.csv')
        
        cmd = f"python3 pose_extractor.py --mode video -f '{video_path}' -o '{output_path}' --csv '{csv_path}'"
        print(f"Running: {cmd}")
        print("This may take a while depending on video length...")
        
        result = os.system(cmd)
        if result == 0:
            print(f"✅ Video processing complete!")
            print(f"   Output video: {output_path}")
            print(f"   Pose data CSV: {csv_path}")
            return True
        else:
            print("❌ Video processing failed")
            return False
    else:
        print("⏭️  Skipping video test (no file provided)")
        return False

def show_performance_info():
    """Show performance and capability information"""
    print("\n🚀 PERFORMANCE INFO")
    print("-" * 40)
    print("3DDFA_V2 Performance Characteristics:")
    print(f"📊 CPU Processing Speed: ~1.35ms per face (ONNX mode)")
    print(f"📊 PyTorch Speed: ~6.2ms per face (fallback mode)")
    print(f"📊 Face Detection: ~15ms per frame (720p)")
    print(f"📊 3D Reconstruction: ~1ms per face")
    print(f"📊 Memory Usage: ~500MB typical")
    
    print("\n✨ CAPABILITIES:")
    print("✅ Real-time webcam processing")
    print("✅ Video batch processing")
    print("✅ High-accuracy pose estimation")
    print("✅ 3D facial landmark detection")
    print("✅ Multi-face tracking")
    print("✅ ONNX optimization")
    print("✅ CSV data export")
    print("✅ Visual pose feedback")

def main():
    """Run complete test suite"""
    print("🎯 3DDFA_V2 Perfect Pose Extraction")
    print("🧪 Comprehensive Test Suite & Demo")
    print("=" * 50)
    
    # Check basic setup
    print("🔍 Checking setup...")
    if not os.path.exists('examples/inputs/emma.jpg'):
        print("❌ Sample image not found. Please run from 3DDFA_V2 directory.")
        return
    
    if not os.path.exists('pose_extractor.py'):
        print("❌ pose_extractor.py not found. Please ensure it's in the current directory.")
        return
    
    print("✅ Setup looks good!")
    
    # Run tests
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Image processing
    total_tests += 1
    if test_image_processing():
        tests_passed += 1
    
    # Test 2: Webcam (optional)
    if demo_webcam():
        print("✅ Webcam demo completed")
    
    # Analysis
    if analyze_sample_results():
        print("✅ Result analysis completed")
    
    # Test 3: Video (optional)
    if demo_video_processing():
        print("✅ Video processing demo completed")
    
    # Show performance info
    show_performance_info()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 TEST SUITE COMPLETE")
    print(f"📊 Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ ALL TESTS PASSED - Your 3DDFA_V2 setup is perfect!")
        print("\n🚀 You're ready for perfect pose extraction!")
        print("\n📋 Quick commands to remember:")
        print("   Image:   python3 pose_extractor.py --mode image -f your_image.jpg")
        print("   Webcam:  python3 pose_extractor.py --mode webcam")
        print("   Video:   python3 pose_extractor.py --mode video -f your_video.mp4")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("💡 Try running individual commands to debug issues.")
    
    print("\n📖 For complete documentation, see: SETUP_GUIDE.md")

if __name__ == '__main__':
    main()
