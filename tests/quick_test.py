#!/usr/bin/env python3
"""
Quick test runner for pose extraction - runs from main 3DDFA_V2 directory
"""

import os
import sys
import subprocess

def test_pose_extraction():
    """Test pose extraction from the main directory"""
    print("🧪 Quick Pose Extraction Test")
    print("=" * 40)
    
    # Ensure we're in the right directory
    if not os.path.exists("examples/inputs/emma.jpg"):
        print("❌ Please run this from the main 3DDFA_V2 directory")
        return False
    
    # Test the tools/pose_extractor.py
    cmd = [
        "python3", "tools/pose_extractor.py",
        "--mode", "image",
        "-f", "examples/inputs/emma.jpg",
        "-o", "results/emma_test.jpg"
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ SUCCESS! Pose extraction working!")
            print("\n📋 Output:")
            print(result.stdout)
            
            # Check if output file was created
            if os.path.exists("results/emma_test.jpg"):
                print("✅ Output image created: results/emma_test.jpg")
            else:
                print("⚠️  No output image found")
                
            return True
        else:
            print("❌ FAILED!")
            print("\n📋 Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT! Process took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_lit_video():
    """Test on Lit.mp4 video"""
    print("\n🎬 Testing Lit.mp4 Video Processing")
    print("=" * 40)
    
    if not os.path.exists("examples/inputs/videos/Lit.mp4"):
        print("❌ Lit.mp4 not found")
        return False
    
    cmd = [
        "python3", "tools/pose_extractor.py",
        "--mode", "video",
        "-f", "examples/inputs/videos/Lit.mp4",
        "--landmarks",
        "-o", "results/Lit_test.mp4",
        "--csv", "results/Lit_test.csv"
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    print("⏳ This may take a few minutes...")
    
    try:
        result = subprocess.run(cmd, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ SUCCESS! Video processing complete!")
            
            # Check outputs
            if os.path.exists("results/Lit_test.mp4"):
                print("✅ Output video: results/Lit_test.mp4")
            if os.path.exists("results/Lit_test.csv"):
                print("✅ CSV data: results/Lit_test.csv")
                
            return True
        else:
            print("❌ Video processing failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT! Video processing took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == '__main__':
    print("🎯 3DDFA_V2 Pose Extraction Test Suite")
    print("=" * 50)
    
    # Test 1: Image processing
    image_success = test_pose_extraction()
    
    if image_success:
        print("\n💡 Ready for video processing!")
        
        # Ask user if they want to test video
        response = input("\nTest Lit.mp4 video processing? (y/n): ").lower()
        if response == 'y':
            video_success = test_lit_video()
            
            if video_success:
                print("\n🎉 ALL TESTS PASSED!")
                print("\n📋 Your commands are working perfectly:")
                print("   Image: python3 tools/pose_extractor.py --mode image -f examples/inputs/emma.jpg")
                print("   Video: python3 tools/pose_extractor.py --mode video -f examples/inputs/videos/Lit.mp4 --landmarks -o results/output.mp4 --csv results/data.csv")
            else:
                print("\n⚠️  Video test failed, but image processing works!")
        else:
            print("\n✅ Image processing confirmed working!")
    else:
        print("\n💡 Check the error messages above and try running from the main 3DDFA_V2 directory")
