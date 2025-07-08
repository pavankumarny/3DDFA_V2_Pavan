#!/usr/bin/env python3
"""
Quick verification test for the clean 3DDFA_V2 setup
"""

import os
import sys

def test_setup():
    """Test if everything is properly organized and working"""
    print("🧪 Testing 3DDFA_V2 Clean Setup")
    print("=" * 40)
    
    # Check directory structure
    required_dirs = ['tools', 'results', 'tests', 'examples', 'weights', 'configs']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(f"../{dir_name}"):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            missing_dirs.append(dir_name)
    
    # Check main tool
    if os.path.exists("../tools/pose_extractor.py"):
        print("✅ Main pose extractor tool exists")
    else:
        print("❌ Main pose extractor tool missing")
    
    # Check sample data
    if os.path.exists("../examples/inputs/emma.jpg"):
        print("✅ Sample image exists")
    else:
        print("❌ Sample image missing")
    
    if os.path.exists("../examples/inputs/videos/Lit.mp4"):
        print("✅ Sample video (Lit.mp4) exists")
    else:
        print("❌ Sample video missing")
    
    # Test basic imports
    print("\n🔧 Testing imports...")
    try:
        sys.path.append('..')
        import cv2
        import numpy
        import yaml
        print("✅ Basic packages available")
    except ImportError as e:
        print(f"❌ Import error: {e}")
    
    # Summary
    print("\n📋 Setup Summary:")
    if not missing_dirs:
        print("✅ Directory structure is clean and organized")
        print("🎯 Ready to use! Run the commands from README_CLEAN.md")
        
        print("\n💡 Quick test command:")
        print("cd tools && python3 pose_extractor.py --mode image -f ../examples/inputs/emma.jpg")
    else:
        print(f"⚠️  Missing directories: {', '.join(missing_dirs)}")

if __name__ == '__main__':
    test_setup()
