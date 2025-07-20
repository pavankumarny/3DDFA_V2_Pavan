#!/usr/bin/env python3
"""
Simple wrapper to run pose extraction from any directory
This handles the directory change automatically
"""

import os
import sys
import subprocess

def main():
    # Find the 3DDFA_V2 root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Change to root directory
    original_cwd = os.getcwd()
    os.chdir(root_dir)
    
    try:
        # Run the pose extractor with all arguments passed through
        cmd = ["python3", "tools/pose_extractor.py"] + sys.argv[1:]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    finally:
        # Return to original directory
        os.chdir(original_cwd)

if __name__ == '__main__':
    main()
