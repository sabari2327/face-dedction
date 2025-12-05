#!/usr/bin/env python
"""
Quick launcher - Downloads and runs the face recognition app
Users can use this to easily install and run the project
"""

import subprocess
import sys
import os

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  FACE RECOGNITION - Quick Setup & Run                      ║
    ╚════════════════════════════════════════════════════════════╝
    
    This script will:
    1. Check for required packages
    2. Install missing packages
    3. Launch the face matcher GUI
    
    """)
    
    # Check if packages are installed
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'imutils': 'imutils'
    }
    
    print("Checking dependencies...\n")
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("\n✓ Packages installed successfully\n")
    
    print("\nLaunching Face Recognition Matcher...\n")
    print("─" * 60)
    
    # Run the face matcher
    subprocess.call([sys.executable, 'match_faces_gui.py'])

if __name__ == '__main__':
    main()
