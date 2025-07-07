#!/usr/bin/env python3
"""
Simple Platform Launcher
快速啟動簡化版參數調整平台
"""

import os
import sys
import subprocess

def main():
    print("=" * 50)
    print("Trading Strategy Parameter Platform")
    print("=" * 50)
    print("Starting Simple Parameter Platform...")
    print()
    
    try:
        # 檢查Python檔案是否存在
        platform_file = "simple_parameter_platform.py"
        if not os.path.exists(platform_file):
            print(f"Error: {platform_file} not found!")
            return
        
        # 啟動平台
        print("Launching platform...")
        subprocess.run([sys.executable, platform_file])
        
    except KeyboardInterrupt:
        print("\nPlatform stopped by user.")
    except Exception as e:
        print(f"Error starting platform: {e}")

if __name__ == "__main__":
    main() 