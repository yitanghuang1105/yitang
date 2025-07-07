"""
Platform Launcher
Simple launcher for parameter adjustment platforms
"""

import os
import sys
import subprocess

def main():
    print("=" * 50)
    print("Trading Strategy Parameter Platform Launcher")
    print("=" * 50)
    print()
    print("請選擇要啟動的平台：")
    print("1. Simple Parameter Platform (推薦)")
    print("2. Advanced Parameter Platform")
    print("3. Basic Parameter Platform")
    print("4. 退出")
    print()
    
    while True:
        try:
            choice = input("請輸入選項 (1-4): ").strip()
            
            if choice == "1":
                print("啟動簡化版參數調整平台...")
                subprocess.run([sys.executable, "simple_parameter_platform.py"])
                break
            elif choice == "2":
                print("啟動進階版參數調整平台...")
                subprocess.run([sys.executable, "advanced_parameter_platform.py"])
                break
            elif choice == "3":
                print("啟動基礎版參數調整平台...")
                subprocess.run([sys.executable, "parameter_platform.py"])
                break
            elif choice == "4":
                print("退出程式")
                break
            else:
                print("無效選項，請重新輸入")
                
        except KeyboardInterrupt:
            print("\n程式已取消")
            break
        except Exception as e:
            print(f"錯誤: {e}")
            break

if __name__ == "__main__":
    main() 