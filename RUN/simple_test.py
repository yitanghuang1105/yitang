#!/usr/bin/env python3
"""
Simple Platform Test
簡單平台測試
"""

import sys
import os

def test_basic_functionality():
    """Test basic platform functionality"""
    print("🧪 開始簡單平台測試")
    print("=" * 50)
    
    # Test 1: Check if we can import basic modules
    print("1. 測試模組導入...")
    try:
        import tkinter as tk
        import pandas as pd
        import numpy as np
        print("   ✅ 基本模組導入成功")
    except Exception as e:
        print(f"   ❌ 模組導入失敗: {e}")
        return False
    
    # Test 2: Check file structure
    print("2. 測試文件結構...")
    required_files = [
        "platforms/multi_strategy_parameter_platform.py",
        "launchers/start_platform.py",
        "quick_start.ps1"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file} 存在")
        else:
            print(f"   ❌ {file} 不存在")
            return False
    
    # Test 3: Test strategy system
    print("3. 測試策略系統...")
    try:
        sys.path.append('multi_strategy_system')
        from multi_strategy_system.strategy_combiner import get_default_params
        params = get_default_params()
        print(f"   ✅ 策略參數載入成功 ({len(params)} 個參數)")
    except Exception as e:
        print(f"   ❌ 策略系統測試失敗: {e}")
        return False
    
    # Test 4: Test data generation
    print("4. 測試數據生成...")
    try:
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1min')
        df = pd.DataFrame({
            'close': [100 + i for i in range(10)],
            'volume': [1000 + i*100 for i in range(10)]
        }, index=dates)
        print(f"   ✅ 測試數據生成成功 ({len(df)} 行)")
    except Exception as e:
        print(f"   ❌ 數據生成失敗: {e}")
        return False
    
    print("\n🎉 所有基本測試通過!")
    return True

def show_platform_info():
    """Show platform information"""
    print("\n📊 平台資訊:")
    print("-" * 30)
    print("• 多策略交易參數優化平台")
    print("• 支援 RSI、布林通道、OBV 三種策略")
    print("• 提供 GUI 界面進行參數調整")
    print("• 包含完整的分析與匯出功能")
    print("• 時間間隔選擇框已優化")

def show_usage_guide():
    """Show usage guide"""
    print("\n🚀 使用指南:")
    print("-" * 30)
    print("1. 快速啟動: .\\quick_start.ps1")
    print("2. 主平台: python platforms\\multi_strategy_parameter_platform.py")
    print("3. 整合平台: python platforms\\integrated_strategy_platform.py")
    print("4. 平台選擇器: python launchers\\start_platform.py")

def main():
    """Main function"""
    print("簡單平台測試工具")
    print("=" * 50)
    
    # Run basic tests
    success = test_basic_functionality()
    
    if success:
        show_platform_info()
        show_usage_guide()
        
        print("\n💡 平台已準備就緒!")
        print("   您可以開始使用平台進行交易策略分析。")
    else:
        print("\n⚠️ 測試失敗，請檢查平台安裝。")
    
    return success

if __name__ == "__main__":
    main() 