# -*- coding: utf-8 -*-
"""
Unified Data Manager - 統一資料管理系統
讓所有平台和腳本共用同一份資料，避免重複讀取
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
from typing import Optional, Dict, Any
import threading

warnings.filterwarnings('ignore')

class UnifiedDataManager:
    """統一資料管理器 - 單例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._data_cache = {}
        self._data_info = {}
        self._initialized = True
        
        # 預設資料檔案路徑
        self.default_data_paths = [
            "data/TXF1_Minute_2020-01-01_2025-07-04.txt",  # 主要資料目錄
            "multi_strategy_system/TXF1_Minute_2020-01-01_2025-07-04.txt",  # 系統目錄
            "TXF1_Minute_2020-01-01_2025-07-04.txt",  # 當前目錄
            "data/TXF1_Minute_2020-01-01_2025-06-16.txt",  # 舊版本備用
            "TXF1_Minute_2020-01-01_2025-06-16.txt"  # 舊版本當前目錄
        ]
        
        print("🚀 統一資料管理器已初始化")
    
    def find_data_file(self, filename: Optional[str] = None) -> str:
        """尋找資料檔案"""
        if filename:
            # 檢查指定檔案是否存在
            for base_path in ["", "data/", "multi_strategy_system/"]:
                full_path = os.path.join(base_path, filename)
                if os.path.exists(full_path):
                    return full_path
            raise FileNotFoundError(f"找不到指定的資料檔案: {filename}")
        
        # 自動尋找最佳資料檔案
        for path in self.default_data_paths:
            if os.path.exists(path):
                print(f"📁 找到資料檔案: {path}")
                return path
        
        # 如果都找不到，嘗試搜尋整個目錄
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.startswith("TXF1_Minute") and file.endswith(".txt"):
                    full_path = os.path.join(root, file)
                    print(f"🔍 搜尋到資料檔案: {full_path}")
                    return full_path
        
        raise FileNotFoundError("找不到任何 TXF1_Minute 資料檔案")
    
    def load_data(self, 
                  filename: Optional[str] = None,
                  use_cache: bool = True,
                  force_reload: bool = False,
                  **kwargs) -> pd.DataFrame:
        """
        載入資料，支援快取機制
        
        Args:
            filename: 資料檔案名稱
            use_cache: 是否使用快取
            force_reload: 強制重新載入
            **kwargs: 傳遞給 pd.read_csv 的參數
            
        Returns:
            pd.DataFrame: 載入的資料
        """
        # 尋找檔案
        file_path = self.find_data_file(filename)
        
        # 檢查快取
        cache_key = f"{file_path}_{hash(str(kwargs))}"
        
        if use_cache and not force_reload:
            with self._lock:
                if cache_key in self._data_cache:
                    print(f"📋 使用快取資料: {file_path}")
                    return self._data_cache[cache_key].copy()
        
        # 載入資料
        print(f"📥 載入資料檔案: {file_path}")
        start_time = datetime.now()
        
        try:
            # 預設參數
            default_params = {
                'sep': '\t',
                'parse_dates': ['datetime'],
                'index_col': 'datetime',
                'engine': 'python'
            }
            
            # 合併參數
            load_params = {**default_params, **kwargs}
            
            # 載入資料
            df = pd.read_csv(file_path, **load_params)
            
            # 資料清理
            df = self._clean_data(df)
            
            # 儲存到快取
            if use_cache:
                with self._lock:
                    self._data_cache[cache_key] = df.copy()
                    self._data_info[cache_key] = {
                        'file_path': file_path,
                        'load_time': datetime.now(),
                        'rows': len(df),
                        'columns': list(df.columns),
                        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                    }
            
            load_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ 資料載入完成: {len(df):,} 筆記錄, 耗時 {load_time:.2f} 秒")
            
            return df
            
        except Exception as e:
            print(f"❌ 資料載入失敗: {str(e)}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理資料"""
        # 移除重複資料
        original_len = len(df)
        df = df.drop_duplicates()
        if len(df) < original_len:
            print(f"🧹 移除 {original_len - len(df)} 筆重複資料")
        
        # 確保索引排序
        df = df.sort_index()
        
        # 移除無效資料
        df = df.dropna()
        
        # 確保數值欄位為數值型別
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除異常值
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # 移除價格為 0 或負數的資料
                df = df[df[col] > 0]
        
        return df
    
    def get_cache_status(self) -> Dict[str, Any]:
        """取得快取狀態"""
        with self._lock:
            total_memory = sum(info.get('memory_usage_mb', 0) for info in self._data_info.values())
            return {
                'cache_count': len(self._data_cache),
                'total_memory_mb': total_memory,
                'cached_files': list(self._data_info.keys())
            }
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """清除快取"""
        with self._lock:
            if cache_key:
                if cache_key in self._data_cache:
                    del self._data_cache[cache_key]
                    del self._data_info[cache_key]
                    print(f"🗑️ 清除快取: {cache_key}")
            else:
                self._data_cache.clear()
                self._data_info.clear()
                print("🗑️ 清除所有快取")

# 全域資料管理器實例
data_manager = UnifiedDataManager()

# 便利函數
def load_txf_data(filename: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    便利函數：載入 TXF 資料
    
    Args:
        filename: 檔案名稱
        **kwargs: 其他參數
        
    Returns:
        pd.DataFrame: 載入的資料
    """
    return data_manager.load_data(filename, **kwargs)

def get_data_info() -> Dict[str, Any]:
    """
    便利函數：取得資料資訊
    
    Returns:
        Dict: 資料資訊
    """
    return data_manager.get_cache_status()

def clear_data_cache():
    """清除資料快取"""
    data_manager.clear_cache()

if __name__ == "__main__":
    # 測試統一資料管理器
    print("🧪 測試統一資料管理器...")
    
    try:
        # 載入資料
        df = load_txf_data()
        print(f"📊 載入成功: {len(df)} 筆記錄")
        print(f"📅 資料期間: {df.index.min()} 到 {df.index.max()}")
        print(f"📈 欄位: {list(df.columns)}")
        
        # 再次載入（應該使用快取）
        df2 = load_txf_data()
        print(f"📋 第二次載入（快取）: {len(df2)} 筆記錄")
        
        # 顯示快取狀態
        cache_status = get_data_info()
        print(f"💾 快取狀態: {cache_status}")
        
    except Exception as e:
        print(f"❌ 測試失敗: {str(e)}") 