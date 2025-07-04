"""
MultiCharts 檔案讀取器
監控 MultiCharts 匯出的資料檔案，每分鐘更新
"""
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MultiChartsReader:
    def __init__(self, data_file_path="TXF1_Minute_2020-01-01_2025-06-16.txt"):
        self.data_file_path = data_file_path
        self.last_modified = None
        self.cached_data = None
        
    def get_latest_data(self):
        """獲取最新的台指期資料"""
        try:
            # 檢查檔案是否存在
            if not os.path.exists(self.data_file_path):
                logger.warning(f"Data file not found: {self.data_file_path}")
                return None
            
            # 檢查檔案是否被修改
            current_modified = os.path.getmtime(self.data_file_path)
            
            if self.last_modified != current_modified:
                logger.info(f"Reading updated data file: {self.data_file_path}")
                
                # 讀取資料
                df = pd.read_csv(self.data_file_path)
                
                # 處理日期時間
                if 'Date' in df.columns and 'Time' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
                    df = df.sort_values('DateTime')
                
                # 計算基本指標
                df = self._calculate_basic_indicators(df)
                
                self.cached_data = df
                self.last_modified = current_modified
                
                return df
            else:
                return self.cached_data
                
        except Exception as e:
            logger.error(f"Error reading data file: {e}")
            return None
    
    def _calculate_basic_indicators(self, df):
        """計算基本技術指標"""
        if len(df) < 20:
            return df
        
        # 計算移動平均
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 計算RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 計算價格變化
        df['PriceChange'] = df['Close'].diff()
        df['PriceChangePct'] = df['Close'].pct_change() * 100
        
        # 計算3分鐘內價格變化
        df['PriceChange3Min'] = df['Close'] - df['Close'].shift(3)
        
        return df
    
    def get_current_price(self):
        """獲取當前價格"""
        df = self.get_latest_data()
        if df is not None and len(df) > 0:
            return df.iloc[-1]['Close']
        return None
    
    def get_price_change(self, minutes=3):
        """獲取指定分鐘數內的價格變化"""
        df = self.get_latest_data()
        if df is not None and len(df) >= minutes:
            current_price = df.iloc[-1]['Close']
            past_price = df.iloc[-minutes]['Close']
            return current_price - past_price
        return None 