"""
Entry Signal Module - 進場訊號模組

主要職責：根據技術指標（BB、RSI、OBV）產生進場布林邏輯條件（True / False）

模組架構：
1. 參數結構定義
2. 主函數接口
3. 子函數結構（可獨立抽出）
4. 條件結合邏輯
5. 可擴充性設計
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EntryParams:
    """進場參數結構"""
    # Bollinger Bands parameters
    bb_window: int = 20
    bb_std: float = 2.0
    
    # RSI parameters
    rsi_period: int = 14
    rsi_threshold: float = 30.0
    
    # OBV parameters
    obv_slope_window: int = 10
    obv_positive_only: bool = True
    
    # Entry mode
    entry_mode: str = 'strict'  # 'strict', 'any2', 'loose'
    
    # Time filter parameters
    start_time: str = '09:00'
    end_time: str = '13:30'
    
    # Cooldown parameters
    cooldown_periods: int = 0
    
    # Multi-timeframe parameters
    higher_timeframe_direction: bool = False
    higher_timeframe_period: str = '60min'


def generate_entry_signal(df: pd.DataFrame, params: EntryParams) -> pd.Series:
    """
    根據布林通道、RSI、OBV 等條件產生進場訊號。
    
    Parameters:
        df: 含有 open/high/low/close/volume 等欄位的 K 棒資料
        params: EntryParams 物件，包含所有進場參數
    
    Returns:
        Boolean Series，True 表示進場點
    """
    # 1. 計算技術指標
    bb_condition = bb_entry_condition(df, params.bb_window, params.bb_std)
    rsi_condition = rsi_entry_condition(df, params.rsi_period, params.rsi_threshold)
    obv_condition = obv_entry_condition(df, params.obv_slope_window, params.obv_positive_only)
    
    # 2. 根據 entry_mode 決定條件綜合方式
    entry_signal = combine_conditions(bb_condition, rsi_condition, obv_condition, params.entry_mode)
    
    # 3. 套用時間過濾器
    if params.start_time and params.end_time:
        entry_signal = apply_time_filter(entry_signal, df, params.start_time, params.end_time)
    
    # 4. 套用冷卻期過濾器
    if params.cooldown_periods > 0:
        entry_signal = apply_cooldown_filter(entry_signal, params.cooldown_periods)
    
    # 5. 套用多時間框架過濾器
    if params.higher_timeframe_direction:
        entry_signal = apply_multi_timeframe_filter(entry_signal, df, params.higher_timeframe_period)
    
    return entry_signal


def bb_entry_condition(df: pd.DataFrame, window: int, std: float) -> pd.Series:
    """
    布林通道進場條件：當收盤價 < 下軌時回傳 True
    
    Parameters:
        df: K棒資料
        window: 布林通道週期
        std: 標準差倍數
    
    Returns:
        Boolean Series
    """
    # 計算布林通道
    bb_middle = df['close'].rolling(window=window).mean()
    bb_std = df['close'].rolling(window=window).std()
    bb_lower = bb_middle - (bb_std * std)
    
    # 進場條件：收盤價 < 下軌
    return df['close'] < bb_lower


def rsi_entry_condition(df: pd.DataFrame, period: int, threshold: float) -> pd.Series:
    """
    RSI 進場條件：RSI < 閾值則回傳 True
    
    Parameters:
        df: K棒資料
        period: RSI 週期
        threshold: RSI 閾值
    
    Returns:
        Boolean Series
    """
    # 計算 RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 進場條件：RSI < 閾值
    return rsi < threshold


def obv_entry_condition(df: pd.DataFrame, slope_window: int, positive_only: bool) -> pd.Series:
    """
    OBV 進場條件：OBV 上升則回傳 True
    
    Parameters:
        df: K棒資料
        slope_window: 斜率計算視窗
        positive_only: 是否只考慮正值
    
    Returns:
        Boolean Series
    """
    # 計算 OBV
    obv = calculate_obv(df)
    
    # 計算 OBV 斜率
    obv_slope = obv.diff(slope_window)
    
    # 進場條件：OBV 斜率 > 0
    if positive_only:
        return obv_slope > 0
    else:
        return obv_slope > obv_slope.rolling(window=slope_window).mean()


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """計算 On-Balance Volume (OBV)"""
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def combine_conditions(bb_cond: pd.Series, rsi_cond: pd.Series, obv_cond: pd.Series, entry_mode: str) -> pd.Series:
    """
    結合各項條件的邏輯
    
    Parameters:
        bb_cond: 布林通道條件
        rsi_cond: RSI 條件
        obv_cond: OBV 條件
        entry_mode: 結合模式 ('strict', 'any2', 'loose')
    
    Returns:
        Boolean Series
    """
    if entry_mode == 'strict':
        # 嚴格模式：所有條件都必須滿足
        return bb_cond & rsi_cond & obv_cond
    elif entry_mode == 'any2':
        # 寬鬆模式：至少滿足兩個條件
        condition_sum = bb_cond.astype(int) + rsi_cond.astype(int) + obv_cond.astype(int)
        return condition_sum >= 2
    elif entry_mode == 'loose':
        # 最寬鬆模式：滿足任一條件即可
        return bb_cond | rsi_cond | obv_cond
    else:
        raise ValueError(f"Unknown entry_mode: {entry_mode}")


def apply_time_filter(signal: pd.Series, df: pd.DataFrame, start_time: str, end_time: str) -> pd.Series:
    """
    時間過濾器：只在指定時段內進場
    
    Parameters:
        signal: 原始訊號
        df: K棒資料
        start_time: 開始時間 (HH:MM)
        end_time: 結束時間 (HH:MM)
    
    Returns:
        Boolean Series
    """
    # 假設 df 有 datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        return signal
    
    time_filter = (df.index.time >= pd.to_datetime(start_time).time()) & \
                  (df.index.time <= pd.to_datetime(end_time).time())
    
    return signal & time_filter


def apply_cooldown_filter(signal: pd.Series, cooldown_periods: int) -> pd.Series:
    """
    冷卻期過濾器：避免頻繁進場
    
    Parameters:
        signal: 原始訊號
        cooldown_periods: 冷卻期（K棒數）
    
    Returns:
        Boolean Series
    """
    if cooldown_periods <= 0:
        return signal
    
    filtered_signal = signal.copy()
    last_signal_idx = -1
    
    for i in range(len(signal)):
        if signal.iloc[i]:
            if last_signal_idx == -1 or (i - last_signal_idx) > cooldown_periods:
                last_signal_idx = i
            else:
                filtered_signal.iloc[i] = False
    
    return filtered_signal


def apply_multi_timeframe_filter(signal: pd.Series, df: pd.DataFrame, higher_timeframe_period: str) -> pd.Series:
    """
    多時間框架過濾器：根據更高時間框架的方向過濾
    
    Parameters:
        signal: 原始訊號
        df: K棒資料
        higher_timeframe_period: 更高時間框架週期
    
    Returns:
        Boolean Series
    """
    # 這裡需要根據實際的更高時間框架資料來實作
    # 暫時回傳原始訊號
    return signal


# 可擴充性設計：未來可以加入的模組
class EntrySignalManager:
    """進場訊號管理器 - 用於未來擴充"""
    
    def __init__(self):
        self.indicators = {}
        self.filters = {}
        self.conditions = {}
    
    def add_indicator(self, name: str, indicator_func):
        """新增技術指標"""
        self.indicators[name] = indicator_func
    
    def add_filter(self, name: str, filter_func):
        """新增過濾器"""
        self.filters[name] = filter_func
    
    def add_condition(self, name: str, condition_func):
        """新增條件"""
        self.conditions[name] = condition_func
    
    def generate_signal(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """產生進場訊號（未來擴充用）"""
        # 實作邏輯
        pass


# 使用範例
if __name__ == "__main__":
    # 建立測試資料
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    test_df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # 設定參數
    params = EntryParams(
        bb_window=20,
        bb_std=2.0,
        rsi_period=14,
        rsi_threshold=30.0,
        obv_slope_window=10,
        obv_positive_only=True,
        entry_mode='strict',
        start_time='09:00',
        end_time='13:30',
        cooldown_periods=5
    )
    
    # 產生進場訊號
    entry_signals = generate_entry_signal(test_df, params)
    print(f"Entry signals generated: {entry_signals.sum()} signals found") 