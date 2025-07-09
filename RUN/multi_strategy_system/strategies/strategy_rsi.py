"""
RSI Strategy Module
Computes 0-100 long position score based on RSI values
"""

import pandas as pd
import numpy as np
import talib

def compute_score(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Compute RSI-based long position score (0-100)
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary containing RSI parameters
            - rsi_window: RSI calculation window (default: 14)
            - rsi_oversold: Oversold threshold (default: 30)
            - rsi_overbought: Overbought threshold (default: 70)
    
    Returns:
        pd.Series: 0-100 score where higher values indicate stronger long signals
    """
    # Extract parameters with defaults
    rsi_window = params.get('rsi_window', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    
    # Calculate RSI
    if 'close' in df.columns:
        rsi = talib.RSI(df['close'].values, timeperiod=rsi_window)
        print(f"DEBUG RSI: df.index length: {len(df.index)}, rsi length: {len(rsi)}")
    else:
        # Fallback if close column not found
        return pd.Series(50, index=df.index)
    
    # Convert RSI to 0-100 score
    # RSI < 30 (oversold) -> high score (80-100)
    # RSI > 70 (overbought) -> low score (0-20)
    # RSI between 30-70 -> linear interpolation
    
    # 確保 rsi 的長度與 df.index 一致
    if len(rsi) != len(df.index):
        print(f"DEBUG RSI: Length mismatch! df.index: {len(df.index)}, rsi: {len(rsi)}")
        # 如果長度不匹配，調整 rsi 的長度
        if len(rsi) < len(df.index):
            # rsi 較短，在前面補 NaN
            rsi = np.concatenate([np.full(len(df.index) - len(rsi), np.nan), rsi])
        else:
            # rsi 較長，截取後面的部分
            rsi = rsi[-len(df.index):]
    
    score = pd.Series(index=df.index, dtype=float)
    
    # Oversold region: RSI < 30 -> score 80-100
    oversold_mask = rsi < rsi_oversold
    score[oversold_mask] = 100 - ((rsi[oversold_mask] - 20) * 1.0)  # RSI 20->100, 30->90
    
    # Overbought region: RSI > 70 -> score 0-20
    overbought_mask = rsi > rsi_overbought
    score[overbought_mask] = 20 - ((rsi[overbought_mask] - 70) * 1.0)  # RSI 70->20, 80->10
    
    # Neutral region: RSI 30-70 -> score 20-80
    neutral_mask = (rsi >= rsi_oversold) & (rsi <= rsi_overbought)
    score[neutral_mask] = 80 - ((rsi[neutral_mask] - rsi_oversold) * 1.5)  # Linear interpolation
    
    # Clip to 0-100 range
    score = score.clip(0, 100)
    
    # Fill NaN values with neutral score (50)
    score = score.fillna(50)
    
    return score 