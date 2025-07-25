"""
OBV (On-Balance Volume) Strategy Module
Computes 0-100 long position score based on OBV momentum
"""

import pandas as pd
import numpy as np
import talib

def compute_score(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Compute OBV-based long position score (0-100)
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary containing OBV parameters
            - obv_window: OBV moving average window (default: 10)
            - obv_threshold: Volume threshold multiplier (default: 1.2)
    
    Returns:
        pd.Series: 0-100 score where higher values indicate stronger long signals
    """
    # Extract parameters with defaults
    obv_window = params.get('obv_window', 10)
    obv_threshold = params.get('obv_threshold', 1.2)
    
    # Calculate OBV
    if 'close' in df.columns and 'volume' in df.columns:
        # Convert to double precision for talib
        close_values = df['close'].values.astype(np.float64)
        volume_values = df['volume'].values.astype(np.float64)
        obv = talib.OBV(close_values, volume_values)
    else:
        # Fallback if required columns not found
        return pd.Series(50, index=df.index)
    
    # Calculate OBV moving average
    obv_ma = pd.Series(obv).rolling(window=obv_window).mean()
    
    # Calculate OBV momentum (current OBV vs moving average)
    obv_diff = pd.Series(obv) - obv_ma
    
    # Calculate volume ratio (current volume vs average volume)
    volume_ma = df['volume'].rolling(window=obv_window).mean()
    volume_ratio = df['volume'] / volume_ma
    
    # Combine OBV momentum and volume ratio for score
    # Positive OBV momentum + high volume = high score
    # Negative OBV momentum + low volume = low score
    
    # Normalize OBV difference to 0-1 range
    obv_max = obv_diff.abs().max()
    if obv_max > 0:
        obv_normalized = (obv_diff / obv_max + 1) / 2  # Convert to 0-1 range
    else:
        obv_normalized = pd.Series(0.5, index=df.index)
    
    # Normalize volume ratio to 0-1 range
    volume_normalized = volume_ratio.clip(0, obv_threshold) / obv_threshold
    
    # Combine signals: 70% weight to OBV momentum, 30% to volume
    score = (obv_normalized * 0.7 + volume_normalized * 0.3) * 100
    
    # Fill NaN values with neutral score (50)
    score = score.fillna(50)
    
    return score 