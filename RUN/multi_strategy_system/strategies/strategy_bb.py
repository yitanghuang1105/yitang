"""
Bollinger Bands Strategy Module
Computes 0-100 long position score based on price position within Bollinger Bands
"""

import pandas as pd
import numpy as np
import talib

def compute_score(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Compute Bollinger Bands-based long position score (0-100)
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary containing Bollinger Bands parameters
            - bb_window: Moving average window (default: 20)
            - bb_std: Standard deviation multiplier (default: 2.0)
    
    Returns:
        pd.Series: 0-100 score where higher values indicate stronger long signals
    """
    # Extract parameters with defaults
    bb_window = params.get('bb_window', 20)
    bb_std = params.get('bb_std', 2.0)
    
    # Calculate Bollinger Bands
    if 'close' in df.columns:
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df['close'].values, 
            timeperiod=bb_window, 
            nbdevup=bb_std, 
            nbdevdn=bb_std, 
            matype=0
        )
    else:
        # Fallback if close column not found
        return pd.Series(50, index=df.index)
    
    # Calculate price position within bands (0 = lower band, 1 = upper band)
    band_width = bb_upper - bb_lower
    band_width = np.where(band_width == 0, 1, band_width)  # Avoid division by zero
    
    deviation = (df['close'] - bb_lower) / band_width
    deviation = deviation.clip(0, 1)  # Clip to 0-1 range
    
    # Convert to score: closer to lower band = higher score (stronger long signal)
    # Closer to upper band = lower score (weaker long signal)
    score = (1 - deviation) * 100
    
    # Fill NaN values with neutral score (50)
    score = pd.Series(score, index=df.index).fillna(50)
    
    return score 