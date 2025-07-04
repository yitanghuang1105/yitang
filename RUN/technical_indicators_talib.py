"""
Technical Indicators Module using TA-Lib
========================================

This module provides optimized technical indicators using TA-Lib library.
It replaces the custom implementations found in various RUN files with
more efficient and standardized calculations.

Features:
- All major technical indicators using TA-Lib
- Consistent API across all indicators
- Better performance and accuracy
- Reduced code complexity
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Technical indicators calculator using TA-Lib"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        self._validate_data()
    
    def _validate_data(self):
        """Validate that required columns exist"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def add_bollinger_bands(self, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands indicators
        
        Args:
            window: Rolling window for moving average
            num_std: Number of standard deviations for bands
            
        Returns:
            DataFrame with BB indicators added
        """
        upper, middle, lower = talib.BBANDS(
            self.df['close'], 
            timeperiod=window, 
            nbdevup=num_std, 
            nbdevdn=num_std
        )
        
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = middle
        self.df['bb_lower'] = lower
        self.df['bb_width'] = (upper - lower) / middle
        self.df['bb_position'] = (self.df['close'] - lower) / (upper - lower)
        
        return self.df
    
    def add_rsi(self, window: int = 14) -> pd.DataFrame:
        """
        Add RSI indicator
        
        Args:
            window: RSI calculation window
            
        Returns:
            DataFrame with RSI indicator added
        """
        self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=window)
        return self.df
    
    def add_obv(self, window: int = 20) -> pd.DataFrame:
        """
        Add OBV (On-Balance Volume) indicator
        
        Args:
            window: Window for OBV moving average
            
        Returns:
            DataFrame with OBV indicators added
        """
        self.df['obv'] = talib.OBV(self.df['close'], self.df['volume'])
        self.df['obv_sma'] = talib.SMA(self.df['obv'], timeperiod=window)
        self.df['obv_ratio'] = self.df['obv'] / self.df['obv_sma']
        
        return self.df
    
    def add_moving_averages(self, short_window: int = 10, long_window: int = 50) -> pd.DataFrame:
        """
        Add moving averages for trend analysis
        
        Args:
            short_window: Short-term MA window
            long_window: Long-term MA window
            
        Returns:
            DataFrame with MA indicators added
        """
        self.df['ma_short'] = talib.SMA(self.df['close'], timeperiod=short_window)
        self.df['ma_long'] = talib.SMA(self.df['close'], timeperiod=long_window)
        self.df['trend_up'] = self.df['ma_short'] > self.df['ma_long']
        self.df['trend_down'] = self.df['ma_short'] < self.df['ma_long']
        
        return self.df
    
    def add_atr(self, window: int = 14) -> pd.DataFrame:
        """
        Add ATR (Average True Range) indicator
        
        Args:
            window: ATR calculation window
            
        Returns:
            DataFrame with ATR indicators added
        """
        self.df['atr'] = talib.ATR(
            self.df['high'], 
            self.df['low'], 
            self.df['close'], 
            timeperiod=window
        )
        self.df['atr_ratio'] = self.df['atr'] / self.df['close']
        
        return self.df
    
    def add_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Add MACD indicator
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with MACD indicators added
        """
        macd, signal, hist = talib.MACD(
            self.df['close'],
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        
        self.df['macd'] = macd
        self.df['macd_signal'] = signal
        self.df['macd_histogram'] = hist
        
        return self.df
    
    def add_stochastic(self, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> pd.DataFrame:
        """
        Add Stochastic oscillator
        
        Args:
            fastk_period: %K period
            slowk_period: %K smoothing period
            slowd_period: %D smoothing period
            
        Returns:
            DataFrame with Stochastic indicators added
        """
        slowk, slowd = talib.STOCH(
            self.df['high'],
            self.df['low'],
            self.df['close'],
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0
        )
        
        self.df['stoch_k'] = slowk
        self.df['stoch_d'] = slowd
        
        return self.df
    
    def add_all_indicators(self, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Add all commonly used technical indicators
        
        Args:
            params: Dictionary of parameters for indicators
            
        Returns:
            DataFrame with all indicators added
        """
        if params is None:
            params = {
                'bb_window': 20,
                'bb_std': 2.0,
                'rsi_window': 14,
                'obv_window': 20,
                'ma_short': 10,
                'ma_long': 50,
                'atr_window': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'stoch_k': 14,
                'stoch_d': 3
            }
        
        # Add all indicators
        self.add_bollinger_bands(params['bb_window'], params['bb_std'])
        self.add_rsi(params['rsi_window'])
        self.add_obv(params['obv_window'])
        self.add_moving_averages(params['ma_short'], params['ma_long'])
        self.add_atr(params['atr_window'])
        self.add_macd(params['macd_fast'], params['macd_slow'], params['macd_signal'])
        self.add_stochastic(params['stoch_k'], params['stoch_d'], params['stoch_d'])
        
        return self.df
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the DataFrame with all indicators"""
        return self.df.copy()


def calculate_technical_indicators_talib(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to calculate all technical indicators using TA-Lib
    
    Args:
        df: OHLCV DataFrame
        params: Parameters for indicators
        
    Returns:
        DataFrame with all indicators added
    """
    ti = TechnicalIndicators(df)
    return ti.add_all_indicators(params)


# Backward compatibility functions for existing code
def calculate_bollinger_bands_talib(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands using TA-Lib"""
    ti = TechnicalIndicators(df)
    return ti.add_bollinger_bands(window, num_std)


def calculate_rsi_talib(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate RSI using TA-Lib"""
    ti = TechnicalIndicators(df)
    return ti.add_rsi(window)


def calculate_obv_talib(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate OBV using TA-Lib"""
    ti = TechnicalIndicators(df)
    return ti.add_obv(window)


def calculate_moving_averages_talib(df: pd.DataFrame, short_window: int = 10, long_window: int = 50) -> pd.DataFrame:
    """Calculate moving averages using TA-Lib"""
    ti = TechnicalIndicators(df)
    return ti.add_moving_averages(short_window, long_window)


def calculate_atr_talib(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate ATR using TA-Lib"""
    ti = TechnicalIndicators(df)
    return ti.add_atr(window)


# Example usage and testing
if __name__ == "__main__":
    # Test the module
    print("Testing Technical Indicators Module with TA-Lib...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Test the module
    ti = TechnicalIndicators(sample_data)
    result = ti.add_all_indicators()
    
    print("✅ Technical indicators calculated successfully!")
    print(f"📊 DataFrame shape: {result.shape}")
    print(f"📈 Indicators added: {[col for col in result.columns if col not in ['open', 'high', 'low', 'close', 'volume']]}")
    
    # Show sample results
    print("\n📋 Sample results:")
    print(result[['close', 'rsi', 'bb_upper', 'bb_lower', 'ma_short', 'ma_long']].tail()) 