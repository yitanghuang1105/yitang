import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import itertools
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

def load_txf_data(file_path: str) -> pd.DataFrame:
    """
    Load TXF 1-minute data from file and convert to proper format.
    
    Args:
        file_path: Path to the TXF data file
        
    Returns:
        DataFrame with datetime index and OHLCV data
    """
    print(f"Loading data from {file_path}...")
    
    # Read the data file
    df = pd.read_csv(file_path)
    
    # Convert Date and Time to datetime
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Rename columns to lowercase
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'TotalVolume': 'volume'
    })
    
    # Sort by timestamp
    df.sort_index(inplace=True)
    
    # Use only first 1000 records for testing
    df = df.head(1000)
    
    print(f"Data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
    return df

def convert_to_4h_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-minute data to 4-hour (240-minute) data.
    
    Args:
        df: DataFrame with 1-minute OHLCV data
        
    Returns:
        DataFrame with 4-hour OHLCV data
    """
    print("Converting to 4-hour data...")
    
    # Resample to 4-hour intervals
    df_4h = df.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"4-hour data created: {len(df_4h)} records")
    return df_4h

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands indicators.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        DataFrame with BB indicators added
    """
    df = df.copy()
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    bb_std = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * num_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std * num_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate RSI indicator.
    
    Args:
        df: DataFrame with OHLCV data
        window: RSI calculation window
        
    Returns:
        DataFrame with RSI indicator added
    """
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV) indicator.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with OBV indicator added
    """
    df = df.copy()
    df['obv'] = 0.0
    df['obv_sma'] = 0.0
    
    # Calculate OBV
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            df['obv'].iloc[i] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            df['obv'].iloc[i] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
        else:
            df['obv'].iloc[i] = df['obv'].iloc[i-1]
    
    # Calculate OBV moving average
    df['obv_sma'] = df['obv'].rolling(window=20).mean()
    df['obv_ratio'] = df['obv'] / df['obv_sma']
    
    return df

def calculate_trend_filter(df: pd.DataFrame, short_window: int = 10, long_window: int = 50) -> pd.DataFrame:
    """
    Calculate trend filter using moving averages.
    
    Args:
        df: DataFrame with OHLCV data
        short_window: Short-term moving average window
        long_window: Long-term moving average window
        
    Returns:
        DataFrame with trend filter added
    """
    df = df.copy()
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['ma_long'] = df['close'].rolling(window=long_window).mean()
    df['trend_up'] = df['ma_short'] > df['ma_long']
    df['trend_down'] = df['ma_short'] < df['ma_long']
    return df

def calculate_volatility_filter(df: pd.DataFrame, atr_window: int = 14, min_atr_multiplier: float = 0.5) -> pd.DataFrame:
    """
    Calculate volatility filter using ATR.
    
    Args:
        df: DataFrame with OHLCV data
        atr_window: ATR calculation window
        min_atr_multiplier: Minimum ATR multiplier for volatility filter
        
    Returns:
        DataFrame with volatility filter added
    """
    df = df.copy()
    
    # Calculate True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR
    df['atr'] = df['true_range'].rolling(window=atr_window).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    # Volatility filter: only trade when volatility is above threshold
    df['volatility_ok'] = df['atr_ratio'] > (df['atr_ratio'].rolling(window=atr_window*2).mean() * min_atr_multiplier)
    
    # Clean up temporary columns
    df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)
    
    return df

def calculate_volume_filter(df: pd.DataFrame, volume_window: int = 20, volume_threshold: float = 1.2) -> pd.DataFrame:
    """
    Calculate volume filter to confirm price movements.
    
    Args:
        df: DataFrame with OHLCV data
        volume_window: Volume moving average window
        volume_threshold: Volume threshold multiplier
        
    Returns:
        DataFrame with volume filter added
    """
    df = df.copy()
    df['volume_ma'] = df['volume'].rolling(window=volume_window).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['volume_ok'] = df['volume_ratio'] > volume_threshold
    return df

def apply_signal_filters(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Apply all signal filters to the data.
    
    Args:
        df: DataFrame with technical indicators
        params: Strategy parameters including filter parameters
        
    Returns:
        DataFrame with filters applied
    """
    df = df.copy()
    
    # Apply trend filter
    df = calculate_trend_filter(df, 
                               short_window=params.get('trend_short_ma', 10),
                               long_window=params.get('trend_long_ma', 50))
    
    # Apply volatility filter
    df = calculate_volatility_filter(df,
                                   atr_window=params.get('atr_window', 14),
                                   min_atr_multiplier=params.get('atr_multiplier', 0.5))
    
    # Apply volume filter
    df = calculate_volume_filter(df,
                               volume_window=params.get('volume_window', 20),
                               volume_threshold=params.get('volume_threshold', 1.2))
    
    return df

def generate_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Generate trading signals with filters applied.
    
    Args:
        df: DataFrame with technical indicators and filters
        params: Strategy parameters
        
    Returns:
        DataFrame with signals added
    """
    df = df.copy()
    df['signal'] = 0
    
    # Buy signals with filters
    buy_conditions = (
        (df['close'] <= df['bb_lower']) &  # Price at or below lower Bollinger Band
        (df['rsi'] <= params['rsi_oversold']) &  # RSI oversold
        (df['obv_ratio'] >= params['obv_threshold']) &  # Strong volume
        (df['trend_up']) &  # Trend filter
        (df['volatility_ok']) &  # Volatility filter
        (df['volume_ok'])  # Volume filter
    )
    
    # Sell signals with filters
    sell_conditions = (
        (df['close'] >= df['bb_upper']) &  # Price at or above upper Bollinger Band
        (df['rsi'] >= params['rsi_overbought']) &  # RSI overbought
        (df['obv_ratio'] <= 1/params['obv_threshold']) &  # Weak volume
        (df['trend_down']) &  # Trend filter
        (df['volatility_ok']) &  # Volatility filter
        (df['volume_ok'])  # Volume filter
    )
    
    # Set signals
    df.loc[buy_conditions, 'signal'] = 1  # Buy signal
    df.loc[sell_conditions, 'signal'] = -1  # Sell signal
    
    return df

def calculate_returns(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Calculate strategy returns and performance metrics.
    
    Args:
        df: DataFrame with signals
        params: Strategy parameters
        
    Returns:
        Dictionary with performance metrics
    """
    df = df.copy()
    
    # Initialize position and returns columns
    df['position'] = 0
    df['returns'] = 0.0
    df['cumulative_returns'] = 0.0
    
    position = 0
    entry_price = 0
    entry_time = None
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]['close']
        current_signal = df.iloc[i]['signal']
        
        # If we have a position, check for exit conditions
        if position != 0:
            # Calculate current return
            if position == 1:  # Long position
                current_return = (current_price - entry_price) / entry_price
            else:  # Short position
                current_return = (entry_price - current_price) / entry_price
            
            # Check stop loss
            if current_return <= -params['stop_loss_pct']:
                # Stop loss hit
                df.iloc[i, df.columns.get_loc('position')] = 0
                df.iloc[i, df.columns.get_loc('returns')] = -params['stop_loss_pct']
                position = 0
                entry_price = 0
                entry_time = None
                continue
            
            # Check take profit
            if current_return >= params['take_profit_pct']:
                # Take profit hit
                df.iloc[i, df.columns.get_loc('position')] = 0
                df.iloc[i, df.columns.get_loc('returns')] = params['take_profit_pct']
                position = 0
                entry_price = 0
                entry_time = None
                continue
        
        # Check for new signals
        if current_signal == 1 and position == 0:  # Buy signal
            position = 1
            entry_price = current_price
            entry_time = df.index[i]
            df.iloc[i, df.columns.get_loc('position')] = 1
        elif current_signal == -1 and position == 0:  # Sell signal
            position = -1
            entry_price = current_price
            entry_time = df.index[i]
            df.iloc[i, df.columns.get_loc('position')] = -1
        elif current_signal != 0 and position != 0:  # Exit signal
            # Close position
            if position == 1:  # Long position
                current_return = (current_price - entry_price) / entry_price
            else:  # Short position
                current_return = (entry_price - current_price) / entry_price
            
            df.iloc[i, df.columns.get_loc('position')] = 0
            df.iloc[i, df.columns.get_loc('returns')] = current_return
            position = 0
            entry_price = 0
            entry_time = None
    
    # Calculate cumulative returns
    df['cumulative_returns'] = df['returns'].cumsum()
    
    # Calculate performance metrics
    total_return = df['cumulative_returns'].iloc[-1]
    num_trades = len(df[df['returns'] != 0])
    winning_trades = len(df[df['returns'] > 0])
    losing_trades = len(df[df['returns'] < 0])
    
    if num_trades > 0:
        win_rate = winning_trades / num_trades
        avg_win = df[df['returns'] > 0]['returns'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['returns'] < 0]['returns'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    # Calculate maximum drawdown
    cumulative_returns = df['cumulative_returns']
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    metrics = {
        'total_return': total_return,
        'num_trades': num_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown
    }
    
    return df, metrics

def optimize_filters(df: pd.DataFrame, base_params: Dict) -> Tuple[Dict, Dict]:
    """
    Optimize filter parameters.
    
    Args:
        df: DataFrame with technical indicators
        base_params: Base strategy parameters
        
    Returns:
        Tuple of (best_params, best_metrics)
    """
    print("Optimizing filter parameters...")
    
    # Define parameter ranges for optimization
    param_ranges = {
        'trend_short_ma': [5, 10, 15],
        'trend_long_ma': [30, 50, 70],
        'atr_window': [10, 14, 20],
        'atr_multiplier': [0.3, 0.5, 0.7],
        'volume_window': [15, 20, 25],
        'volume_threshold': [1.0, 1.2, 1.5]
    }
    
    best_metrics = None
    best_params = None
    best_sharpe = -999
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    total_combinations = np.prod([len(vals) for vals in param_values])
    print(f"Testing {total_combinations} parameter combinations...")
    
    for i, combination in enumerate(itertools.product(*param_values)):
        if i % 10 == 0:  # Progress update every 10 combinations
            print(f"Progress: {i}/{total_combinations} combinations tested")
        
        params = base_params.copy()
        for name, value in zip(param_names, combination):
            params[name] = value
        
        # Apply filters and generate signals
        df_filters = apply_signal_filters(df, params)
        df_signals = generate_signals(df_filters, params)
        metrics = calculate_returns(df_signals, params)
        
        # Calculate Sharpe ratio (simplified)
        if metrics['num_trades'] > 0:
            sharpe = metrics['total_return'] / (abs(metrics['max_drawdown']) + 0.001)
        else:
            sharpe = -999
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_metrics = metrics
            best_params = params
    
    return best_params, best_metrics

def main():
    """Main function to run signal filtering optimization"""
    print("="*60)
    print("SYSTEM 3: Signal Filtering Optimization (TEST - 1000 records)")
    print("="*60)
    
    # Load and process data
    df = pd.read_csv('TXF1_Minute_2020-01-01_2025-06-16.txt')
    
    # Convert Date and Time to datetime
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Rename columns to lowercase
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'TotalVolume': 'volume'
    })
    
    # Sort by timestamp
    df.sort_index(inplace=True)
    
    # Use only first 1000 records for testing
    df = df.head(1000)
    
    print(f"Data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    df_bb = calculate_bollinger_bands(df, window=20, num_std=2.0)
    df_rsi = calculate_rsi(df_bb, window=14)
    df_obv = calculate_obv(df_rsi)
    
    # Base strategy parameters
    base_params = {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'obv_threshold': 1.2,
        'stop_loss_pct': 0.02,  # 2% stop loss
        'take_profit_pct': 0.01,  # 1% take profit
        'trend_short_ma': 10,
        'trend_long_ma': 50,
        'atr_window': 14,
        'atr_multiplier': 0.5,
        'volume_window': 20,
        'volume_threshold': 1.2
    }
    
    print(f"Base parameters: {base_params}")
    
    # Test without filters first
    print("\nTesting strategy without filters...")
    df_signals_no_filter = generate_signals(df_obv, base_params)
    metrics_no_filter = calculate_returns(df_signals_no_filter, base_params)
    
    print(f"Without filters - Total Return: {metrics_no_filter['total_return']:.4f}, Trades: {metrics_no_filter['num_trades']}")
    
    # Optimize filters
    best_params, best_metrics = optimize_filters(df_obv, base_params)
    
    # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best filter parameters:")
    for key, value in best_params.items():
        if key not in base_params or base_params[key] != value:
            print(f"  {key}: {value}")
    
    print(f"\nPerformance comparison:")
    print(f"Without filters: Return={metrics_no_filter['total_return']:.4f}, Trades={metrics_no_filter['num_trades']}")
    print(f"With filters: Return={best_metrics['total_return']:.4f}, Trades={best_metrics['num_trades']}")
    
    print(f"\nOptimized results:")
    print(f"Total Return: {best_metrics['total_return']:.4f} ({best_metrics['total_return']*100:.2f}%)")
    print(f"Number of Trades: {best_metrics['num_trades']}")
    print(f"Winning Trades: {best_metrics['winning_trades']}")
    print(f"Losing Trades: {best_metrics['losing_trades']}")
    print(f"Win Rate: {best_metrics['win_rate']:.4f} ({best_metrics['win_rate']*100:.2f}%)")
    print(f"Average Win: {best_metrics['avg_win']:.4f} ({best_metrics['avg_win']*100:.2f}%)")
    print(f"Average Loss: {best_metrics['avg_loss']:.4f} ({best_metrics['avg_loss']*100:.2f}%)")
    print(f"Profit Factor: {best_metrics['profit_factor']:.4f}")
    print(f"Maximum Drawdown: {best_metrics['max_drawdown']:.4f} ({best_metrics['max_drawdown']*100:.2f}%)")
    
    print("System 3 test execution completed successfully!")

if __name__ == "__main__":
    main() 