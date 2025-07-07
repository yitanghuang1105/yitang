import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import itertools
from datetime import datetime, timedelta
import warnings
import glob
import json
warnings.filterwarnings('ignore')

def load_txf_data(file_path: str) -> pd.DataFrame:
    """
    Load TXF 5-minute K data from txt format and convert to datetime index.
    
    Args:
        file_path: Path to the TXF data file
        
    Returns:
        DataFrame with datetime index and OHLCV data
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    if 'timestamp' not in df.columns and 'Date' in df.columns and 'Time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('timestamp', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'TotalVolume']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']  # Standardize column names for technical indicators
    return df

def convert_to_4h_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-minute data to 4-hour (240-minute) data.
    
    Args:
        df: DataFrame with 1-minute OHLCV data
        
    Returns:
        DataFrame with 4-hour OHLCV data
    """
    df = df.copy()
    
    # Reset index to get datetime as column for grouping
    df = df.reset_index()
    
    # Create 4-hour periods
    # Group by date and 4-hour blocks (240 minutes)
    df['date'] = df['timestamp'].dt.date
    df['hour_block'] = df['timestamp'].dt.hour // 4  # 0-5 for 4-hour blocks
    df['period'] = df['date'].astype(str) + '_' + df['hour_block'].astype(str)
    
    # Group by period and aggregate
    grouped = df.groupby('period').agg({
        'timestamp': 'first',  # First datetime of the period
        'open': 'first',      # First open price
        'high': 'max',        # Maximum high price
        'low': 'min',         # Minimum low price
        'close': 'last',      # Last close price
        'volume': 'sum'       # Sum of volume
    })
    
    # Set datetime as index
    grouped = grouped.set_index('timestamp')
    
    # Sort by datetime
    grouped = grouped.sort_index()
    
    print(f"Converted {len(df)} 1-minute records to {len(grouped)} 4-hour records")
    print(f"Date range: {grouped.index[0]} to {grouped.index[-1]}")
    
    return grouped

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
    
    # Calculate volume moving average
    df['volume_ma'] = df['volume'].rolling(window=volume_window).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Volume filter: only trade when volume is above threshold
    df['volume_ok'] = df['volume_ratio'] > volume_threshold
    
    return df

def calculate_time_filter(df: pd.DataFrame, start_time: str = "09:00", end_time: str = "13:30") -> pd.DataFrame:
    """
    Calculate time filter to avoid trading during low liquidity periods.
    
    Args:
        df: DataFrame with datetime index
        start_time: Start time for trading (HH:MM)
        end_time: End time for trading (HH:MM)
        
    Returns:
        DataFrame with time filter added
    """
    df = df.copy()
    
    # Extract time from datetime index
    df['time'] = df.index.time
    
    # Convert time strings to time objects
    start_time_obj = datetime.strptime(start_time, "%H:%M").time()
    end_time_obj = datetime.strptime(end_time, "%H:%M").time()
    
    # Time filter: only trade during specified hours
    df['time_ok'] = (df['time'] >= start_time_obj) & (df['time'] <= end_time_obj)
    
    return df

def apply_signal_filters(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Apply all signal filters to reduce false signals.
    
    Args:
        df: DataFrame with all indicators
        params: Strategy parameters including filter settings
        
    Returns:
        DataFrame with filter results added
    """
    df = df.copy()
    
    # Apply trend filter
    short_ma = params.get('trend_short_ma', 10)
    long_ma = params.get('trend_long_ma', 50)
    df = calculate_trend_filter(df, short_ma, long_ma)
    
    # Apply volatility filter
    atr_window = params.get('atr_window', 14)
    atr_multiplier = params.get('atr_multiplier', 0.5)
    df = calculate_volatility_filter(df, atr_window, atr_multiplier)
    
    # Apply volume filter
    volume_window = params.get('volume_window', 20)
    volume_threshold = params.get('volume_threshold', 1.2)
    df = calculate_volume_filter(df, volume_window, volume_threshold)
    
    # Apply time filter
    start_time = params.get('start_time', "09:00")
    end_time = params.get('end_time', "13:30")
    df = calculate_time_filter(df, start_time, end_time)
    
    # Combine all filters
    df['all_filters_ok'] = (
        df['trend_up'] | df['trend_down'] &  # Trend filter
        df['volatility_ok'] &                # Volatility filter
        df['volume_ok'] &                    # Volume filter
        df['time_ok']                        # Time filter
    )
    
    return df

def generate_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Generate trading signals with comprehensive filtering.
    
    Args:
        df: DataFrame with technical indicators and filters
        params: Strategy parameters
        
    Returns:
        DataFrame with signals and positions added
    """
    df = df.copy()
    df['signal'] = 0
    df['position'] = 0
    df['entry_price'] = np.nan
    
    stop_loss_pct = params.get('stop_loss_pct', 0.02)
    take_profit_pct = params.get('take_profit_pct', 0.01)
    max_position = params.get('max_position', 1)
    min_hold_periods = params.get('min_hold_periods', 5)  # Minimum holding periods
    
    position = 0
    entry_price = np.nan
    entry_time = 0
    
    for i in range(len(df)):
        # Entry conditions with filters
        if position == 0:
            buy_condition = (
                (df['close'].iloc[i] <= df['bb_lower'].iloc[i]) &
                (df['rsi'].iloc[i] < params['rsi_oversold']) &
                (df['obv_ratio'].iloc[i] > params['obv_threshold']) &
                df['trend_up'].iloc[i] &  # Trend filter
                df['all_filters_ok'].iloc[i]  # All filters
            )
            
            sell_condition = (
                (df['close'].iloc[i] >= df['bb_upper'].iloc[i]) &
                (df['rsi'].iloc[i] > params['rsi_overbought']) &
                (df['obv_ratio'].iloc[i] < 1/params['obv_threshold']) &
                df['trend_down'].iloc[i] &  # Trend filter
                df['all_filters_ok'].iloc[i]  # All filters
            )
            
            if buy_condition and position < max_position:
                df['signal'].iloc[i] = 1
                position = 1
                entry_price = df['close'].iloc[i]
                entry_time = i
            elif sell_condition and position > -max_position:
                df['signal'].iloc[i] = -1
                position = -1
                entry_price = df['close'].iloc[i]
                entry_time = i
        else:
            # Check minimum holding period
            if i - entry_time < min_hold_periods:
                continue
                
            # Check stop loss and take profit
            if position == 1:
                if (df['close'].iloc[i] <= entry_price * (1 - stop_loss_pct)) or (df['close'].iloc[i] >= entry_price * (1 + take_profit_pct)):
                    df['signal'].iloc[i] = -1
                    position = 0
                    entry_price = np.nan
            elif position == -1:
                if (df['close'].iloc[i] >= entry_price * (1 + stop_loss_pct)) or (df['close'].iloc[i] <= entry_price * (1 - take_profit_pct)):
                    df['signal'].iloc[i] = 1
                    position = 0
                    entry_price = np.nan
                    
        df['position'].iloc[i] = position
        df['entry_price'].iloc[i] = entry_price
    
    return df

def calculate_returns(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Calculate strategy returns and performance metrics.
    
    Args:
        df: DataFrame with signals and positions
        params: Strategy parameters
        
    Returns:
        Dictionary with performance metrics
    """
    df = df.copy()
    
    # Calculate strategy returns
    df['price_change'] = df['close'].pct_change()
    df['strategy_return'] = df['position'].shift(1) * df['price_change']
    
    # Apply transaction costs
    transaction_cost = params.get('transaction_cost', 0.0001)
    position_changes = df['position'].diff().abs()
    df['transaction_costs'] = position_changes * transaction_cost
    df['net_return'] = df['strategy_return'] - df['transaction_costs']
    
    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['net_return']).cumprod()
    
    # Calculate performance metrics
    total_return = df['cumulative_return'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(df)) - 1
    
    # Calculate volatility
    daily_returns = df['net_return'].resample('D').sum()
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative = df['cumulative_return']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calculate win rate and filter effectiveness
    trades = df[df['signal'] != 0]
    if len(trades) > 0:
        winning_trades = trades[trades['strategy_return'] > 0]
        win_rate = len(winning_trades) / len(trades)
    else:
        win_rate = 0
    
    # Calculate filter effectiveness
    total_signals = len(df[df['all_filters_ok'] == False])
    filtered_signals = len(df[(df['all_filters_ok'] == False) & (df['signal'] != 0)])
    filter_effectiveness = filtered_signals / total_signals if total_signals > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'filter_effectiveness': filter_effectiveness,
        'final_cumulative_return': df['cumulative_return'].iloc[-1]
    }

def optimize_parameters(df: pd.DataFrame, param_ranges: Dict) -> Tuple[Dict, Dict]:
    """
    Optimize strategy parameters including filter parameters.
    
    Args:
        df: DataFrame with OHLCV data
        param_ranges: Dictionary with parameter ranges to test
        
    Returns:
        Tuple of (best_params, best_metrics)
    """
    print("Starting parameter optimization with signal filtering...")
    
    # Generate all parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))
    
    best_sharpe = -np.inf
    best_params = None
    best_metrics = None
    results = []
    
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")
    
    # Add default parameters if not specified
    default_params = {
        'stop_loss_pct': [0.02],
        'take_profit_pct': [0.01],
        'max_position': [1],
        'min_hold_periods': [5],
        'trend_short_ma': [10],
        'trend_long_ma': [50],
        'atr_window': [14],
        'atr_multiplier': [0.5],
        'volume_window': [20],
        'volume_threshold': [1.2],
        'start_time': ["09:00"],
        'end_time': ["13:30"]
    }
    
    for key, value in default_params.items():
        if key not in param_ranges:
            param_ranges[key] = value
    
    for i, combination in enumerate(param_combinations):
        if i % 100 == 0:
            print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
        
        # Create parameter dictionary
        params = dict(zip(param_names, combination))
        
        # Add default parameters
        default_params = {
            'bb_window': 20,
            'bb_std': 2.0,
            'rsi_window': 14,
            'transaction_cost': 0.0001
        }
        params.update(default_params)
        
        try:
            # Calculate indicators
            df_with_indicators = calculate_bollinger_bands(df, params['bb_window'], params['bb_std'])
            df_with_indicators = calculate_rsi(df_with_indicators, params['rsi_window'])
            df_with_indicators = calculate_obv(df_with_indicators)
            
            # Apply signal filters
            df_with_filters = apply_signal_filters(df_with_indicators, params)
            
            # Generate signals
            df_with_signals = generate_signals(df_with_filters, params)
            
            # Calculate performance
            metrics = calculate_returns(df_with_signals, params)
            
            results.append({
                'params': params.copy(),
                'metrics': metrics
            })
            
            # Update best parameters if Sharpe ratio is better
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = params.copy()
                best_metrics = metrics.copy()
                
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            continue
    
    print(f"Optimization completed. Best Sharpe ratio: {best_sharpe:.4f}")
    return best_params, best_metrics

def load_latest_params():
    param_files = sorted(glob.glob("strategy_params_filter_optimization_*.json"), reverse=True)
    if not param_files:
        print("No parameter file found! Using default parameters.")
        return {
            'bb_window': 20,
            'bb_std': 2.0,
            'rsi_window': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'obv_threshold': 1.2,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.01,
            'max_position': 1,
            'trend_short_ma': 10,
            'trend_long_ma': 50,
            'atr_window': 14,
            'atr_multiplier': 0.5,
            'volume_window': 20,
            'volume_threshold': 1.2,
            'min_hold_periods': 5
        }
    with open(param_files[0], "r") as f:
        data = json.load(f)
        return data.get("filter_optimization", data)

def main():
    """
    Main function to run the parameter optimization with signal filtering.
    """
    # Load data
    print("Loading TXF 1-minute data...")
    file_path = "TXF1_Minute_2020-01-01_2025-06-16.txt"
    df_1min = load_txf_data(file_path)
    print(f"1-minute data loaded: {len(df_1min)} records from {df_1min.index[0]} to {df_1min.index[-1]}")
    
    # Convert to 4-hour data
    print("Converting to 4-hour data...")
    df = convert_to_4h_data(df_1min)
    print(f"4-hour data ready: {len(df)} records from {df.index[0]} to {df.index[-1]}")
    
    # Enhanced strategy parameters
    params = load_latest_params()
    print(f"Strategy parameters: {params}")
    
    # Run optimization
    best_params, best_metrics = optimize_parameters(df, params)
    
    # Print results
    print("\n" + "="*60)
    print("SIGNAL FILTERING OPTIMIZATION RESULTS (4-HOUR DATA)")
    print("="*60)
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print("\nBest Performance Metrics:")
    for key, value in best_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return best_params, best_metrics

if __name__ == "__main__":
    best_params, best_metrics = main()
