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

def generate_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Generate trading signals based on BB + RSI + OBV strategy, with stop loss, take profit, and position management.
    
    Args:
        df: DataFrame with technical indicators
        params: Strategy parameters (including stop_loss_pct, take_profit_pct, max_position)
        
    Returns:
        DataFrame with signals and positions added
    """
    df = df.copy()
    df['signal'] = 0
    df['position'] = 0
    df['entry_price'] = np.nan
    
    stop_loss_pct = params.get('stop_loss_pct', 0.02)  # 2% default
    take_profit_pct = params.get('take_profit_pct', 0.01)  # 1% default
    max_position = params.get('max_position', 1)
    
    position = 0
    entry_price = np.nan
    for i in range(len(df)):
        # Entry conditions
        if position == 0:
            buy_condition = (
                (df['close'].iloc[i] <= df['bb_lower'].iloc[i]) &
                (df['rsi'].iloc[i] < params['rsi_oversold']) &
                (df['obv_ratio'].iloc[i] > params['obv_threshold'])
            )
            sell_condition = (
                (df['close'].iloc[i] >= df['bb_upper'].iloc[i]) &
                (df['rsi'].iloc[i] > params['rsi_overbought']) &
                (df['obv_ratio'].iloc[i] < 1/params['obv_threshold'])
            )
            if buy_condition and position < max_position:
                df['signal'].iloc[i] = 1
                position = 1
                entry_price = df['close'].iloc[i]
            elif sell_condition and position > -max_position:
                df['signal'].iloc[i] = -1
                position = -1
                entry_price = df['close'].iloc[i]
        else:
            # Check stop loss and take profit while holding position
            if position == 1:
                # Long position stop loss/take profit
                if (df['close'].iloc[i] <= entry_price * (1 - stop_loss_pct)) or (df['close'].iloc[i] >= entry_price * (1 + take_profit_pct)):
                    df['signal'].iloc[i] = -1  # Close long
                    position = 0
                    entry_price = np.nan
            elif position == -1:
                # Short position stop loss/take profit
                if (df['close'].iloc[i] >= entry_price * (1 + stop_loss_pct)) or (df['close'].iloc[i] <= entry_price * (1 - take_profit_pct)):
                    df['signal'].iloc[i] = 1  # Close short
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
    df['returns'] = 0.0
    
    # Calculate returns based on position changes
    for i in range(1, len(df)):
        if df['signal'].iloc[i] != 0 and df['position'].iloc[i-1] != 0:
            # Position closed
            if df['position'].iloc[i-1] == 1:  # Long position closed
                df['returns'].iloc[i] = (df['close'].iloc[i] - df['entry_price'].iloc[i-1]) / df['entry_price'].iloc[i-1]
            elif df['position'].iloc[i-1] == -1:  # Short position closed
                df['returns'].iloc[i] = (df['entry_price'].iloc[i-1] - df['close'].iloc[i]) / df['entry_price'].iloc[i-1]
    
    # Calculate cumulative returns
    df['cumulative_returns'] = df['returns'].cumsum()
    
    # Performance metrics
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
    
    # Maximum drawdown
    cumulative_returns = df['cumulative_returns']
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
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

def optimize_take_profit(df: pd.DataFrame, base_params: Dict) -> Tuple[Dict, Dict]:
    """
    Optimize take profit parameter.
    
    Args:
        df: DataFrame with technical indicators
        base_params: Base strategy parameters
        
    Returns:
        Tuple of (best_params, best_metrics)
    """
    print("Optimizing take profit parameter...")
    
    take_profit_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]  # 0.5% to 3%
    best_metrics = None
    best_params = None
    best_sharpe = -999
    
    for tp in take_profit_values:
        params = base_params.copy()
        params['take_profit_pct'] = tp
        
        # Generate signals and calculate returns
        df_signals = generate_signals(df, params)
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
    """Main function to run take profit optimization"""
    print("="*60)
    print("SYSTEM 2: Take Profit Optimization (TEST - 1000 records)")
    print("="*60)
    
    # Load and process data
    df_1min = load_txf_data("TXF1_Minute_2020-01-01_2025-06-16.txt")
    df_4h = convert_to_4h_data(df_1min)
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    df_bb = calculate_bollinger_bands(df_4h, window=20, num_std=2.0)
    df_rsi = calculate_rsi(df_bb, window=14)
    df_obv = calculate_obv(df_rsi)
    
    # Base strategy parameters
    base_params = {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'obv_threshold': 1.2,
        'stop_loss_pct': 0.02,  # 2% stop loss
        'take_profit_pct': 0.01,  # 1% take profit (will be optimized)
        'max_position': 1
    }
    
    print(f"Base parameters: {base_params}")
    
    # Optimize take profit
    best_params, best_metrics = optimize_take_profit(df_obv, base_params)
    
    # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best take profit: {best_params['take_profit_pct']:.3f} ({best_params['take_profit_pct']*100:.1f}%)")
    print(f"Total Return: {best_metrics['total_return']:.4f} ({best_metrics['total_return']*100:.2f}%)")
    print(f"Number of Trades: {best_metrics['num_trades']}")
    print(f"Winning Trades: {best_metrics['winning_trades']}")
    print(f"Losing Trades: {best_metrics['losing_trades']}")
    print(f"Win Rate: {best_metrics['win_rate']:.4f} ({best_metrics['win_rate']*100:.2f}%)")
    print(f"Average Win: {best_metrics['avg_win']:.4f} ({best_metrics['avg_win']*100:.2f}%)")
    print(f"Average Loss: {best_metrics['avg_loss']:.4f} ({best_metrics['avg_loss']*100:.2f}%)")
    print(f"Profit Factor: {best_metrics['profit_factor']:.4f}")
    print(f"Maximum Drawdown: {best_metrics['max_drawdown']:.4f} ({best_metrics['max_drawdown']*100:.2f}%)")
    
    print("System 2 test execution completed successfully!")

if __name__ == "__main__":
    main() 