import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import talib
warnings.filterwarnings('ignore')

def load_txf_data(file_path):
    """Load TXF 1-minute data from file"""
    print(f"Loading data from {file_path}...")
    
    # Read the data file
    df = pd.read_csv(file_path, sep='\t')
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Sort by timestamp
    df.sort_index(inplace=True)
    
    print(f"Data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
    return df

def convert_to_4h_data(df_1min):
    """Convert 1-minute data to 4-hour data"""
    print("Converting to 4-hour data...")
    
    # Resample to 4-hour intervals
    df_4h = df_1min.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"4-hour data created: {len(df_4h)} records")
    return df_4h

def calculate_technical_indicators_talib(df):
    """Calculate technical indicators using TA-Lib"""
    df = df.copy()
    
    # Convert to numpy arrays for TA-Lib
    close = df['close'].values
    volume = df['volume'].values
    
    # Bollinger Bands using TA-Lib
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        close, 
        timeperiod=20, 
        nbdevup=2, 
        nbdevdn=2, 
        matype=0
    )
    
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle  # This is the moving average
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    # RSI using TA-Lib
    df['rsi'] = talib.RSI(close, timeperiod=14)
    
    # OBV using TA-Lib
    df['obv'] = talib.OBV(close, volume)
    df['obv_ratio'] = df['obv'] / df['obv'].rolling(window=20).mean()
    
    return df

def generate_signals_enhanced(df, params):
    """Generate enhanced trading signals using multiple TA-Lib indicators"""
    df = df.copy()
    
    # Initialize signal column
    df['signal'] = 0
    
    # Buy signals - Enhanced conditions
    buy_conditions = (
        (df['close'] <= df['bb_lower']) &  # Price at or below lower Bollinger Band
        (df['rsi'] <= params['rsi_oversold']) &  # RSI oversold
        (df['obv_ratio'] >= params['obv_threshold']) &  # Strong volume
        (df['macd'] > df['macd_signal']) &  # MACD bullish crossover
        (df['stoch_k'] <= 20) &  # Stochastic oversold
        (df['williams_r'] <= -80) &  # Williams %R oversold
        (df['adx'] >= params['adx_threshold'])  # Strong trend
    )
    
    # Sell signals - Enhanced conditions
    sell_conditions = (
        (df['close'] >= df['bb_upper']) &  # Price at or above upper Bollinger Band
        (df['rsi'] >= params['rsi_overbought']) &  # RSI overbought
        (df['obv_ratio'] <= 1/params['obv_threshold']) &  # Weak volume
        (df['macd'] < df['macd_signal']) &  # MACD bearish crossover
        (df['stoch_k'] >= 80) &  # Stochastic overbought
        (df['williams_r'] >= -20) &  # Williams %R overbought
        (df['adx'] >= params['adx_threshold'])  # Strong trend
    )
    
    # Set signals
    df.loc[buy_conditions, 'signal'] = 1  # Buy signal
    df.loc[sell_conditions, 'signal'] = -1  # Sell signal
    
    return df

def calculate_returns(df, params):
    """Calculate trading returns and performance metrics"""
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

def plot_enhanced_indicators(df_results):
    """Plot enhanced technical indicators"""
    plt.figure(figsize=(20, 15))
    
    # Price and Bollinger Bands
    plt.subplot(4, 2, 1)
    plt.plot(df_results.index, df_results['close'], label='Close Price', alpha=0.7)
    plt.plot(df_results.index, df_results['bb_upper'], label='BB Upper', alpha=0.5)
    plt.plot(df_results.index, df_results['bb_lower'], label='BB Lower', alpha=0.5)
    plt.plot(df_results.index, df_results['bb_middle'], label='BB Middle', alpha=0.5)
    
    # Mark buy/sell signals
    buy_signals = df_results[df_results['signal'] == 1]
    sell_signals = df_results[df_results['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], 
               color='green', marker='^', s=100, label='Buy Signal', alpha=0.7)
    plt.scatter(sell_signals.index, sell_signals['close'], 
               color='red', marker='v', s=100, label='Sell Signal', alpha=0.7)
    
    plt.title('Price Chart with Bollinger Bands and Signals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # RSI
    plt.subplot(4, 2, 2)
    plt.plot(df_results.index, df_results['rsi'], label='RSI', color='purple')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    plt.title('RSI Indicator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MACD
    plt.subplot(4, 2, 3)
    plt.plot(df_results.index, df_results['macd'], label='MACD', color='blue')
    plt.plot(df_results.index, df_results['macd_signal'], label='MACD Signal', color='red')
    plt.bar(df_results.index, df_results['macd_hist'], label='MACD Histogram', alpha=0.3)
    plt.title('MACD Indicator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Stochastic
    plt.subplot(4, 2, 4)
    plt.plot(df_results.index, df_results['stoch_k'], label='%K', color='blue')
    plt.plot(df_results.index, df_results['stoch_d'], label='%D', color='red')
    plt.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Overbought')
    plt.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Oversold')
    plt.title('Stochastic Oscillator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Williams %R
    plt.subplot(4, 2, 5)
    plt.plot(df_results.index, df_results['williams_r'], label='Williams %R', color='orange')
    plt.axhline(y=-20, color='r', linestyle='--', alpha=0.5, label='Overbought')
    plt.axhline(y=-80, color='g', linestyle='--', alpha=0.5, label='Oversold')
    plt.title('Williams %R')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ADX
    plt.subplot(4, 2, 6)
    plt.plot(df_results.index, df_results['adx'], label='ADX', color='brown')
    plt.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Strong Trend')
    plt.title('Average Directional Index (ADX)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ATR
    plt.subplot(4, 2, 7)
    plt.plot(df_results.index, df_results['atr'], label='ATR', color='gray')
    plt.title('Average True Range (ATR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative Returns
    plt.subplot(4, 2, 8)
    plt.plot(df_results.index, df_results['cumulative_returns'], 
             label='Cumulative Returns', color='blue')
    plt.title('Cumulative Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('RUN/1/system1_talib_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the enhanced strategy using TA-Lib"""
    print("="*60)
    print("SYSTEM 1: Enhanced Strategy with TA-Lib")
    print("="*60)
    
    # Load and process data
    df_1min = load_txf_data("TXF1_Minute_2020-01-01_2025-06-16.txt")
    df_4h = convert_to_4h_data(df_1min)
    
    # Calculate technical indicators using TA-Lib
    print("Calculating technical indicators using TA-Lib...")
    df_indicators = calculate_technical_indicators_talib(df_4h)
    
    # Enhanced strategy parameters
    params = {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'obv_threshold': 1.2,
        'adx_threshold': 25,  # Minimum ADX for trend strength
        'stop_loss_pct': 0.02,  # 2% stop loss
        'take_profit_pct': 0.01  # 1% take profit
    }
    
    print(f"Enhanced strategy parameters: {params}")
    
    # Generate enhanced signals
    print("Generating enhanced trading signals...")
    df_signals = generate_signals_enhanced(df_indicators, params)
    
    # Count signals
    signal_counts = df_signals['signal'].value_counts()
    print(f"Signal distribution: {signal_counts.to_dict()}")
    
    # Calculate performance
    print("Calculating performance metrics...")
    df_results, metrics = calculate_returns(df_signals, params)
    
    # Display results
    print("\n" + "="*60)
    print("ENHANCED PERFORMANCE RESULTS")
    print("="*60)
    print(f"Total Return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.4f} ({metrics['win_rate']*100:.2f}%)")
    print(f"Average Win: {metrics['avg_win']:.4f} ({metrics['avg_win']*100:.2f}%)")
    print(f"Average Loss: {metrics['avg_loss']:.4f} ({metrics['avg_loss']*100:.2f}%)")
    print(f"Profit Factor: {metrics['profit_factor']:.4f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
    
    # Plot enhanced results
    print("\nGenerating enhanced performance chart...")
    plot_enhanced_indicators(df_results)
    
    print(f"\nChart saved as: RUN/1/system1_talib_performance.png")
    print("Enhanced System 1 with TA-Lib execution completed successfully!")

if __name__ == "__main__":
    main()