import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimpleExitAnalyzer:
    """簡化出場分析器 - 專注於高勝率交易策略"""
    
    def __init__(self, data_file="TXF1_Minute_2020-01-01_2025-06-16.txt"):
        """初始化簡化出場分析器"""
        self.data_file = data_file
        self.df = None
        
    def load_data(self):
        """載入數據"""
        print(f"Loading data from {self.data_file}...")
        
        # Read the data file with proper column names
        df = pd.read_csv(self.data_file, sep='\t')
        
        # The first row contains the actual column names, so we need to parse it properly
        # Split the first column which contains all data
        first_col = df.columns[0]
        data_rows = df[first_col].str.split(',', expand=True)
        
        # Set proper column names
        data_rows.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TotalVolume']
        
        # Combine date and time
        data_rows['timestamp'] = pd.to_datetime(data_rows['Date'] + ' ' + data_rows['Time'])
        
        # Convert price columns to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'TotalVolume']:
            data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')
        
        # Set timestamp as index
        data_rows.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        data_rows.sort_index(inplace=True)
        
        # Drop rows with NaN values
        data_rows = data_rows.dropna()
        
        print(f"Data loaded: {len(data_rows)} records from {data_rows.index.min()} to {data_rows.index.max()}")
        self.df = data_rows
        return data_rows
    
    def convert_timeframe(self, timeframe='4H'):
        """轉換時間框架"""
        print(f"Converting to {timeframe} data...")
        
        # Resample to specified timeframe
        df_resampled = self.df.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'TotalVolume': 'sum'
        }).dropna()
        
        print(f"{timeframe} data created: {len(df_resampled)} records")
        return df_resampled
    
    def calculate_simple_indicators(self, df):
        """計算簡化的技術指標 - 只使用關鍵指標"""
        print("Calculating simple exit indicators...")
        
        # Convert to numpy arrays for TA-Lib
        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        volume = df['TotalVolume'].values.astype(float)
        
        # 1. RSI (相對強弱指數)
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # 2. 布林通道
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # 3. MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        
        # 4. 成交量指標
        df['obv'] = talib.OBV(close, volume)
        df['obv_ratio'] = df['obv'] / df['obv'].rolling(window=20).mean()
        
        return df
    
    def identify_simple_exit_signals(self, df, params=None):
        """識別簡化的出場信號"""
        if params is None:
            params = {
                'stop_loss_pct': 0.015,  # 1.5% 止損
                'take_profit_pct': 0.03,  # 3% 止盈
                'trailing_stop_pct': 0.008,  # 0.8% 追蹤止損
                'max_hold_periods': 20,  # 最大持倉期數
                'rsi_exit_oversold': 25,
                'rsi_exit_overbought': 75,
                'volume_exit_threshold': 0.7
            }
        
        print("Identifying simple exit signals...")
        
        # Initialize exit signal columns
        df['exit_signal'] = 0
        df['exit_reason'] = ''
        df['exit_type'] = ''
        
        # Simulate positions for exit analysis
        df['position'] = 0
        df['entry_price'] = 0.0
        df['entry_period'] = 0
        df['max_price'] = 0.0
        df['min_price'] = float('inf')
        
        position = 0
        entry_price = 0
        entry_period = 0
        max_price = 0
        min_price = float('inf')
        
        for i in range(20, len(df)):  # Start from 20 to ensure indicators are calculated
            current_price = df.iloc[i]['Close']
            current_period = i
            
            # Update position tracking
            if position != 0:
                df.iloc[i, df.columns.get_loc('position')] = position
                df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                df.iloc[i, df.columns.get_loc('entry_period')] = entry_period
                
                # Update max/min prices for trailing stops
                if position == 1:  # Long position
                    max_price = max(max_price, current_price)
                    df.iloc[i, df.columns.get_loc('max_price')] = max_price
                else:  # Short position
                    min_price = min(min_price, current_price)
                    df.iloc[i, df.columns.get_loc('min_price')] = min_price
                
                # Calculate current return
                if position == 1:  # Long position
                    current_return = (current_price - entry_price) / entry_price
                else:  # Short position
                    current_return = (entry_price - current_price) / entry_price
                
                # Check exit conditions
                exit_reason = ''
                exit_type = ''
                
                # 1. Stop Loss (優先級最高)
                if current_return <= -params['stop_loss_pct']:
                    exit_reason = f'Stop Loss ({current_return:.2%})'
                    exit_type = 'stop_loss'
                
                # 2. Take Profit
                elif current_return >= params['take_profit_pct']:
                    exit_reason = f'Take Profit ({current_return:.2%})'
                    exit_type = 'take_profit'
                
                # 3. Trailing Stop
                elif position == 1 and current_price <= max_price * (1 - params['trailing_stop_pct']):
                    exit_reason = f'Trailing Stop ({current_return:.2%})'
                    exit_type = 'trailing_stop'
                elif position == -1 and current_price >= min_price * (1 + params['trailing_stop_pct']):
                    exit_reason = f'Trailing Stop ({current_return:.2%})'
                    exit_type = 'trailing_stop'
                
                # 4. Time-based exit
                elif (current_period - entry_period) >= params['max_hold_periods']:
                    exit_reason = f'Time Exit ({current_return:.2%})'
                    exit_type = 'time'
                
                # 5. Signal-based exit (最後檢查)
                else:
                    signal_exit = self.check_simple_signal_exit(df.iloc[i], position, params)
                    if signal_exit:
                        exit_reason = signal_exit
                        exit_type = 'signal'
                
                # Execute exit if any condition is met
                if exit_reason:
                    df.iloc[i, df.columns.get_loc('exit_signal')] = 1
                    df.iloc[i, df.columns.get_loc('exit_reason')] = exit_reason
                    df.iloc[i, df.columns.get_loc('exit_type')] = exit_type
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_period = 0
                    max_price = 0
                    min_price = float('inf')
                    continue
            
            # Simulate entry signals (for demonstration)
            # In real implementation, this would come from entry strategy
            if position == 0 and i % 40 == 0:  # Simulate entry every 40 periods
                position = 1 if df.iloc[i]['rsi'] < 30 else -1
                entry_price = current_price
                entry_period = current_period
                max_price = current_price if position == 1 else 0
                min_price = current_price if position == -1 else float('inf')
        
        return df
    
    def check_simple_signal_exit(self, row, position, params):
        """檢查簡化的技術指標出場信號"""
        exit_reasons = []
        
        # RSI exit conditions
        if position == 1 and row['rsi'] >= params['rsi_exit_overbought']:
            exit_reasons.append('RSI Overbought')
        elif position == -1 and row['rsi'] <= params['rsi_exit_oversold']:
            exit_reasons.append('RSI Oversold')
        
        # MACD exit conditions
        if position == 1 and row['macd'] < row['macd_signal']:
            exit_reasons.append('MACD Bearish')
        elif position == -1 and row['macd'] > row['macd_signal']:
            exit_reasons.append('MACD Bullish')
        
        # Volume exit conditions
        if row['obv_ratio'] <= params['volume_exit_threshold']:
            exit_reasons.append('Low Volume')
        
        # Bollinger Bands exit
        if position == 1 and row['Close'] >= row['bb_upper']:
            exit_reasons.append('BB Upper')
        elif position == -1 and row['Close'] <= row['bb_lower']:
            exit_reasons.append('BB Lower')
        
        return ', '.join(exit_reasons) if exit_reasons else None
    
    def analyze_simple_exit_performance(self, df, lookback_days=30):
        """分析簡化出場表現"""
        print(f"Analyzing simple exit performance for the last {lookback_days} days...")
        
        # Get recent data
        recent_date = df.index.max() - timedelta(days=lookback_days)
        recent_df = df[df.index >= recent_date].copy()
        
        # Find exit signals
        exit_signals = recent_df[recent_df['exit_signal'] == 1]
        
        print(f"\n=== SIMPLE EXIT PERFORMANCE ANALYSIS ===")
        print(f"Analysis period: {recent_date} to {df.index.max()}")
        print(f"Total exit signals: {len(exit_signals)}")
        
        if len(exit_signals) > 0:
            # Analyze exit types
            exit_types = exit_signals['exit_type'].value_counts()
            print(f"\n--- EXIT TYPE BREAKDOWN ---")
            for exit_type, count in exit_types.items():
                print(f"{exit_type}: {count} ({count/len(exit_signals)*100:.1f}%)")
            
            # Analyze exit reasons
            exit_reasons = exit_signals['exit_reason'].value_counts()
            print(f"\n--- EXIT REASONS ---")
            for reason, count in exit_reasons.head(10).items():
                print(f"{reason}: {count}")
            
            # Calculate performance metrics
            total_trades = len(exit_signals)
            profitable_trades = len(exit_signals[exit_signals['exit_reason'].str.contains('Profit', na=False)])
            loss_trades = len(exit_signals[exit_signals['exit_reason'].str.contains('Loss', na=False)])
            
            print(f"\n--- PERFORMANCE METRICS ---")
            print(f"Total trades: {total_trades}")
            print(f"Profitable trades: {profitable_trades}")
            print(f"Loss trades: {loss_trades}")
            print(f"Win rate: {profitable_trades/total_trades*100:.1f}%" if total_trades > 0 else "Win rate: N/A")
            
            # Calculate average returns by exit type
            print(f"\n--- AVERAGE RETURNS BY EXIT TYPE ---")
            for exit_type in exit_types.index:
                type_exits = exit_signals[exit_signals['exit_type'] == exit_type]
                if len(type_exits) > 0:
                    # Extract return percentage from reason string
                    returns = []
                    for reason in type_exits['exit_reason']:
                        try:
                            return_pct = float(reason.split('(')[1].split('%')[0]) / 100
                            returns.append(return_pct)
                        except:
                            returns.append(0)
                    
                    avg_return = np.mean(returns) if returns else 0
                    print(f"{exit_type}: {avg_return:.2%} ({len(type_exits)} trades)")
        
        return exit_signals
    
    def plot_simple_exit_signals(self, df, lookback_days=30):
        """繪製簡化出場信號圖表"""
        print(f"Plotting simple exit signals for the last {lookback_days} days...")
        
        # Get recent data
        recent_date = df.index.max() - timedelta(days=lookback_days)
        recent_df = df[df.index >= recent_date].copy()
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Price and exit signals
        axes[0].plot(recent_df.index, recent_df['Close'], label='Close Price', alpha=0.7)
        
        # Plot exit signals by type
        exit_points = recent_df[recent_df['exit_signal'] == 1]
        if len(exit_points) > 0:
            # Color code by exit type
            colors = {'stop_loss': 'red', 'take_profit': 'green', 'trailing_stop': 'orange', 'time': 'purple', 'signal': 'blue'}
            for exit_type in exit_points['exit_type'].unique():
                type_points = exit_points[exit_points['exit_type'] == exit_type]
                color = colors.get(exit_type, 'black')
                axes[0].scatter(type_points.index, type_points['Close'], 
                               color=color, s=100, marker='x', label=f'{exit_type}')
        
        axes[0].set_title('Price and Exit Signals (Color-coded by Type)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: RSI and MACD
        axes[1].plot(recent_df.index, recent_df['rsi'], label='RSI', color='purple')
        axes[1].axhline(y=75, color='r', linestyle='--', alpha=0.5, label='Overbought (75)')
        axes[1].axhline(y=25, color='g', linestyle='--', alpha=0.5, label='Oversold (25)')
        axes[1].set_title('RSI')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_simple_analysis(self, timeframe='4H', lookback_days=30):
        """執行簡化的出場分析"""
        print("="*80)
        print("SIMPLE EXIT STRATEGY ANALYSIS")
        print("="*80)
        
        # Load and process data
        df = self.load_data()
        df = self.convert_timeframe(timeframe)
        df = self.calculate_simple_indicators(df)
        df = self.identify_simple_exit_signals(df)
        
        # Analyze results
        exit_signals = self.analyze_simple_exit_performance(df, lookback_days)
        
        # Plot results
        self.plot_simple_exit_signals(df, lookback_days)
        
        print("\nSimple exit analysis completed!")
        return df, exit_signals

def main():
    """主函數"""
    print("Simple Exit Strategy Analyzer")
    print("="*50)
    
    # Create analyzer
    analyzer = SimpleExitAnalyzer()
    
    # Run analysis
    df, exit_signals = analyzer.run_simple_analysis(timeframe='4H', lookback_days=30)
    
    print("\nSimple exit strategy analysis completed successfully!")

if __name__ == "__main__":
    main() 