import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimpleEntryAnalyzer:
    """簡化進場信號分析器 - 只使用 OBV、RSI 和布林通道"""
    
    def __init__(self, data_file="TXF1_Minute_2020-01-01_2025-06-16.txt"):
        """初始化分析器"""
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
    
    def calculate_indicators(self, df):
        """計算技術指標 - 只使用 OBV、RSI 和布林通道"""
        print("Calculating OBV, RSI and Bollinger Bands...")
        
        # Convert to numpy arrays for TA-Lib
        close = df['Close'].values.astype(float)
        volume = df['TotalVolume'].values.astype(float)
        
        # 1. RSI (相對強弱指數)
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # 2. OBV (能量潮)
        df['obv'] = talib.OBV(close, volume)
        df['obv_ratio'] = df['obv'] / df['obv'].rolling(window=20).mean()
        
        # 3. 布林通道
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        return df
    
    def identify_entry_signals(self, df, params=None):
        """識別進場信號 - 只使用三個指標"""
        if params is None:
            params = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'obv_threshold': 1.2,
                'bb_squeeze_threshold': 0.1
            }
        
        print("Identifying entry signals using OBV, RSI and Bollinger Bands...")
        
        # Initialize signal columns
        df['buy_signal'] = 0
        df['sell_signal'] = 0
        df['signal_strength'] = 0
        df['signal_reason'] = ''
        
        for i in range(20, len(df)):  # Start from 20 to ensure indicators are calculated
            buy_conditions = []
            sell_conditions = []
            signal_strength = 0
            reasons = []
            
            # 1. 布林通道信號
            if df.iloc[i]['Close'] <= df.iloc[i]['bb_lower']:
                buy_conditions.append(True)
                signal_strength += 1
                reasons.append('BB_Lower')
            elif df.iloc[i]['Close'] >= df.iloc[i]['bb_upper']:
                sell_conditions.append(True)
                signal_strength += 1
                reasons.append('BB_Upper')
            
            # 2. RSI 信號
            if df.iloc[i]['rsi'] <= params['rsi_oversold']:
                buy_conditions.append(True)
                signal_strength += 1
                reasons.append('RSI_Oversold')
            elif df.iloc[i]['rsi'] >= params['rsi_overbought']:
                sell_conditions.append(True)
                signal_strength += 1
                reasons.append('RSI_Overbought')
            
            # 3. OBV 成交量確認
            if df.iloc[i]['obv_ratio'] >= params['obv_threshold']:
                buy_conditions.append(True)
                signal_strength += 0.5
                reasons.append('OBV_Strong')
            elif df.iloc[i]['obv_ratio'] <= 1/params['obv_threshold']:
                sell_conditions.append(True)
                signal_strength += 0.5
                reasons.append('OBV_Weak')
            
            # 4. 布林通道擠壓 (低波動率)
            if df.iloc[i]['bb_width'] <= params['bb_squeeze_threshold']:
                signal_strength += 0.5
                reasons.append('BB_Squeeze')
            
            # 決定最終信號 - 需要至少2個條件
            if len(buy_conditions) >= 2 and signal_strength >= 2:
                df.iloc[i, df.columns.get_loc('buy_signal')] = 1
                df.iloc[i, df.columns.get_loc('signal_strength')] = signal_strength
                df.iloc[i, df.columns.get_loc('signal_reason')] = ', '.join(reasons)
            elif len(sell_conditions) >= 2 and signal_strength >= 2:
                df.iloc[i, df.columns.get_loc('sell_signal')] = 1
                df.iloc[i, df.columns.get_loc('signal_strength')] = signal_strength
                df.iloc[i, df.columns.get_loc('signal_reason')] = ', '.join(reasons)
        
        return df
    
    def analyze_entry_points(self, df, lookback_days=30):
        """分析最近的進場點"""
        print(f"Analyzing entry points for the last {lookback_days} days...")
        
        # Get recent data
        recent_date = df.index.max() - timedelta(days=lookback_days)
        recent_df = df[df.index >= recent_date].copy()
        
        # Find entry signals
        buy_signals = recent_df[recent_df['buy_signal'] == 1]
        sell_signals = recent_df[recent_df['sell_signal'] == 1]
        
        print(f"\n=== SIMPLE ENTRY POINT ANALYSIS ===")
        print(f"Analysis period: {recent_date} to {df.index.max()}")
        print(f"Total buy signals: {len(buy_signals)}")
        print(f"Total sell signals: {len(sell_signals)}")
        
        # Analyze buy signals
        if len(buy_signals) > 0:
            print(f"\n--- BUY SIGNALS ---")
            for idx, row in buy_signals.iterrows():
                print(f"Date: {idx}")
                print(f"Price: {row['Close']:.2f}")
                print(f"Signal Strength: {row['signal_strength']:.1f}")
                print(f"Reasons: {row['signal_reason']}")
                print(f"RSI: {row['rsi']:.1f}")
                print(f"OBV Ratio: {row['obv_ratio']:.2f}")
                print(f"BB Width: {row['bb_width']:.4f}")
                print(f"BB Position: {((row['Close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])):.2f}")
                print("-" * 50)
        
        # Analyze sell signals
        if len(sell_signals) > 0:
            print(f"\n--- SELL SIGNALS ---")
            for idx, row in sell_signals.iterrows():
                print(f"Date: {idx}")
                print(f"Price: {row['Close']:.2f}")
                print(f"Signal Strength: {row['signal_strength']:.1f}")
                print(f"Reasons: {row['signal_reason']}")
                print(f"RSI: {row['rsi']:.1f}")
                print(f"OBV Ratio: {row['obv_ratio']:.2f}")
                print(f"BB Width: {row['bb_width']:.4f}")
                print(f"BB Position: {((row['Close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])):.2f}")
                print("-" * 50)
        
        return buy_signals, sell_signals
    
    def plot_entry_signals(self, df, lookback_days=30):
        """繪製進場信號圖表 - 只顯示三個指標"""
        print("Generating entry signal charts...")
        
        # Get recent data
        recent_date = df.index.max() - timedelta(days=lookback_days)
        recent_df = df[df.index >= recent_date].copy()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Simple Entry Signal Analysis (OBV, RSI, Bollinger Bands)', fontsize=16, fontweight='bold')
        
        # 1. Price and Bollinger Bands with Entry Signals
        axes[0, 0].plot(recent_df.index, recent_df['Close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(recent_df.index, recent_df['bb_upper'], label='BB Upper', alpha=0.7)
        axes[0, 0].plot(recent_df.index, recent_df['bb_middle'], label='BB Middle', alpha=0.7)
        axes[0, 0].plot(recent_df.index, recent_df['bb_lower'], label='BB Lower', alpha=0.7)
        
        # Mark entry signals
        buy_signals = recent_df[recent_df['buy_signal'] == 1]
        sell_signals = recent_df[recent_df['sell_signal'] == 1]
        
        axes[0, 0].scatter(buy_signals.index, buy_signals['Close'], 
                          color='green', marker='^', s=100, label='Buy Signal', alpha=0.8)
        axes[0, 0].scatter(sell_signals.index, sell_signals['Close'], 
                          color='red', marker='v', s=100, label='Sell Signal', alpha=0.8)
        
        axes[0, 0].set_title('Price Chart with Bollinger Bands and Entry Signals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RSI with Entry Zones
        axes[0, 1].plot(recent_df.index, recent_df['rsi'], label='RSI', color='purple', linewidth=2)
        axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[0, 1].scatter(buy_signals.index, buy_signals['rsi'], color='green', marker='^', s=50)
        axes[0, 1].scatter(sell_signals.index, sell_signals['rsi'], color='red', marker='v', s=50)
        axes[0, 1].set_title('RSI with Entry Signals')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. OBV Ratio
        axes[1, 0].plot(recent_df.index, recent_df['obv_ratio'], label='OBV Ratio', color='orange', linewidth=2)
        axes[1, 0].axhline(y=1.2, color='g', linestyle='--', alpha=0.5, label='Strong Volume')
        axes[1, 0].axhline(y=0.83, color='r', linestyle='--', alpha=0.5, label='Weak Volume')
        axes[1, 0].scatter(buy_signals.index, buy_signals['obv_ratio'], color='green', marker='^', s=50)
        axes[1, 0].scatter(sell_signals.index, sell_signals['obv_ratio'], color='red', marker='v', s=50)
        axes[1, 0].set_title('OBV Ratio with Entry Signals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Bollinger Band Width
        axes[1, 1].plot(recent_df.index, recent_df['bb_width'], label='BB Width', color='darkred', linewidth=2)
        axes[1, 1].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Squeeze')
        axes[1, 1].scatter(buy_signals.index, buy_signals['bb_width'], color='green', marker='^', s=50)
        axes[1, 1].scatter(sell_signals.index, sell_signals['bb_width'], color='red', marker='v', s=50)
        axes[1, 1].set_title('Bollinger Band Width (Volatility)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('RUN/進場/進場/simple_entry_signals_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Chart saved as: RUN/進場/進場/simple_entry_signals_analysis.png")
    
    def run_analysis(self, timeframe='4H', lookback_days=30):
        """執行簡化的進場點分析"""
        print("="*60)
        print("SIMPLE ENTRY SIGNAL ANALYSIS (OBV, RSI, BOLLINGER BANDS)")
        print("="*60)
        
        # Load and process data
        self.load_data()
        df_4h = self.convert_timeframe(timeframe)
        
        # Calculate indicators
        df_indicators = self.calculate_indicators(df_4h)
        
        # Identify entry signals
        df_signals = self.identify_entry_signals(df_indicators)
        
        # Analyze entry points
        buy_signals, sell_signals = self.analyze_entry_points(df_signals, lookback_days)
        
        # Plot results
        self.plot_entry_signals(df_signals, lookback_days)
        
        # Save results
        df_signals.to_csv('RUN/進場/進場/simple_entry_signals_data.csv')
        print(f"\nData saved as: RUN/進場/進場/simple_entry_signals_data.csv")
        
        return df_signals, buy_signals, sell_signals

def main():
    """主函數"""
    # Create analyzer
    analyzer = SimpleEntryAnalyzer()
    
    # Run analysis
    df_signals, buy_signals, sell_signals = analyzer.run_analysis(
        timeframe='4H',
        lookback_days=30
    )
    
    print("\n" + "="*60)
    print("SIMPLE ANALYSIS COMPLETED")
    print("="*60)
    print("Key findings:")
    print(f"- Total buy signals in last 30 days: {len(buy_signals)}")
    print(f"- Total sell signals in last 30 days: {len(sell_signals)}")
    
    if len(buy_signals) > 0:
        avg_buy_strength = buy_signals['signal_strength'].mean()
        print(f"- Average buy signal strength: {avg_buy_strength:.2f}")
    
    if len(sell_signals) > 0:
        avg_sell_strength = sell_signals['signal_strength'].mean()
        print(f"- Average sell signal strength: {avg_sell_strength:.2f}")
    
    print("\nCheck the generated files for detailed analysis:")
    print("- simple_entry_signals_analysis.png: Visual charts")
    print("- simple_entry_signals_data.csv: Raw data with signals")
    
    print("\nSignal Criteria:")
    print("- Buy: Price at/below BB lower + RSI oversold + Strong OBV")
    print("- Sell: Price at/above BB upper + RSI overbought + Weak OBV")

if __name__ == "__main__":
    main() 