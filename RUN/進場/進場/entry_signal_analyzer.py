import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EntrySignalAnalyzer:
    """進場信號分析器 - 使用 TA-Lib 技術指標"""
    
    def __init__(self, data_file="TXF1_Minute_2020-01-01_2025-06-16.txt"):
        """初始化分析器"""
        self.data_file = data_file
        self.df = None
        self.signals = None
        
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
        """計算技術指標"""
        print("Calculating technical indicators...")
        
        # Convert to numpy arrays for TA-Lib
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        close = df['Close'].values.astype(float)
        volume = df['TotalVolume'].values.astype(float)
        
        # 1. 移動平均線
        df['sma_20'] = talib.SMA(close, timeperiod=20)
        df['sma_50'] = talib.SMA(close, timeperiod=50)
        df['ema_12'] = talib.EMA(close, timeperiod=12)
        df['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # 2. 布林通道
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # 3. RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)
        
        # 4. MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # 5. 隨機指標
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # 6. Williams %R
        df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # 7. ATR (波動率)
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # 8. ADX (趨勢強度)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # 9. OBV (成交量)
        df['obv'] = talib.OBV(close, volume)
        df['obv_ratio'] = df['obv'] / df['obv'].rolling(window=20).mean()
        
        # 10. CCI
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)
        
        # 11. 動量指標
        df['mom'] = talib.MOM(close, timeperiod=10)
        df['roc'] = talib.ROC(close, timeperiod=10)
        
        return df
    
    def identify_entry_signals(self, df, params=None):
        """識別進場信號"""
        if params is None:
            params = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'stoch_oversold': 20,
                'stoch_overbought': 80,
                'williams_oversold': -80,
                'williams_overbought': -20,
                'cci_oversold': -100,
                'cci_overbought': 100,
                'obv_threshold': 1.2,
                'adx_threshold': 25,
                'bb_squeeze_threshold': 0.1
            }
        
        print("Identifying entry signals...")
        
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
            
            # 3. MACD 信號
            if (df.iloc[i]['macd'] > df.iloc[i]['macd_signal'] and 
                df.iloc[i-1]['macd'] <= df.iloc[i-1]['macd_signal']):
                buy_conditions.append(True)
                signal_strength += 1
                reasons.append('MACD_Bullish_Cross')
            elif (df.iloc[i]['macd'] < df.iloc[i]['macd_signal'] and 
                  df.iloc[i-1]['macd'] >= df.iloc[i-1]['macd_signal']):
                sell_conditions.append(True)
                signal_strength += 1
                reasons.append('MACD_Bearish_Cross')
            
            # 4. 隨機指標信號
            if df.iloc[i]['stoch_k'] <= params['stoch_oversold']:
                buy_conditions.append(True)
                signal_strength += 1
                reasons.append('Stoch_Oversold')
            elif df.iloc[i]['stoch_k'] >= params['stoch_overbought']:
                sell_conditions.append(True)
                signal_strength += 1
                reasons.append('Stoch_Overbought')
            
            # 5. Williams %R 信號
            if df.iloc[i]['williams_r'] <= params['williams_oversold']:
                buy_conditions.append(True)
                signal_strength += 1
                reasons.append('Williams_Oversold')
            elif df.iloc[i]['williams_r'] >= params['williams_overbought']:
                sell_conditions.append(True)
                signal_strength += 1
                reasons.append('Williams_Overbought')
            
            # 6. CCI 信號
            if df.iloc[i]['cci'] <= params['cci_oversold']:
                buy_conditions.append(True)
                signal_strength += 1
                reasons.append('CCI_Oversold')
            elif df.iloc[i]['cci'] >= params['cci_overbought']:
                sell_conditions.append(True)
                signal_strength += 1
                reasons.append('CCI_Overbought')
            
            # 7. 移動平均線交叉
            if (df.iloc[i]['ema_12'] > df.iloc[i]['ema_26'] and 
                df.iloc[i-1]['ema_12'] <= df.iloc[i-1]['ema_26']):
                buy_conditions.append(True)
                signal_strength += 1
                reasons.append('EMA_Bullish_Cross')
            elif (df.iloc[i]['ema_12'] < df.iloc[i]['ema_26'] and 
                  df.iloc[i-1]['ema_12'] >= df.iloc[i-1]['ema_26']):
                sell_conditions.append(True)
                signal_strength += 1
                reasons.append('EMA_Bearish_Cross')
            
            # 8. 成交量確認
            if df.iloc[i]['obv_ratio'] >= params['obv_threshold']:
                buy_conditions.append(True)
                signal_strength += 0.5
                reasons.append('OBV_Strong')
            elif df.iloc[i]['obv_ratio'] <= 1/params['obv_threshold']:
                sell_conditions.append(True)
                signal_strength += 0.5
                reasons.append('OBV_Weak')
            
            # 9. 趨勢強度確認
            if df.iloc[i]['adx'] >= params['adx_threshold']:
                if len(buy_conditions) > len(sell_conditions):
                    signal_strength += 0.5
                    reasons.append('ADX_Strong_Trend')
                elif len(sell_conditions) > len(buy_conditions):
                    signal_strength += 0.5
                    reasons.append('ADX_Strong_Trend')
            
            # 10. 布林通道擠壓 (低波動率)
            if df.iloc[i]['bb_width'] <= params['bb_squeeze_threshold']:
                signal_strength += 0.5
                reasons.append('BB_Squeeze')
            
            # 決定最終信號
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
        
        print(f"\n=== ENTRY POINT ANALYSIS ===")
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
                print(f"MACD: {row['macd']:.4f}")
                print(f"Stoch %K: {row['stoch_k']:.1f}")
                print(f"Williams %R: {row['williams_r']:.1f}")
                print(f"CCI: {row['cci']:.1f}")
                print(f"BB Width: {row['bb_width']:.4f}")
                print(f"ADX: {row['adx']:.1f}")
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
                print(f"MACD: {row['macd']:.4f}")
                print(f"Stoch %K: {row['stoch_k']:.1f}")
                print(f"Williams %R: {row['williams_r']:.1f}")
                print(f"CCI: {row['cci']:.1f}")
                print(f"BB Width: {row['bb_width']:.4f}")
                print(f"ADX: {row['adx']:.1f}")
                print("-" * 50)
        
        return buy_signals, sell_signals
    
    def plot_entry_signals(self, df, lookback_days=30):
        """繪製進場信號圖表"""
        print("Generating entry signal charts...")
        
        # Get recent data
        recent_date = df.index.max() - timedelta(days=lookback_days)
        recent_df = df[df.index >= recent_date].copy()
        
        # Create subplots
        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        fig.suptitle('Entry Signal Analysis with TA-Lib Indicators', fontsize=16, fontweight='bold')
        
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
        
        axes[0, 0].set_title('Price Chart with Entry Signals')
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
        
        # 3. MACD
        axes[1, 0].plot(recent_df.index, recent_df['macd'], label='MACD', color='blue', linewidth=2)
        axes[1, 0].plot(recent_df.index, recent_df['macd_signal'], label='Signal', color='red', linewidth=2)
        axes[1, 0].bar(recent_df.index, recent_df['macd_hist'], label='Histogram', alpha=0.3)
        axes[1, 0].scatter(buy_signals.index, buy_signals['macd'], color='green', marker='^', s=50)
        axes[1, 0].scatter(sell_signals.index, sell_signals['macd'], color='red', marker='v', s=50)
        axes[1, 0].set_title('MACD with Entry Signals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Stochastic
        axes[1, 1].plot(recent_df.index, recent_df['stoch_k'], label='%K', color='blue', linewidth=2)
        axes[1, 1].plot(recent_df.index, recent_df['stoch_d'], label='%D', color='red', linewidth=2)
        axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[1, 1].axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[1, 1].scatter(buy_signals.index, buy_signals['stoch_k'], color='green', marker='^', s=50)
        axes[1, 1].scatter(sell_signals.index, sell_signals['stoch_k'], color='red', marker='v', s=50)
        axes[1, 1].set_title('Stochastic with Entry Signals')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Williams %R
        axes[2, 0].plot(recent_df.index, recent_df['williams_r'], label='Williams %R', color='orange', linewidth=2)
        axes[2, 0].axhline(y=-20, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[2, 0].axhline(y=-80, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[2, 0].scatter(buy_signals.index, buy_signals['williams_r'], color='green', marker='^', s=50)
        axes[2, 0].scatter(sell_signals.index, sell_signals['williams_r'], color='red', marker='v', s=50)
        axes[2, 0].set_title('Williams %R with Entry Signals')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. CCI
        axes[2, 1].plot(recent_df.index, recent_df['cci'], label='CCI', color='teal', linewidth=2)
        axes[2, 1].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[2, 1].axhline(y=-100, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[2, 1].scatter(buy_signals.index, buy_signals['cci'], color='green', marker='^', s=50)
        axes[2, 1].scatter(sell_signals.index, sell_signals['cci'], color='red', marker='v', s=50)
        axes[2, 1].set_title('CCI with Entry Signals')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 7. ADX (Trend Strength)
        axes[3, 0].plot(recent_df.index, recent_df['adx'], label='ADX', color='brown', linewidth=2)
        axes[3, 0].axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Strong Trend')
        axes[3, 0].scatter(buy_signals.index, buy_signals['adx'], color='green', marker='^', s=50)
        axes[3, 0].scatter(sell_signals.index, sell_signals['adx'], color='red', marker='v', s=50)
        axes[3, 0].set_title('ADX (Trend Strength)')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)
        
        # 8. Bollinger Band Width
        axes[3, 1].plot(recent_df.index, recent_df['bb_width'], label='BB Width', color='darkred', linewidth=2)
        axes[3, 1].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Squeeze')
        axes[3, 1].scatter(buy_signals.index, buy_signals['bb_width'], color='green', marker='^', s=50)
        axes[3, 1].scatter(sell_signals.index, sell_signals['bb_width'], color='red', marker='v', s=50)
        axes[3, 1].set_title('Bollinger Band Width')
        axes[3, 1].legend()
        axes[3, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('RUN/進場/進場/entry_signals_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Chart saved as: RUN/進場/進場/entry_signals_analysis.png")
    
    def run_analysis(self, timeframe='4H', lookback_days=30):
        """執行完整的進場點分析"""
        print("="*60)
        print("ENTRY SIGNAL ANALYSIS WITH TA-LIB")
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
        df_signals.to_csv('RUN/進場/進場/entry_signals_data.csv')
        print(f"\nData saved as: RUN/進場/進場/entry_signals_data.csv")
        
        return df_signals, buy_signals, sell_signals

def main():
    """主函數"""
    # Create analyzer
    analyzer = EntrySignalAnalyzer()
    
    # Run analysis
    df_signals, buy_signals, sell_signals = analyzer.run_analysis(
        timeframe='4H',
        lookback_days=30
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
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
    print("- entry_signals_analysis.png: Visual charts")
    print("- entry_signals_data.csv: Raw data with signals")

if __name__ == "__main__":
    main() 