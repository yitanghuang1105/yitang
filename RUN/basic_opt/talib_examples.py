import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import warnings
warnings.filterwarnings('ignore')

def demonstrate_talib_indicators():
    """Demonstrate various TA-Lib indicators with examples"""
    print("="*60)
    print("TA-Lib Technical Indicators Demonstration")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.uniform(0, 2, 100)
    low_prices = close_prices - np.random.uniform(0, 2, 100)
    open_prices = close_prices + np.random.uniform(-1, 1, 100)
    volume = np.random.uniform(1000, 5000, 100)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    print("Sample data created with 100 days of OHLCV data")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Convert to numpy arrays for TA-Lib
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    
    # 1. Moving Averages
    print("\n1. Moving Averages:")
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['wma_10'] = talib.WMA(close, timeperiod=10)
    print("   - Simple Moving Average (SMA) - 20 periods")
    print("   - Exponential Moving Average (EMA) - 12 periods")
    print("   - Weighted Moving Average (WMA) - 10 periods")
    
    # 2. Bollinger Bands
    print("\n2. Bollinger Bands:")
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    print("   - Upper Band, Middle Band (SMA), Lower Band")
    print("   - Band Width = (Upper - Lower) / Middle")
    
    # 3. RSI (Relative Strength Index)
    print("\n3. RSI (Relative Strength Index):")
    df['rsi'] = talib.RSI(close, timeperiod=14)
    print("   - 14-period RSI")
    print("   - Values range from 0 to 100")
    print("   - >70: Overbought, <30: Oversold")
    
    # 4. MACD (Moving Average Convergence Divergence)
    print("\n4. MACD:")
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    print("   - MACD Line (12,26,9)")
    print("   - Signal Line")
    print("   - Histogram")
    
    # 5. Stochastic Oscillator
    print("\n5. Stochastic Oscillator:")
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['stoch_k'] = slowk
    df['stoch_d'] = slowd
    print("   - %K line (5,3,3)")
    print("   - %D line")
    print("   - Values range from 0 to 100")
    
    # 6. Williams %R
    print("\n6. Williams %R:")
    df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
    print("   - 14-period Williams %R")
    print("   - Values range from 0 to -100")
    print("   - >-20: Overbought, <-80: Oversold")
    
    # 7. ATR (Average True Range)
    print("\n7. ATR (Average True Range):")
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    print("   - 14-period ATR")
    print("   - Measures volatility")
    
    # 8. ADX (Average Directional Index)
    print("\n8. ADX (Average Directional Index):")
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    print("   - 14-period ADX")
    print("   - Measures trend strength")
    print("   - >25: Strong trend, <20: Weak trend")
    
    # 9. OBV (On-Balance Volume)
    print("\n9. OBV (On-Balance Volume):")
    df['obv'] = talib.OBV(close, volume)
    print("   - Cumulative volume indicator")
    print("   - Confirms price trends")
    
    # 10. CCI (Commodity Channel Index)
    print("\n10. CCI (Commodity Channel Index):")
    df['cci'] = talib.CCI(high, low, close, timeperiod=14)
    print("   - 14-period CCI")
    print("   - >100: Overbought, <-100: Oversold")
    
    # 11. Momentum Indicators
    print("\n11. Momentum Indicators:")
    df['mom'] = talib.MOM(close, timeperiod=10)
    df['roc'] = talib.ROC(close, timeperiod=10)
    print("   - Momentum (10-period)")
    print("   - Rate of Change (10-period)")
    
    # 12. Volume Indicators
    print("\n12. Volume Indicators:")
    df['ad'] = talib.AD(high, low, close, volume)
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    print("   - Accumulation/Distribution Line")
    print("   - Chaikin A/D Oscillator")
    
    return df

def plot_talib_indicators(df):
    """Plot various TA-Lib indicators"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle('TA-Lib Technical Indicators Demonstration', fontsize=16, fontweight='bold')
    
    # 1. Price and Bollinger Bands
    axes[0, 0].plot(df.index, df['close'], label='Close', linewidth=2)
    axes[0, 0].plot(df.index, df['bb_upper'], label='BB Upper', alpha=0.7)
    axes[0, 0].plot(df.index, df['bb_middle'], label='BB Middle', alpha=0.7)
    axes[0, 0].plot(df.index, df['bb_lower'], label='BB Lower', alpha=0.7)
    axes[0, 0].fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1)
    axes[0, 0].set_title('Price & Bollinger Bands')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Moving Averages
    axes[0, 1].plot(df.index, df['close'], label='Close', linewidth=2)
    axes[0, 1].plot(df.index, df['sma_20'], label='SMA(20)', alpha=0.8)
    axes[0, 1].plot(df.index, df['ema_12'], label='EMA(12)', alpha=0.8)
    axes[0, 1].set_title('Moving Averages')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. RSI
    axes[0, 2].plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=2)
    axes[0, 2].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    axes[0, 2].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    axes[0, 2].set_title('RSI (14)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. MACD
    axes[1, 0].plot(df.index, df['macd'], label='MACD', color='blue', linewidth=2)
    axes[1, 0].plot(df.index, df['macd_signal'], label='Signal', color='red', linewidth=2)
    axes[1, 0].bar(df.index, df['macd_hist'], label='Histogram', alpha=0.3)
    axes[1, 0].set_title('MACD (12,26,9)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Stochastic
    axes[1, 1].plot(df.index, df['stoch_k'], label='%K', color='blue', linewidth=2)
    axes[1, 1].plot(df.index, df['stoch_d'], label='%D', color='red', linewidth=2)
    axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Overbought')
    axes[1, 1].axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Oversold')
    axes[1, 1].set_title('Stochastic Oscillator')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Williams %R
    axes[1, 2].plot(df.index, df['williams_r'], label='Williams %R', color='orange', linewidth=2)
    axes[1, 2].axhline(y=-20, color='r', linestyle='--', alpha=0.5, label='Overbought')
    axes[1, 2].axhline(y=-80, color='g', linestyle='--', alpha=0.5, label='Oversold')
    axes[1, 2].set_title('Williams %R (14)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. ATR
    axes[2, 0].plot(df.index, df['atr'], label='ATR', color='gray', linewidth=2)
    axes[2, 0].set_title('Average True Range (14)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. ADX
    axes[2, 1].plot(df.index, df['adx'], label='ADX', color='brown', linewidth=2)
    axes[2, 1].axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Strong Trend')
    axes[2, 1].set_title('Average Directional Index (14)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. CCI
    axes[2, 2].plot(df.index, df['cci'], label='CCI', color='teal', linewidth=2)
    axes[2, 2].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Overbought')
    axes[2, 2].axhline(y=-100, color='g', linestyle='--', alpha=0.5, label='Oversold')
    axes[2, 2].set_title('Commodity Channel Index (14)')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # 10. Momentum
    axes[3, 0].plot(df.index, df['mom'], label='Momentum', color='navy', linewidth=2)
    axes[3, 0].set_title('Momentum (10)')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    
    # 11. Rate of Change
    axes[3, 1].plot(df.index, df['roc'], label='ROC', color='darkgreen', linewidth=2)
    axes[3, 1].set_title('Rate of Change (10)')
    axes[3, 1].legend()
    axes[3, 1].grid(True, alpha=0.3)
    
    # 12. Bollinger Band Width
    axes[3, 2].plot(df.index, df['bb_width'], label='BB Width', color='darkred', linewidth=2)
    axes[3, 2].set_title('Bollinger Band Width')
    axes[3, 2].legend()
    axes[3, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('RUN/1/talib_indicators_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nChart saved as: RUN/1/talib_indicators_demo.png")

def show_talib_functions():
    """Show available TA-Lib function groups"""
    print("\n" + "="*60)
    print("TA-Lib Function Categories")
    print("="*60)
    
    categories = {
        'Overlap Studies': ['SMA', 'EMA', 'WMA', 'BBANDS', 'DEMA', 'TEMA', 'TRIMA', 'KAMA'],
        'Momentum Indicators': ['RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'MACD', 'MOM', 'ROC', 'ROCP', 'ROCR', 'ROCR100'],
        'Volume Indicators': ['OBV', 'AD', 'ADOSC'],
        'Volatility Indicators': ['ATR', 'NATR', 'TRANGE'],
        'Price Transform': ['AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'],
        'Cycle Indicators': ['HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE'],
        'Pattern Recognition': ['CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'],
        'Statistic Functions': ['BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR']
    }
    
    for category, functions in categories.items():
        print(f"\n{category}:")
        if len(functions) > 8:
            print(f"   {', '.join(functions[:8])}... (and {len(functions)-8} more)")
        else:
            print(f"   {', '.join(functions)}")

def main():
    """Main function to demonstrate TA-Lib usage"""
    print("TA-Lib Technical Analysis Library Demonstration")
    print("This script shows how to use various TA-Lib indicators")
    
    # Demonstrate indicators
    df = demonstrate_talib_indicators()
    
    # Show available functions
    show_talib_functions()
    
    # Plot indicators
    print("\nGenerating indicator charts...")
    plot_talib_indicators(df)
    
    print("\n" + "="*60)
    print("Key TA-Lib Usage Tips:")
    print("="*60)
    print("1. Always convert pandas Series to numpy arrays: series.values")
    print("2. Most functions return numpy arrays, convert back to pandas if needed")
    print("3. Handle NaN values appropriately (many indicators have warm-up periods)")
    print("4. Use appropriate time periods for your trading strategy")
    print("5. Combine multiple indicators for better signal confirmation")
    print("6. Consider market conditions when interpreting indicators")
    
    print("\nTA-Lib demonstration completed successfully!")

if __name__ == "__main__":
    main() 