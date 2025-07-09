import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_strategy_performance(csv_file):
    """Analyze strategy performance from CSV results"""
    
    print(f"Analyzing strategy performance from: {csv_file}")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Basic statistics
    print(f"Data period: {df.index.min()} to {df.index.max()}")
    print(f"Total data points: {len(df):,}")
    print(f"Data frequency: {df.index.freq if df.index.freq else 'Irregular'}")
    
    # Decision analysis
    decision_counts = df['decision'].value_counts()
    print(f"\nDecision Distribution:")
    for decision, count in decision_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {decision}: {count:,} ({percentage:.1f}%)")
    
    # Strategy scores analysis
    print(f"\nStrategy Scores Summary:")
    score_columns = ['rsi_score', 'bollinger_bands_score', 'obv_score', 'combined_score']
    for col in score_columns:
        if col in df.columns:
            mean_score = df[col].mean()
            std_score = df[col].std()
            min_score = df[col].min()
            max_score = df[col].max()
            print(f"  {col}:")
            print(f"    Mean: {mean_score:.2f}")
            print(f"    Std:  {std_score:.2f}")
            print(f"    Range: {min_score:.2f} - {max_score:.2f}")
    
    # Price movement analysis
    df['price_change'] = df['close'].pct_change()
    df['price_change_pct'] = df['price_change'] * 100
    
    print(f"\nPrice Movement Analysis:")
    print(f"  Average price change: {df['price_change_pct'].mean():.4f}%")
    print(f"  Price volatility: {df['price_change_pct'].std():.4f}%")
    print(f"  Total price change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # Volume analysis
    if 'volume' in df.columns:
        print(f"\nVolume Analysis:")
        print(f"  Average volume: {df['volume'].mean():.0f}")
        print(f"  Total volume: {df['volume'].sum():,}")
        print(f"  Volume volatility: {df['volume'].std():.0f}")
    
    # Strategy signal analysis
    print(f"\nStrategy Signal Analysis:")
    
    # Count signals by strategy
    signal_threshold = 60  # Threshold for buy/sell signals
    
    for strategy in ['rsi_score', 'bollinger_bands_score', 'obv_score']:
        if strategy in df.columns:
            buy_signals = (df[strategy] > signal_threshold).sum()
            sell_signals = (df[strategy] < (100 - signal_threshold)).sum()
            neutral_signals = len(df) - buy_signals - sell_signals
            
            print(f"  {strategy}:")
            print(f"    Buy signals: {buy_signals:,} ({buy_signals/len(df)*100:.1f}%)")
            print(f"    Sell signals: {sell_signals:,} ({sell_signals/len(df)*100:.1f}%)")
            print(f"    Neutral: {neutral_signals:,} ({neutral_signals/len(df)*100:.1f}%)")
    
    # Combined strategy analysis
    if 'combined_score' in df.columns:
        strong_buy = (df['combined_score'] > 70).sum()
        buy = ((df['combined_score'] > 60) & (df['combined_score'] <= 70)).sum()
        hold = ((df['combined_score'] >= 40) & (df['combined_score'] <= 60)).sum()
        sell = ((df['combined_score'] >= 30) & (df['combined_score'] < 40)).sum()
        strong_sell = (df['combined_score'] < 30).sum()
        
        print(f"\nCombined Strategy Signal Distribution:")
        print(f"  Strong Buy (>70): {strong_buy:,} ({strong_buy/len(df)*100:.1f}%)")
        print(f"  Buy (60-70): {buy:,} ({buy/len(df)*100:.1f}%)")
        print(f"  Hold (40-60): {hold:,} ({hold/len(df)*100:.1f}%)")
        print(f"  Sell (30-40): {sell:,} ({sell/len(df)*100:.1f}%)")
        print(f"  Strong Sell (<30): {strong_sell:,} ({strong_sell/len(df)*100:.1f}%)")
    
    # Performance metrics (simplified)
    print(f"\nPerformance Metrics:")
    
    # Calculate simple returns based on decisions
    df['strategy_return'] = 0.0
    
    # Simple strategy: Buy when combined_score > 60, Sell when < 40
    buy_condition = df['combined_score'] > 60
    sell_condition = df['combined_score'] < 40
    
    df.loc[buy_condition, 'strategy_return'] = df.loc[buy_condition, 'price_change']
    df.loc[sell_condition, 'strategy_return'] = -df.loc[sell_condition, 'price_change']
    
    # Calculate cumulative returns
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    
    total_return = (df['cumulative_return'].iloc[-1] - 1) * 100
    buy_hold_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
    
    print(f"  Strategy Total Return: {total_return:.2f}%")
    print(f"  Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"  Strategy vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
    
    # Calculate win rate
    positive_returns = (df['strategy_return'] > 0).sum()
    total_signals = (df['strategy_return'] != 0).sum()
    win_rate = (positive_returns / total_signals * 100) if total_signals > 0 else 0
    
    print(f"  Win Rate: {win_rate:.1f}% ({positive_returns}/{total_signals} signals)")
    
    # Calculate Sharpe ratio (simplified)
    if df['strategy_return'].std() > 0:
        sharpe_ratio = (df['strategy_return'].mean() / df['strategy_return'].std()) * np.sqrt(252)
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print(f"  Sharpe Ratio: N/A (no volatility)")
    
    # Maximum drawdown
    cumulative = df['cumulative_return']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    print(f"  Maximum Drawdown: {max_drawdown:.2f}%")
    
    return df

def main():
    # Find the most recent CSV file
    result_dir = "result"
    if not os.path.exists(result_dir):
        print(f"Result directory not found: {result_dir}")
        return
    
    csv_files = [f for f in os.listdir(result_dir) if f.endswith('.csv') and 'multi_strategy_results' in f]
    
    if not csv_files:
        print("No strategy result files found")
        return
    
    # Sort by modification time and get the most recent
    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(result_dir, x)), reverse=True)
    latest_file = os.path.join(result_dir, csv_files[0])
    
    print(f"Latest strategy results file: {csv_files[0]}")
    print(f"File size: {os.path.getsize(latest_file) / (1024*1024):.1f} MB")
    print(f"Last modified: {datetime.fromtimestamp(os.path.getmtime(latest_file))}")
    print()
    
    # Analyze the performance
    try:
        df = analyze_strategy_performance(latest_file)
        print(f"\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    main() 