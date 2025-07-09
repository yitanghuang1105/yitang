"""
Multi-Strategy System Demo
Demonstrates the usage of the multi-strategy scoring system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_strategy_system.strategy_combiner import (
    run_multi_strategy_analysis, 
    get_default_params, 
    get_default_weights
)
from multi_strategy_system.performance_analyzer import analyze_strategy_performance

# 全域資料快取
_global_data_cache = None

def load_sample_data(file_path: str = None) -> pd.DataFrame:
    """
    Load sample data for demonstration - 只讀一次
    
    Args:
        file_path: Path to data file (uses default if None)
    
    Returns:
        pd.DataFrame: Sample OHLCV data
    """
    global _global_data_cache
    
    # 檢查快取
    if _global_data_cache is not None:
        print("📋 使用快取資料，避免重複讀取")
        return _global_data_cache
    
    if file_path is None:
        # 優先嘗試載入 RUN 目錄下的真實資料檔案
        possible_paths = [
            "TXF1_Minute_2020-01-01_2025-07-04.txt",  # RUN 目錄下
            "../data/TXF1_Minute_2020-01-01_2025-07-04.txt",  # data 目錄下
            "../TXF1_Minute_2020-01-01_2025-06-16.txt"  # 舊的預設路徑
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                print(f"📁 找到資料檔案: {path}")
                break
        else:
            file_path = None
    
    if file_path and os.path.exists(file_path):
        try:
            # Try to load the TXF data (CSV format, not TSV)
            df = pd.read_csv(file_path, sep=',')
            
            print(f"DEBUG: Raw data columns: {df.columns.tolist()}")
            print(f"DEBUG: Raw data shape: {df.shape}")
            print(f"DEBUG: First few rows of Date column: {df['Date'].head().tolist()}")
            print(f"DEBUG: First few rows of Time column: {df['Time'].head().tolist()}")
            
            # Rename columns to standard format (handle quoted column names)
            column_mapping = {
                'Date': 'date',
                'Time': 'time',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'TotalVolume': 'volume'  # Note: actual column is TotalVolume, not Volume
            }
            
            df = df.rename(columns=column_mapping)
            
            # Combine date and time
            print(f"DEBUG: Creating datetime from date and time columns...")
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            print(f"DEBUG: Datetime range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            df = df.set_index('datetime')
            print(f"DEBUG: Index type after set_index: {type(df.index)}")
            print(f"DEBUG: Index sample: {df.index[:5]}")
            
            # Select only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # 為了加快處理速度，只使用最近1年的資料
            print("Taking last 1 year of data for faster processing...")
            df = df.tail(525600).copy()  # 1年 = 525,600分鐘 (365天 * 24小時 * 60分鐘)
            
            print(f"Loaded {len(df)} rows of data from {file_path}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print(f"Total time period: {(df.index[-1] - df.index[0]).days} days")
            print(f"Index is monotonic: {df.index.is_monotonic_increasing}")
            print(f"Index is unique: {df.index.is_unique}")
            return df
            
        except Exception as e:
            print(f"❌ 無法載入資料檔案 {file_path}: {e}")
    
    print("🔄 生成合成資料...")
    
    # Generate synthetic data for demo (increased to 3 days of data)
    dates = pd.date_range(start='2024-01-01', periods=4320, freq='1min')  # 3 days = 4320 minutes
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.001, 4320)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, 4320)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, 4320))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, 4320))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 4320)
    }, index=dates)
    
    # 儲存到快取
    _global_data_cache = df
    
    print(f"✅ 生成合成資料: {len(df):,} 筆記錄")
    return df

def plot_results(df: pd.DataFrame, results: dict, save_path: str = None):
    """
    Plot the analysis results
    
    Args:
        df: Original data
        results: Analysis results from run_multi_strategy_analysis
        save_path: Path to save the plot (optional)
    """
    import matplotlib.dates as mdates
    from multi_strategy_system.performance_analyzer import PerformanceAnalyzer
    
    # 詳細除錯資訊
    print(f"DEBUG: df.index type: {type(df.index)}")
    print(f"DEBUG: df.index length: {len(df.index)}")
    print(f"DEBUG: df.index is monotonic: {df.index.is_monotonic_increasing}")
    print(f"DEBUG: df.index is unique: {df.index.is_unique}")
    print(f"DEBUG: df.index sample: {df.index[:5]}")
    print(f"DEBUG: df.index range: {df.index[0]} to {df.index[-1]}")
    
    # 確保 df 和 decisions 的 index 都是 DatetimeIndex 並且單調遞增
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        print("WARNING: df.index is not DatetimeIndex, converting...")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    print(f"DEBUG: After sort - df.index is monotonic: {df.index.is_monotonic_increasing}")
    print(f"DEBUG: Final df.index range: {df.index[0]} to {df.index[-1]}")
    
    decisions = results['decisions']
    if not isinstance(decisions.index, pd.DatetimeIndex):
        try:
            decisions.index = pd.to_datetime(decisions.index)
        except Exception:
            # 若無法轉換，直接用 df.index
            decisions.index = df.index
    decisions = decisions.sort_index()
    # reindex 前再次確保 index 單調遞增且型別一致
    decisions = decisions.reindex(df.index, method='ffill')

    # ===== 新增：計算權益曲線與 Buy and Hold =====
    analyzer = PerformanceAnalyzer()
    equity_curve = analyzer.calculate_equity_curve(df, decisions)
    # Buy and Hold 權益曲線
    initial_capital = analyzer.initial_capital
    buy_and_hold = initial_capital * (df['close'] / df['close'].iloc[0])

    # ===== 調整 subplot 數量 =====
    fig, axes = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
    
    # Check if reverse mode is enabled
    reverse_mode = results['params_used'].get('reverse_mode', False)
    mode_str = " (Reverse Mode)" if reverse_mode else ""
    fig.suptitle(f'Multi-Strategy Analysis Results{mode_str}', fontsize=16)
    
    # Plot 1: Price and decisions
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7)
    buy_points = df.index[decisions == 'Buy']
    sell_points = df.index[decisions == 'Sell']
    ax1.scatter(buy_points, df.loc[buy_points, 'close'], 
               color='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
    ax1.scatter(sell_points, df.loc[sell_points, 'close'], 
               color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
    ax1.set_title('Price and Trading Signals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual strategy scores
    ax2 = axes[1]
    individual_scores = results['individual_scores']
    for name, score in individual_scores.items():
        score_series = pd.Series(score)
        if not score_series.index.is_monotonic_increasing:
            score_series = score_series.sort_index()
        score_series = score_series.reindex(df.index, method='ffill')
        ax2.plot(df.index, score_series, label=name.replace('_', ' ').title(), alpha=0.8)
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
    ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
    ax2.set_title('Individual Strategy Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Plot 3: Combined score
    ax3 = axes[2]
    combined_score = pd.Series(results['combined_score'])
    combined_score = combined_score.reindex(df.index, method='ffill')
    ax3.plot(df.index, combined_score, label='Combined Score', linewidth=2, color='purple')
    ax3.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
    ax3.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
    ax3.fill_between(df.index, 70, 100, alpha=0.2, color='green', label='Buy Zone')
    ax3.fill_between(df.index, 0, 30, alpha=0.2, color='red', label='Sell Zone')
    ax3.set_title('Combined Strategy Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Plot 4: 權益曲線比較
    ax4 = axes[3]
    ax4.plot(equity_curve.index, equity_curve, label='Strategy Equity Curve', color='blue', linewidth=2)
    ax4.plot(buy_and_hold.index, buy_and_hold, label='Buy & Hold Equity Curve', color='orange', linestyle='--', linewidth=2)
    ax4.set_title('Equity Curve Comparison')
    ax4.set_ylabel('Equity ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Volume
    ax5 = axes[4]
    ax5.bar(df.index, df['volume'], alpha=0.6, color='blue', label='Volume')
    ax5.set_title('Volume')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # x 軸標籤優化
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def print_analysis_summary(results: dict):
    """
    Print a summary of the analysis results
    
    Args:
        results: Analysis results from run_multi_strategy_analysis
    """
    print("\n" + "="*60)
    print("MULTI-STRATEGY ANALYSIS SUMMARY")
    print("="*60)
    
    # Print parameters used
    print("\nParameters Used:")
    params = results['params_used']
    for key, value in params.items():
        if key == 'reverse_mode':
            mode_str = "Reverse Mode" if value else "Normal Mode"
            print(f"  {key}: {mode_str}")
        else:
            print(f"  {key}: {value}")
    
    # Print weights used
    print("\nStrategy Weights:")
    weights = results['weights_used']
    for strategy, weight in weights.items():
        print(f"  {strategy.replace('_', ' ').title()}: {weight:.2%}")
    
    # Print decision statistics
    decisions = results['decisions']
    decision_counts = decisions.value_counts()
    
    print("\nDecision Statistics:")
    for decision, count in decision_counts.items():
        percentage = (count / len(decisions)) * 100
        print(f"  {decision}: {count} ({percentage:.1f}%)")
    
    # Print score statistics
    combined_score = results['combined_score']
    print(f"\nCombined Score Statistics:")
    print(f"  Mean: {combined_score.mean():.2f}")
    print(f"  Std: {combined_score.std():.2f}")
    print(f"  Min: {combined_score.min():.2f}")
    print(f"  Max: {combined_score.max():.2f}")
    
    # Print individual strategy statistics
    print(f"\nIndividual Strategy Statistics:")
    individual_scores = results['individual_scores']
    for name, score in individual_scores.items():
        print(f"  {name.replace('_', ' ').title()}:")
        print(f"    Mean: {score.mean():.2f}")
        print(f"    Std: {score.std():.2f}")

def main():
    """
    Main demonstration function
    """
    print("Multi-Strategy Scoring System Demo")
    print("="*40)
    
    # Load data
    df = load_sample_data()
    
    # Get default parameters and weights
    params = get_default_params()
    weights = get_default_weights()
    
    # Check for command line arguments (for parameter file)
    if len(sys.argv) > 2 and sys.argv[1] == "--params":
        param_file = sys.argv[2]
        if os.path.exists(param_file):
            try:
                import json
                with open(param_file, 'r', encoding='utf-8') as f:
                    loaded_params = json.load(f)
                params.update(loaded_params)
                print(f"Loaded parameters from {param_file}")
                
                # Check if reverse mode is enabled
                if params.get('reverse_mode', False):
                    print("🔄 Reverse Mode: ENABLED - Trading against strategy signals")
                else:
                    print("📈 Normal Mode: Trading with strategy signals")
                    
            except Exception as e:
                print(f"Warning: Failed to load parameters from {param_file}: {e}")
    
    print(f"\nData loaded: {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Run analysis
    print("\nRunning multi-strategy analysis...")
    results = run_multi_strategy_analysis(df, params, weights)
    
    # Print summary
    print_analysis_summary(results)
    
    # Create plots
    print("\nGenerating plots...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"multi_strategy_analysis_{timestamp}.png"
    plot_results(df, results, plot_path)
    
    # Performance analysis
    print("\nPerforming performance analysis...")
    performance_metrics = analyze_strategy_performance(df, results['decisions'])
    
    # Create performance analysis plot
    performance_plot_path = f"performance_analysis_{timestamp}.png"
    from multi_strategy_system.performance_analyzer import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer()
    analyzer.plot_performance_analysis(df, performance_metrics, performance_plot_path)
    
    # Save results to CSV
    print("\nSaving results to CSV...")
    results_df = df.copy()
    
    # Add individual scores
    for name, score in results['individual_scores'].items():
        results_df[f'{name}_score'] = score
    
    # Add combined score and decisions
    results_df['combined_score'] = results['combined_score']
    results_df['decision'] = results['decisions']
    
    csv_path = f"multi_strategy_results_{timestamp}.csv"
    results_df.to_csv(csv_path)
    print(f"Results saved to {csv_path}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 