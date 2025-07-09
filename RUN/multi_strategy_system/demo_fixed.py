"""
Multi-Strategy Scoring System Demo
Demonstrates the multi-strategy analysis system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from strategy_combiner import run_multi_strategy_analysis, get_default_params, get_default_weights
from performance_analyzer import PerformanceAnalyzer

def load_sample_data(file_path: str = None) -> pd.DataFrame:
    """
    Load sample data for demonstration
    
    Args:
        file_path: Path to data file (uses default if None)
    
    Returns:
        pd.DataFrame: Sample OHLCV data
    """
    if file_path is None:
        # Create synthetic data for demonstration
        dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        base_price = 17000
        price_changes = np.random.normal(0, 0.001, len(dates))  # 0.1% volatility
        prices = base_price * np.exp(np.cumsum(price_changes))
        
        # Add some trend and cycles
        trend = np.linspace(0, 0.1, len(dates))  # 10% upward trend
        cycle = 0.02 * np.sin(2 * np.pi * np.arange(len(dates)) / 100)  # Daily cycle
        
        prices = prices * (1 + trend + cycle)
        
        # Generate OHLCV data
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(dates)))),
            'close': prices,
            'volume': np.random.randint(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    else:
        # Load from file
        try:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return load_sample_data()  # Fall back to synthetic data

def analyze_strategy_performance(df: pd.DataFrame, decisions: pd.Series) -> dict:
    """
    Analyze strategy performance
    
    Args:
        df: DataFrame with OHLCV data
        decisions: Trading decisions
    
    Returns:
        dict: Performance metrics
    """
    analyzer = PerformanceAnalyzer()
    performance_metrics = analyzer.calculate_performance_metrics(df, decisions)
    return performance_metrics

def plot_results(df: pd.DataFrame, results: dict, save_path: str = None):
    """
    Create comprehensive plots of the analysis results
    
    Args:
        df: DataFrame with OHLCV data
        results: Analysis results from run_multi_strategy_analysis
        save_path: Path to save the plot (optional)
    """
    # Calculate equity curves for comparison
    analyzer = PerformanceAnalyzer()
    equity_curve = analyzer.calculate_equity_curve(df, results['decisions'])
    
    # Calculate buy and hold equity curve
    buy_and_hold = df['close'] / df['close'].iloc[0] * 100000
    
    # Find buy and sell points
    buy_points = results['decisions'][results['decisions'] == 'Buy'].index
    sell_points = results['decisions'][results['decisions'] == 'Sell'].index
    
    # Create subplots
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    fig.suptitle('Multi-Strategy Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Price and signals
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Price', linewidth=1, alpha=0.8)
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
    
    # Plot 4: Equity curve comparison
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

    # x-axis label optimization
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