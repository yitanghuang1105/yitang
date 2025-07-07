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

def load_sample_data(file_path: str = None) -> pd.DataFrame:
    """
    Load sample data for demonstration
    
    Args:
        file_path: Path to data file (uses default if None)
    
    Returns:
        pd.DataFrame: Sample OHLCV data
    """
    if file_path is None:
        # Use the TXF data file in RUN directory
        file_path = "../TXF1_Minute_2020-01-01_2025-06-16.txt"
    
    try:
        # Try to load the TXF data
        df = pd.read_csv(file_path, sep='\t')
        
        # Rename columns to standard format
        column_mapping = {
            'Date': 'date',
            'Time': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Combine date and time
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.set_index('datetime')
        
        # Select only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Take last 1000 rows for demo
        df = df.tail(1000)
        
        print(f"Loaded {len(df)} rows of data from {file_path}")
        return df
        
    except Exception as e:
        print(f"Warning: Could not load data from {file_path}: {e}")
        print("Generating synthetic data for demo...")
        
        # Generate synthetic data for demo
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0, 0.001, 1000)
        prices = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0005, 1000)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, 1000))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, 1000))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        print(f"Generated {len(df)} rows of synthetic data")
        return df

def plot_results(df: pd.DataFrame, results: dict, save_path: str = None):
    """
    Plot the analysis results
    
    Args:
        df: Original data
        results: Analysis results from run_multi_strategy_analysis
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle('Multi-Strategy Analysis Results', fontsize=16)
    
    # Plot 1: Price and decisions
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', alpha=0.7)
    
    # Add decision markers
    decisions = results['decisions']
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
        ax2.plot(df.index, score, label=name.replace('_', ' ').title(), alpha=0.8)
    
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
    ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
    ax2.set_title('Individual Strategy Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Plot 3: Combined score
    ax3 = axes[2]
    combined_score = results['combined_score']
    ax3.plot(df.index, combined_score, label='Combined Score', linewidth=2, color='purple')
    ax3.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
    ax3.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
    ax3.fill_between(df.index, 70, 100, alpha=0.2, color='green', label='Buy Zone')
    ax3.fill_between(df.index, 0, 30, alpha=0.2, color='red', label='Sell Zone')
    ax3.set_title('Combined Strategy Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Plot 4: Volume
    ax4 = axes[3]
    ax4.bar(df.index, df['volume'], alpha=0.6, color='blue', label='Volume')
    ax4.set_title('Volume')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
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