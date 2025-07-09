#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify equity curve initial capital
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the RUN directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'RUN'))

from multi_strategy_system.performance_analyzer import PerformanceAnalyzer

def test_equity_curve():
    """Test equity curve calculation with different initial capitals"""
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    prices = np.linspace(100, 110, 100) + np.random.normal(0, 0.5, 100)
    
    df = pd.DataFrame({
        'close': prices,
        'open': prices + np.random.normal(0, 0.1, 100),
        'high': prices + np.abs(np.random.normal(0, 0.2, 100)),
        'low': prices - np.abs(np.random.normal(0, 0.2, 100)),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Create sample decisions
    decisions = pd.Series(['Hold'] * len(df), index=df.index)
    decisions.iloc[10] = 'Buy'
    decisions.iloc[50] = 'Sell'
    decisions.iloc[80] = 'Buy'
    
    print("Testing Equity Curve Initial Capital")
    print("=" * 50)
    
    # Test with different initial capitals
    test_capitals = [1000, 10000, 100000]
    
    for initial_capital in test_capitals:
        print(f"\nTesting with initial capital: ${initial_capital:,}")
        
        # Create analyzer
        analyzer = PerformanceAnalyzer(initial_capital=initial_capital)
        
        # Calculate equity curve
        equity_curve = analyzer.calculate_equity_curve(df, decisions)
        
        # Check initial value
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        
        print(f"  Initial equity value: ${initial_value:,.2f}")
        print(f"  Final equity value: ${final_value:,.2f}")
        print(f"  Expected initial value: ${initial_capital:,.2f}")
        print(f"  Match: {'✅' if abs(initial_value - initial_capital) < 0.01 else '❌'}")
        
        # Check if all values are reasonable
        min_value = equity_curve.min()
        max_value = equity_curve.max()
        print(f"  Min value: ${min_value:,.2f}")
        print(f"  Max value: ${max_value:,.2f}")
        
        # Check for any NaN or inf values
        if equity_curve.isna().any() or np.isinf(equity_curve).any():
            print(f"  ⚠️  WARNING: Found NaN or inf values in equity curve!")
            print(f"  ⚠️  Check for division by zero or other calculation errors")

if __name__ == "__main__":
    test_equity_curve() 