"""
Strategy Combiner Module
Combines multiple strategy scores into weighted decisions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .strategies import rsi_score, bb_score, obv_score

def aggregate_scores(score_list: List[pd.Series], weights: List[float]) -> pd.Series:
    """
    Aggregate multiple strategy scores using weighted average
    
    Args:
        score_list: List of strategy scores (0-100)
        weights: List of weights for each strategy
    
    Returns:
        pd.Series: Weighted average score (0-100)
    """
    if len(score_list) != len(weights):
        raise ValueError("Number of scores must match number of weights")
    
    if len(score_list) == 0:
        raise ValueError("At least one score must be provided")
    
    # Calculate weighted sum
    weighted_scores = [score * weight for score, weight in zip(score_list, weights)]
    total_score = sum(weighted_scores) / sum(weights)
    
    return total_score

def decision_from_score(score: pd.Series, buy_threshold: float = 70, sell_threshold: float = 30, reverse_mode: bool = False) -> pd.Series:
    """
    Convert score to trading decision
    
    Args:
        score: Score series (0-100)
        buy_threshold: Score threshold for buy signal (default: 70)
        sell_threshold: Score threshold for sell signal (default: 30)
        reverse_mode: If True, reverse the buy/sell signals
    
    Returns:
        pd.Series: Decision series ('Buy', 'Sell', 'Hold')
    """
    def _decision(x):
        if pd.isna(x):
            return 'Hold'
        elif x >= buy_threshold:
            return 'Sell' if reverse_mode else 'Buy'
        elif x <= sell_threshold:
            return 'Buy' if reverse_mode else 'Sell'
        else:
            return 'Hold'
    
    return score.apply(_decision)

def compute_all_strategies(df: pd.DataFrame, params: Dict) -> Dict[str, pd.Series]:
    """
    Compute scores for all available strategies
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary containing parameters for all strategies
    
    Returns:
        Dict: Dictionary with strategy names as keys and scores as values
    """
    strategies = {
        'rsi': rsi_score,
        'bollinger_bands': bb_score,
        'obv': obv_score
    }
    
    results = {}
    for name, strategy_func in strategies.items():
        try:
            results[name] = strategy_func(df, params)
        except Exception as e:
            print(f"Warning: Failed to compute {name} strategy: {e}")
            # Return neutral score on error
            results[name] = pd.Series(50, index=df.index)
    
    return results

def get_default_weights() -> Dict[str, float]:
    """
    Get default weights for strategies
    
    Returns:
        Dict: Default weights for each strategy
    """
    return {
        'rsi': 0.4,
        'bollinger_bands': 0.35,
        'obv': 0.25
    }

def get_default_params() -> Dict:
    """
    Get default parameters for all strategies
    
    Returns:
        Dict: Default parameters for all strategies
    """
    return {
        # RSI parameters
        'rsi_window': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        
        # Bollinger Bands parameters
        'bb_window': 20,
        'bb_std': 2.0,
        
        # OBV parameters
        'obv_window': 10,
        'obv_threshold': 1.2,
        
        # Decision thresholds
        'buy_threshold': 70,
        'sell_threshold': 30,
        
        # Reverse mode
        'reverse_mode': False
    }

def run_multi_strategy_analysis(df: pd.DataFrame, 
                               params: Dict = None, 
                               weights: Dict[str, float] = None) -> Dict[str, pd.Series]:
    """
    Run complete multi-strategy analysis
    
    Args:
        df: DataFrame with OHLCV data
        params: Strategy parameters (uses defaults if None)
        weights: Strategy weights (uses defaults if None)
    
    Returns:
        Dict: Complete analysis results including individual scores, combined score, and decisions
    """
    # Use defaults if not provided
    if params is None:
        params = get_default_params()
    if weights is None:
        weights = get_default_weights()
    
    # Compute individual strategy scores
    strategy_scores = compute_all_strategies(df, params)
    
    # Prepare for aggregation
    score_list = list(strategy_scores.values())
    weight_list = [weights.get(name, 0.0) for name in strategy_scores.keys()]
    
    # Aggregate scores
    combined_score = aggregate_scores(score_list, weight_list)
    
    # Generate decisions
    buy_threshold = params.get('buy_threshold', 70)
    sell_threshold = params.get('sell_threshold', 30)
    reverse_mode = params.get('reverse_mode', False)
    decisions = decision_from_score(combined_score, buy_threshold, sell_threshold, reverse_mode)
    
    # Return complete results
    results = {
        'individual_scores': strategy_scores,
        'combined_score': combined_score,
        'decisions': decisions,
        'weights_used': weights,
        'params_used': params
    }
    
    return results 