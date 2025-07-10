"""
Position Sizing Calculator
Provides various methods to determine appropriate position sizes for trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PositionSizingConfig:
    """Configuration for position sizing calculations"""
    # Capital management
    total_capital: float = 100000  # Total available capital
    max_risk_per_trade: float = 0.02  # Maximum risk per trade (2%)
    max_position_size: float = 0.1  # Maximum position size (10% of capital)
    
    # Risk management
    stop_loss_pct: float = 0.02  # Stop loss percentage
    volatility_lookback: int = 20  # Days for volatility calculation
    
    # Market conditions
    min_volume_threshold: float = 1000  # Minimum volume for position sizing
    max_slippage_pct: float = 0.001  # Maximum expected slippage (0.1%)
    
    # Strategy confidence
    confidence_threshold: float = 0.7  # Minimum confidence for full position
    min_position_size: float = 0.01  # Minimum position size (1% of capital)

class PositionSizingCalculator:
    """
    Comprehensive position sizing calculator with multiple methods
    """
    
    def __init__(self, config: PositionSizingConfig = None):
        """
        Initialize position sizing calculator
        
        Args:
            config: Position sizing configuration
        """
        self.config = config or PositionSizingConfig()
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion position size
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
        
        Returns:
            float: Optimal position size as fraction of capital
        """
        if avg_loss == 0:
            return 0.0
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply constraints
        kelly_fraction = max(0, min(kelly_fraction, self.config.max_position_size))
        
        return kelly_fraction
    
    def calculate_volatility_based_size(self, price_series: pd.Series, 
                                      confidence_score: float = 0.5) -> float:
        """
        Calculate position size based on price volatility
        
        Args:
            price_series: Historical price data
            confidence_score: Strategy confidence (0-1)
        
        Returns:
            float: Position size as fraction of capital
        """
        # Calculate volatility
        returns = price_series.pct_change().dropna()
        volatility = returns.rolling(self.config.volatility_lookback).std().iloc[-1]
        
        if volatility == 0:
            return self.config.min_position_size
        
        # Inverse relationship: higher volatility = smaller position
        volatility_factor = 1 / (volatility * 100)  # Scale volatility
        
        # Apply confidence adjustment
        confidence_factor = confidence_score
        
        # Calculate position size
        position_size = volatility_factor * confidence_factor * self.config.max_position_size
        
        # Apply constraints
        position_size = max(self.config.min_position_size, 
                           min(position_size, self.config.max_position_size))
        
        return position_size
    
    def calculate_risk_based_size(self, entry_price: float, stop_loss_price: float,
                                confidence_score: float = 0.5) -> float:
        """
        Calculate position size based on risk per trade
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            confidence_score: Strategy confidence (0-1)
        
        Returns:
            float: Position size as fraction of capital
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return self.config.min_position_size
        
        # Calculate maximum dollar risk
        max_dollar_risk = self.config.total_capital * self.config.max_risk_per_trade
        
        # Calculate position size in shares
        max_shares = max_dollar_risk / risk_per_share
        
        # Calculate position value
        position_value = max_shares * entry_price
        
        # Convert to fraction of capital
        position_size = position_value / self.config.total_capital
        
        # Apply confidence adjustment
        position_size *= confidence_score
        
        # Apply constraints
        position_size = max(self.config.min_position_size, 
                           min(position_size, self.config.max_position_size))
        
        return position_size
    
    def calculate_volume_based_size(self, volume: float, avg_volume: float,
                                  confidence_score: float = 0.5) -> float:
        """
        Calculate position size based on volume conditions
        
        Args:
            volume: Current volume
            avg_volume: Average volume
            confidence_score: Strategy confidence (0-1)
        
        Returns:
            float: Position size as fraction of capital
        """
        if avg_volume == 0:
            return self.config.min_position_size
        
        # Volume ratio
        volume_ratio = volume / avg_volume
        
        # Volume factor (higher volume = larger position)
        volume_factor = min(volume_ratio, 3.0) / 3.0  # Cap at 3x average
        
        # Apply confidence adjustment
        confidence_factor = confidence_score
        
        # Calculate position size
        position_size = volume_factor * confidence_factor * self.config.max_position_size
        
        # Apply constraints
        position_size = max(self.config.min_position_size, 
                           min(position_size, self.config.max_position_size))
        
        return position_size
    
    def calculate_combined_position_size(self, 
                                       price_data: pd.DataFrame,
                                       strategy_score: float,
                                       entry_price: float,
                                       stop_loss_price: float = None,
                                       volume: float = None) -> Dict[str, float]:
        """
        Calculate position size using multiple methods and combine them
        
        Args:
            price_data: DataFrame with OHLCV data
            strategy_score: Strategy confidence score (0-100)
            entry_price: Entry price
            stop_loss_price: Stop loss price (optional)
            volume: Current volume (optional)
        
        Returns:
            Dict: Position sizing results from different methods
        """
        # Convert strategy score to confidence (0-1)
        confidence_score = strategy_score / 100.0
        
        results = {}
        
        # 1. Volatility-based sizing
        if len(price_data) >= self.config.volatility_lookback:
            results['volatility_based'] = self.calculate_volatility_based_size(
                price_data['close'], confidence_score
            )
        
        # 2. Risk-based sizing
        if stop_loss_price is not None:
            results['risk_based'] = self.calculate_risk_based_size(
                entry_price, stop_loss_price, confidence_score
            )
        
        # 3. Volume-based sizing
        if volume is not None and 'volume' in price_data.columns:
            avg_volume = price_data['volume'].rolling(20).mean().iloc[-1]
            results['volume_based'] = self.calculate_volume_based_size(
                volume, avg_volume, confidence_score
            )
        
        # 4. Kelly Criterion (if we have historical performance data)
        # This would require historical win rate and average win/loss data
        
        # 5. Simple confidence-based sizing
        results['confidence_based'] = confidence_score * self.config.max_position_size
        
        # Calculate weighted average of available methods
        available_methods = list(results.keys())
        if available_methods:
            weights = {
                'volatility_based': 0.3,
                'risk_based': 0.4,
                'volume_based': 0.2,
                'confidence_based': 0.1
            }
            
            weighted_sum = 0
            total_weight = 0
            
            for method in available_methods:
                weight = weights.get(method, 0.1)
                weighted_sum += results[method] * weight
                total_weight += weight
            
            results['combined'] = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            results['combined'] = self.config.min_position_size
        
        return results
    
    def calculate_quantity(self, position_size_pct: float, 
                          entry_price: float, 
                          available_capital: float = None) -> int:
        """
        Calculate actual quantity to buy/sell
        
        Args:
            position_size_pct: Position size as percentage of capital
            entry_price: Entry price
            available_capital: Available capital (uses config if None)
        
        Returns:
            int: Number of shares/contracts to trade
        """
        if available_capital is None:
            available_capital = self.config.total_capital
        
        # Calculate position value
        position_value = available_capital * position_size_pct
        
        # Calculate initial quantity
        quantity = int(position_value / entry_price)
        
        # 僅允許在 available_capital 足夠時下單，否則最大只買得起的數量
        max_affordable_quantity = int(available_capital // entry_price)
        quantity = min(quantity, max_affordable_quantity)
        
        return max(1, quantity)  # Minimum 1 unit
    
    def get_position_recommendation(self, 
                                  price_data: pd.DataFrame,
                                  strategy_score: float,
                                  entry_price: float,
                                  signal_type: str = 'buy',
                                  stop_loss_pct: float = None) -> Dict:
        """
        Get complete position sizing recommendation
        
        Args:
            price_data: DataFrame with OHLCV data
            strategy_score: Strategy confidence score (0-100)
            entry_price: Entry price
            signal_type: 'buy' or 'sell'
            stop_loss_pct: Stop loss percentage (uses config if None)
        
        Returns:
            Dict: Complete position sizing recommendation
        """
        if stop_loss_pct is None:
            stop_loss_pct = self.config.stop_loss_pct
        
        # Calculate stop loss price
        if signal_type.lower() == 'buy':
            stop_loss_price = entry_price * (1 - stop_loss_pct)
        else:  # sell
            stop_loss_price = entry_price * (1 + stop_loss_pct)
        
        # Get current volume
        current_volume = price_data['volume'].iloc[-1] if 'volume' in price_data.columns else None
        
        # Calculate position sizes
        position_sizes = self.calculate_combined_position_size(
            price_data, strategy_score, entry_price, stop_loss_price, current_volume
        )
        
        # Get recommended position size
        recommended_size = position_sizes['combined']
        
        # Calculate quantity
        quantity = self.calculate_quantity(recommended_size, entry_price)
        
        # Calculate position value
        position_value = quantity * entry_price
        
        # Calculate risk metrics
        risk_per_share = abs(entry_price - stop_loss_price)
        total_risk = risk_per_share * quantity
        risk_pct = total_risk / self.config.total_capital
        
        recommendation = {
            'signal_type': signal_type,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'strategy_score': strategy_score,
            'confidence_score': strategy_score / 100.0,
            'position_size_pct': recommended_size * 100,
            'quantity': quantity,
            'position_value': position_value,
            'risk_per_share': risk_per_share,
            'total_risk': total_risk,
            'risk_pct': risk_pct * 100,
            'position_sizes': position_sizes,
            'recommendation': self._get_recommendation_text(recommended_size, risk_pct)
        }
        
        return recommendation
    
    def _get_recommendation_text(self, position_size: float, risk_pct: float) -> str:
        """Generate recommendation text"""
        if position_size < self.config.min_position_size:
            return "SKIP TRADE - Position size too small"
        elif risk_pct > self.config.max_risk_per_trade:
            return "REDUCE POSITION - Risk too high"
        elif position_size > self.config.max_position_size * 0.8:
            return "FULL POSITION - High confidence signal"
        else:
            return "MODERATE POSITION - Standard sizing"

def demo_position_sizing():
    """Demonstrate position sizing calculations"""
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    price_data = pd.DataFrame({
        'open': np.random.normal(100, 2, 100),
        'high': np.random.normal(102, 2, 100),
        'low': np.random.normal(98, 2, 100),
        'close': np.random.normal(100, 2, 100),
        'volume': np.random.normal(10000, 2000, 100)
    }, index=dates)
    
    # Initialize calculator
    config = PositionSizingConfig(
        total_capital=100000,
        max_risk_per_trade=0.02,
        max_position_size=0.1
    )
    calculator = PositionSizingCalculator(config)
    
    # Example calculations
    print("Position Sizing Calculator Demo")
    print("=" * 50)
    
    # Example 1: High confidence buy signal
    print("\n1. High Confidence Buy Signal (Score: 85)")
    recommendation1 = calculator.get_position_recommendation(
        price_data, 85, 100, 'buy', 0.02
    )
    print(f"   Position Size: {recommendation1['position_size_pct']:.1f}%")
    print(f"   Quantity: {recommendation1['quantity']}")
    print(f"   Position Value: ${recommendation1['position_value']:,.0f}")
    print(f"   Risk: {recommendation1['risk_pct']:.2f}%")
    print(f"   Recommendation: {recommendation1['recommendation']}")
    
    # Example 2: Low confidence sell signal
    print("\n2. Low Confidence Sell Signal (Score: 35)")
    recommendation2 = calculator.get_position_recommendation(
        price_data, 35, 100, 'sell', 0.02
    )
    print(f"   Position Size: {recommendation2['position_size_pct']:.1f}%")
    print(f"   Quantity: {recommendation2['quantity']}")
    print(f"   Position Value: ${recommendation2['position_value']:,.0f}")
    print(f"   Risk: {recommendation2['risk_pct']:.2f}%")
    print(f"   Recommendation: {recommendation2['recommendation']}")
    
    # Example 3: Kelly Criterion
    print("\n3. Kelly Criterion Example")
    kelly_size = calculator.calculate_kelly_criterion(0.6, 0.03, 0.02)
    print(f"   Win Rate: 60%")
    print(f"   Avg Win: 3%")
    print(f"   Avg Loss: 2%")
    print(f"   Kelly Position Size: {kelly_size*100:.1f}%")

if __name__ == "__main__":
    demo_position_sizing() 