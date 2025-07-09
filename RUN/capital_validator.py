"""
Capital Validation System
Prevents over-leveraging and insufficient funds errors
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

class ValidationResult(Enum):
    """Validation result types"""
    VALID = "VALID"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    OVER_LEVERAGE = "OVER_LEVERAGE"
    EXCESSIVE_RISK = "EXCESSIVE_RISK"
    INVALID_QUANTITY = "INVALID_QUANTITY"

@dataclass
class CapitalConfig:
    """Capital management configuration"""
    total_capital: float = 100000  # Total available capital
    available_capital: float = 100000  # Currently available capital
    max_leverage: float = 1.0  # Maximum leverage (1.0 = no leverage)
    max_position_value: float = 0.1  # Maximum position value as % of capital
    max_risk_per_trade: float = 0.02  # Maximum risk per trade (2%)
    min_capital_buffer: float = 0.1  # Minimum capital buffer (10%)
    
    # Transaction costs
    commission_rate: float = 0.001  # Commission rate (0.1%)
    slippage_rate: float = 0.0005  # Slippage rate (0.05%)
    
    # Position limits
    max_positions: int = 10  # Maximum number of concurrent positions
    max_correlation_exposure: float = 0.3  # Maximum exposure to correlated assets

class CapitalValidator:
    """
    Comprehensive capital validation system
    """
    
    def __init__(self, config: CapitalConfig = None):
        """
        Initialize capital validator
        
        Args:
            config: Capital configuration
        """
        self.config = config or CapitalConfig()
        self.positions = []  # Track current positions
        self.transaction_history = []  # Track transaction history
    
    def validate_trade(self, 
                      entry_price: float,
                      quantity: int,
                      signal_type: str = 'buy',
                      asset_name: str = 'unknown',
                      strategy_score: float = 50.0) -> Dict:
        """
        Validate a potential trade
        
        Args:
            entry_price: Entry price
            quantity: Number of shares/contracts
            signal_type: 'buy' or 'sell'
            asset_name: Name of the asset
            strategy_score: Strategy confidence score (0-100)
        
        Returns:
            Dict: Validation result with details
        """
        # Calculate position value
        position_value = entry_price * quantity
        
        # Calculate transaction costs
        commission = position_value * self.config.commission_rate
        slippage = position_value * self.config.slippage_rate
        total_cost = position_value + commission + slippage
        
        # Basic validation checks
        validation_checks = []
        
        # 1. Check if we have enough capital
        if signal_type.lower() == 'buy':
            if total_cost > self.config.available_capital:
                return {
                    'valid': False,
                    'result': ValidationResult.INSUFFICIENT_FUNDS,
                    'message': f"Insufficient funds: Need ${total_cost:,.2f}, have ${self.config.available_capital:,.2f}",
                    'required_capital': total_cost,
                    'available_capital': self.config.available_capital,
                    'shortfall': total_cost - self.config.available_capital
                }
        
        # 2. Check leverage limits
        leverage = position_value / self.config.total_capital
        if leverage > self.config.max_leverage:
            return {
                'valid': False,
                'result': ValidationResult.OVER_LEVERAGE,
                'message': f"Over-leverage: {leverage:.1%} leverage exceeds {self.config.max_leverage:.1%} limit",
                'current_leverage': leverage,
                'max_leverage': self.config.max_leverage
            }
        
        # 3. Check position size limits
        position_size_pct = position_value / self.config.total_capital
        if position_size_pct > self.config.max_position_value:
            return {
                'valid': False,
                'result': ValidationResult.EXCESSIVE_RISK,
                'message': f"Position too large: {position_size_pct:.1%} exceeds {self.config.max_position_value:.1%} limit",
                'position_size_pct': position_size_pct,
                'max_position_size_pct': self.config.max_position_value
            }
        
        # 4. Check quantity validity
        if quantity <= 0:
            return {
                'valid': False,
                'result': ValidationResult.INVALID_QUANTITY,
                'message': f"Invalid quantity: {quantity} must be positive",
                'quantity': quantity
            }
        
        # 5. Check capital buffer
        remaining_capital = self.config.available_capital - total_cost
        buffer_pct = remaining_capital / self.config.total_capital
        if buffer_pct < self.config.min_capital_buffer:
            return {
                'valid': False,
                'result': ValidationResult.EXCESSIVE_RISK,
                'message': f"Insufficient capital buffer: {buffer_pct:.1%} remaining, need {self.config.min_capital_buffer:.1%}",
                'remaining_capital': remaining_capital,
                'buffer_pct': buffer_pct,
                'min_buffer_pct': self.config.min_capital_buffer
            }
        
        # 6. Check number of positions
        if len(self.positions) >= self.config.max_positions:
            return {
                'valid': False,
                'result': ValidationResult.EXCESSIVE_RISK,
                'message': f"Too many positions: {len(self.positions)} positions, max {self.config.max_positions}",
                'current_positions': len(self.positions),
                'max_positions': self.config.max_positions
            }
        
        # All checks passed
        return {
            'valid': True,
            'result': ValidationResult.VALID,
            'message': "Trade validation passed",
            'position_value': position_value,
            'total_cost': total_cost,
            'commission': commission,
            'slippage': slippage,
            'leverage': leverage,
            'position_size_pct': position_size_pct,
            'remaining_capital': remaining_capital,
            'buffer_pct': buffer_pct
        }
    
    def calculate_max_quantity(self, 
                             entry_price: float,
                             signal_type: str = 'buy',
                             strategy_score: float = 50.0) -> Dict:
        """
        Calculate maximum safe quantity for a trade
        
        Args:
            entry_price: Entry price
            signal_type: 'buy' or 'sell'
            strategy_score: Strategy confidence score (0-100)
        
        Returns:
            Dict: Maximum quantity calculation with details
        """
        # Adjust position size based on strategy score
        confidence_factor = strategy_score / 100.0
        max_position_pct = self.config.max_position_value * confidence_factor
        
        # Calculate maximum position value
        max_position_value = self.config.total_capital * max_position_pct
        
        # Calculate maximum quantity based on position value
        max_quantity_by_position = int(max_position_value / entry_price)
        
        # Calculate maximum quantity based on available capital
        if signal_type.lower() == 'buy':
            # Account for transaction costs
            cost_per_share = entry_price * (1 + self.config.commission_rate + self.config.slippage_rate)
            max_quantity_by_capital = int(self.config.available_capital / cost_per_share)
        else:
            # For sell orders, we can sell what we have
            max_quantity_by_capital = max_quantity_by_position
        
        # Take the minimum of the two limits
        max_quantity = min(max_quantity_by_position, max_quantity_by_capital)
        
        # Ensure minimum quantity
        max_quantity = max(1, max_quantity)
        
        # Calculate actual position value and costs
        actual_position_value = max_quantity * entry_price
        commission = actual_position_value * self.config.commission_rate
        slippage = actual_position_value * self.config.slippage_rate
        total_cost = actual_position_value + commission + slippage
        
        return {
            'max_quantity': max_quantity,
            'max_quantity_by_position': max_quantity_by_position,
            'max_quantity_by_capital': max_quantity_by_capital,
            'position_value': actual_position_value,
            'total_cost': total_cost,
            'commission': commission,
            'slippage': slippage,
            'leverage': actual_position_value / self.config.total_capital,
            'position_size_pct': actual_position_value / self.config.total_capital,
            'remaining_capital': self.config.available_capital - total_cost,
            'confidence_factor': confidence_factor
        }
    
    def execute_trade(self, 
                     entry_price: float,
                     quantity: int,
                     signal_type: str = 'buy',
                     asset_name: str = 'unknown',
                     strategy_score: float = 50.0) -> Dict:
        """
        Execute a validated trade
        
        Args:
            entry_price: Entry price
            quantity: Number of shares/contracts
            signal_type: 'buy' or 'sell'
            asset_name: Name of the asset
            strategy_score: Strategy confidence score (0-100)
        
        Returns:
            Dict: Trade execution result
        """
        # Validate trade first
        validation = self.validate_trade(entry_price, quantity, signal_type, asset_name, strategy_score)
        
        if not validation['valid']:
            return {
                'success': False,
                'error': validation['message'],
                'validation': validation
            }
        
        # Calculate costs
        position_value = entry_price * quantity
        commission = position_value * self.config.commission_rate
        slippage = position_value * self.config.slippage_rate
        total_cost = position_value + commission + slippage
        
        # Update capital
        if signal_type.lower() == 'buy':
            self.config.available_capital -= total_cost
        else:  # sell
            self.config.available_capital += (position_value - commission - slippage)
        
        # Record position
        position = {
            'asset_name': asset_name,
            'entry_price': entry_price,
            'quantity': quantity,
            'signal_type': signal_type,
            'position_value': position_value,
            'total_cost': total_cost,
            'commission': commission,
            'slippage': slippage,
            'strategy_score': strategy_score,
            'timestamp': pd.Timestamp.now()
        }
        
        self.positions.append(position)
        
        # Record transaction
        transaction = {
            'timestamp': pd.Timestamp.now(),
            'asset_name': asset_name,
            'signal_type': signal_type,
            'entry_price': entry_price,
            'quantity': quantity,
            'position_value': position_value,
            'total_cost': total_cost,
            'available_capital_before': self.config.available_capital + total_cost if signal_type.lower() == 'buy' else self.config.available_capital - (position_value - commission - slippage),
            'available_capital_after': self.config.available_capital,
            'strategy_score': strategy_score
        }
        
        self.transaction_history.append(transaction)
        
        return {
            'success': True,
            'message': f"Trade executed successfully: {signal_type} {quantity} {asset_name} at ${entry_price:.2f}",
            'position': position,
            'transaction': transaction,
            'remaining_capital': self.config.available_capital
        }
    
    def get_capital_summary(self) -> Dict:
        """
        Get current capital summary
        
        Returns:
            Dict: Capital summary information
        """
        total_position_value = sum(pos['position_value'] for pos in self.positions)
        total_commission = sum(pos['commission'] for pos in self.positions)
        total_slippage = sum(pos['slippage'] for pos in self.positions)
        
        return {
            'total_capital': self.config.total_capital,
            'available_capital': self.config.available_capital,
            'used_capital': self.config.total_capital - self.config.available_capital,
            'total_position_value': total_position_value,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'number_of_positions': len(self.positions),
            'capital_utilization': (self.config.total_capital - self.config.available_capital) / self.config.total_capital,
            'positions': self.positions,
            'transaction_count': len(self.transaction_history)
        }
    
    def reset_capital(self, new_capital: float = None):
        """
        Reset capital to initial state
        
        Args:
            new_capital: New capital amount (uses config if None)
        """
        if new_capital is not None:
            self.config.total_capital = new_capital
            self.config.available_capital = new_capital
        else:
            self.config.available_capital = self.config.total_capital
        
        self.positions = []
        self.transaction_history = []

def demo_capital_validation():
    """Demonstrate capital validation system"""
    
    print("Capital Validation System Demo")
    print("=" * 50)
    
    # Initialize validator with $100 capital
    config = CapitalConfig(
        total_capital=100,
        available_capital=100,
        max_leverage=1.0,
        max_position_value=0.1,  # 10% max position
        max_risk_per_trade=0.02
    )
    
    validator = CapitalValidator(config)
    
    # Test 1: Valid trade
    print("\n1. Testing Valid Trade")
    print("   Capital: $100, Price: $20, Quantity: 1")
    validation1 = validator.validate_trade(20, 1, 'buy', 'TEST', 80)
    print(f"   Result: {validation1['valid']}")
    print(f"   Message: {validation1['message']}")
    
    # Test 2: Insufficient funds (your bug case)
    print("\n2. Testing Insufficient Funds Bug")
    print("   Capital: $100, Price: $20,000, Quantity: 1")
    validation2 = validator.validate_trade(20000, 1, 'buy', 'EXPENSIVE', 80)
    print(f"   Result: {validation2['valid']}")
    print(f"   Message: {validation2['message']}")
    if not validation2['valid']:
        print(f"   Shortfall: ${validation2['shortfall']:,.2f}")
    
    # Test 3: Over-leverage
    print("\n3. Testing Over-leverage")
    print("   Capital: $100, Price: $50, Quantity: 10")
    validation3 = validator.validate_trade(50, 10, 'buy', 'LEVERAGE', 80)
    print(f"   Result: {validation3['valid']}")
    print(f"   Message: {validation3['message']}")
    
    # Test 4: Calculate max quantity
    print("\n4. Calculating Maximum Safe Quantity")
    print("   Capital: $100, Price: $20, Strategy Score: 80")
    max_qty = validator.calculate_max_quantity(20, 'buy', 80)
    print(f"   Max Quantity: {max_qty['max_quantity']}")
    print(f"   Position Value: ${max_qty['position_value']:.2f}")
    print(f"   Total Cost: ${max_qty['total_cost']:.2f}")
    print(f"   Remaining Capital: ${max_qty['remaining_capital']:.2f}")
    
    # Test 5: Execute valid trade
    print("\n5. Executing Valid Trade")
    result = validator.execute_trade(20, 1, 'buy', 'TEST', 80)
    print(f"   Success: {result['success']}")
    print(f"   Message: {result['message']}")
    print(f"   Remaining Capital: ${result['remaining_capital']:.2f}")
    
    # Show capital summary
    print("\n6. Capital Summary")
    summary = validator.get_capital_summary()
    print(f"   Total Capital: ${summary['total_capital']:.2f}")
    print(f"   Available Capital: ${summary['available_capital']:.2f}")
    print(f"   Used Capital: ${summary['used_capital']:.2f}")
    print(f"   Positions: {summary['number_of_positions']}")
    print(f"   Utilization: {summary['capital_utilization']:.1%}")

if __name__ == "__main__":
    demo_capital_validation() 