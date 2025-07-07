"""
Performance Analysis Module
Calculates performance metrics including MDD, equity curve, and other statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

class PerformanceAnalyzer:
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize performance analyzer
        
        Args:
            initial_capital: Initial trading capital
        """
        self.initial_capital = initial_capital
        
    def calculate_equity_curve(self, df: pd.DataFrame, decisions: pd.Series, 
                              position_size: float = 0.1) -> pd.Series:
        """
        Calculate equity curve based on trading decisions
        
        Args:
            df: DataFrame with OHLCV data
            decisions: Trading decisions (Buy/Sell/Hold)
            position_size: Position size as fraction of capital (0.1 = 10%)
        
        Returns:
            pd.Series: Equity curve values
        """
        equity = pd.Series(index=df.index, dtype=float)
        equity.iloc[0] = self.initial_capital
        
        position = 0  # Current position (1 = long, -1 = short, 0 = no position)
        entry_price = 0
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            decision = decisions.iloc[i]
            
            # Update equity based on current position
            if position == 1:  # Long position
                equity.iloc[i] = equity.iloc[i-1] + (current_price - entry_price) * position_size * self.initial_capital / entry_price
            elif position == -1:  # Short position
                equity.iloc[i] = equity.iloc[i-1] + (entry_price - current_price) * position_size * self.initial_capital / entry_price
            else:  # No position
                equity.iloc[i] = equity.iloc[i-1]
            
            # Handle trading decisions
            if decision == 'Buy' and position <= 0:
                position = 1
                entry_price = current_price
            elif decision == 'Sell' and position >= 0:
                position = -1
                entry_price = current_price
            elif decision == 'Hold':
                # Keep current position
                pass
        
        return equity
    
    def calculate_mdd(self, equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate Maximum Drawdown (MDD)
        
        Args:
            equity_curve: Series of equity values
        
        Returns:
            Tuple: (MDD percentage, peak date, trough date)
        """
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max * 100
        
        # Find maximum drawdown
        mdd = drawdown.min()
        trough_idx = drawdown.idxmin()
        
        # Find peak before trough
        peak_idx = equity_curve.loc[:trough_idx].idxmax()
        
        return mdd, peak_idx, trough_idx
    
    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate period returns
        
        Args:
            equity_curve: Series of equity values
        
        Returns:
            pd.Series: Period returns
        """
        returns = equity_curve.pct_change().dropna()
        return returns
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
        
        Returns:
            float: Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Annualize returns and volatility
        annual_return = returns.mean() * 252  # Assuming daily data
        annual_volatility = returns.std() * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0.0
        
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        return sharpe
    
    def calculate_calmar_ratio(self, equity_curve: pd.Series, returns: pd.Series) -> float:
        """
        Calculate Calmar ratio (annual return / maximum drawdown)
        
        Args:
            equity_curve: Series of equity values
            returns: Series of returns
        
        Returns:
            float: Calmar ratio
        """
        mdd, _, _ = self.calculate_mdd(equity_curve)
        annual_return = returns.mean() * 252
        
        if abs(mdd) < 0.0001:  # Avoid division by zero
            return 0.0
        
        calmar = annual_return / abs(mdd)
        return calmar
    
    def calculate_win_rate(self, equity_curve: pd.Series) -> float:
        """
        Calculate win rate (percentage of profitable periods)
        
        Args:
            equity_curve: Series of equity values
        
        Returns:
            float: Win rate percentage
        """
        returns = self.calculate_returns(equity_curve)
        wins = (returns > 0).sum()
        total = len(returns)
        
        if total == 0:
            return 0.0
        
        win_rate = (wins / total) * 100
        return win_rate
    
    def calculate_performance_metrics(self, df: pd.DataFrame, decisions: pd.Series, 
                                    position_size: float = 0.1) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            df: DataFrame with OHLCV data
            decisions: Trading decisions
            position_size: Position size as fraction of capital
        
        Returns:
            Dict: Dictionary containing all performance metrics
        """
        # Calculate equity curve
        equity_curve = self.calculate_equity_curve(df, decisions, position_size)
        returns = self.calculate_returns(equity_curve)
        
        # Calculate MDD
        mdd, peak_date, trough_date = self.calculate_mdd(equity_curve)
        
        # Calculate other metrics
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(equity_curve, returns)
        win_rate = self.calculate_win_rate(equity_curve)
        
        # Calculate volatility
        annual_volatility = returns.std() * np.sqrt(252) * 100
        
        # Calculate average win/loss
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        avg_win = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() * 100 if len(negative_returns) > 0 else 0
        
        metrics = {
            'equity_curve': equity_curve,
            'returns': returns,
            'total_return_pct': total_return,
            'mdd_pct': mdd,
            'peak_date': peak_date,
            'trough_date': trough_date,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'win_rate_pct': win_rate,
            'annual_volatility_pct': annual_volatility,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'final_equity': equity_curve.iloc[-1],
            'max_equity': equity_curve.max(),
            'min_equity': equity_curve.min()
        }
        
        return metrics
    
    def plot_performance_analysis(self, df: pd.DataFrame, performance_metrics: Dict, 
                                save_path: str = None) -> None:
        """
        Plot comprehensive performance analysis
        
        Args:
            df: Original price data
            performance_metrics: Performance metrics dictionary
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Performance Analysis', fontsize=16)
        
        equity_curve = performance_metrics['equity_curve']
        returns = performance_metrics['returns']
        
        # Plot 1: Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(equity_curve.index, equity_curve, label='Equity Curve', linewidth=2, color='blue')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        ax2 = axes[0, 1]
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Drawdown Analysis')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Returns Distribution
        ax3 = axes[1, 0]
        ax3.hist(returns * 100, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=returns.mean() * 100, color='red', linestyle='--', label=f'Mean: {returns.mean()*100:.2f}%')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative Returns
        ax4 = axes[1, 1]
        cumulative_returns = (1 + returns).cumprod() - 1
        ax4.plot(cumulative_returns.index, cumulative_returns * 100, label='Cumulative Returns', linewidth=2, color='purple')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax4.set_title('Cumulative Returns')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Rolling Sharpe Ratio
        ax5 = axes[2, 0]
        rolling_sharpe = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
        ax5.plot(rolling_sharpe.index, rolling_sharpe, label='30-Day Rolling Sharpe', linewidth=2, color='orange')
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax5.set_title('Rolling Sharpe Ratio (30-day)')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Performance Metrics Summary
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Create text summary
        summary_text = f"""
Performance Summary:
===================
Total Return: {performance_metrics['total_return_pct']:.2f}%
Maximum Drawdown: {performance_metrics['mdd_pct']:.2f}%
Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}
Calmar Ratio: {performance_metrics['calmar_ratio']:.2f}
Win Rate: {performance_metrics['win_rate_pct']:.1f}%
Annual Volatility: {performance_metrics['annual_volatility_pct']:.2f}%
Average Win: {performance_metrics['avg_win_pct']:.2f}%
Average Loss: {performance_metrics['avg_loss_pct']:.2f}%
Final Equity: ${performance_metrics['final_equity']:,.0f}
Peak Date: {performance_metrics['peak_date'].strftime('%Y-%m-%d')}
Trough Date: {performance_metrics['trough_date'].strftime('%Y-%m-%d')}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance analysis plot saved to {save_path}")
        
        plt.show()

def analyze_strategy_performance(df: pd.DataFrame, decisions: pd.Series, 
                               initial_capital: float = 100000, 
                               position_size: float = 0.1) -> Dict:
    """
    Convenience function to analyze strategy performance
    
    Args:
        df: DataFrame with OHLCV data
        decisions: Trading decisions
        initial_capital: Initial capital
        position_size: Position size as fraction of capital
    
    Returns:
        Dict: Performance metrics
    """
    analyzer = PerformanceAnalyzer(initial_capital)
    metrics = analyzer.calculate_performance_metrics(df, decisions, position_size)
    return metrics 