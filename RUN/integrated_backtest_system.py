"""
Integrated Backtest System - 整合回測系統

整合RUN資料夾中的所有交易策略，使用TXF1_Minute_2020-01-01_2025-06-16.txt進行回測，
並將結果匯出到Excel檔案中。

系統架構：
1. 資料載入與預處理
2. 多策略執行
3. 績效計算
4. Excel匯出
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Any
import itertools
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

class IntegratedBacktestSystem:
    """整合回測系統主類別"""
    
    def __init__(self, data_file: str = "TXF1_Minute_2020-01-01_2025-06-16.txt"):
        """
        初始化整合回測系統
        
        Args:
            data_file: TXF資料檔案路徑
        """
        self.data_file = data_file
        self.raw_data = None
        self.processed_data = None
        self.strategies = {}
        self.results = {}
        self.excel_folder = "RUN/excel"
        
        # 確保excel資料夾存在
        os.makedirs(self.excel_folder, exist_ok=True)
        
        print("Integrated Backtest System initialized")
    
    def load_data(self) -> pd.DataFrame:
        """
        載入TXF資料並進行預處理
        
        Returns:
            處理後的DataFrame
        """
        print("Loading TXF data...")
        
        try:
            # 載入原始資料
            df = pd.read_csv(self.data_file)
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df.set_index('Datetime', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'TotalVolume']]
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            print(f"Data loaded: {len(df)} records from {df.index[0]} to {df.index[-1]}")
            
            # 移除缺失值
            df = df.dropna()
            
            self.raw_data = df
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def convert_to_4h_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        將1分鐘資料轉換為4小時資料
        
        Args:
            df: 1分鐘資料DataFrame
            
        Returns:
            4小時資料DataFrame
        """
        print("Converting to 4-hour data...")
        
        df = df.copy()
        df = df.reset_index()
        
        # 建立4小時區間
        df['date'] = df['Datetime'].dt.date
        df['hour_block'] = df['Datetime'].dt.hour // 4
        df['period'] = df['date'].astype(str) + '_' + df['hour_block'].astype(str)
        
        # 分組聚合
        grouped = df.groupby('period').agg({
            'Datetime': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        grouped = grouped.set_index('Datetime').sort_index()
        
        print(f"Converted to {len(grouped)} 4-hour records")
        
        self.processed_data = grouped
        return grouped
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標
        
        Args:
            df: OHLCV資料
            
        Returns:
            包含技術指標的DataFrame
        """
        df = df.copy()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2.0)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2.0)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # OBV
        df['obv'] = 0.0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df['obv'].iloc[i] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df['obv'].iloc[i] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                df['obv'].iloc[i] = df['obv'].iloc[i-1]
        
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma']
        
        # Moving Averages for trend filter
        df['ma_short'] = df['close'].rolling(window=10).mean()
        df['ma_long'] = df['close'].rolling(window=50).mean()
        df['trend_up'] = df['ma_short'] > df['ma_long']
        df['trend_down'] = df['ma_short'] < df['ma_long']
        
        # ATR for volatility filter
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)
        
        return df
    
    def strategy_1_basic_risk_management(self, df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """
        策略1：基本策略 + 風險管理
        
        Args:
            df: 包含技術指標的DataFrame
            params: 策略參數
            
        Returns:
            包含交易訊號的DataFrame
        """
        if params is None:
            params = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'obv_threshold': 1.2,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.01,
                'max_position': 1
            }
        
        df = df.copy()
        df['signal'] = 0
        df['position'] = 0
        df['entry_price'] = np.nan
        
        position = 0
        entry_price = np.nan
        
        for i in range(len(df)):
            # 進場條件
            if position == 0:
                buy_condition = (
                    (df['close'].iloc[i] <= df['bb_lower'].iloc[i]) &
                    (df['rsi'].iloc[i] < params['rsi_oversold']) &
                    (df['obv_ratio'].iloc[i] > params['obv_threshold'])
                )
                sell_condition = (
                    (df['close'].iloc[i] >= df['bb_upper'].iloc[i]) &
                    (df['rsi'].iloc[i] > params['rsi_overbought']) &
                    (df['obv_ratio'].iloc[i] < 1/params['obv_threshold'])
                )
                
                if buy_condition and position < params['max_position']:
                    df['signal'].iloc[i] = 1
                    position = 1
                    entry_price = df['close'].iloc[i]
                elif sell_condition and position > -params['max_position']:
                    df['signal'].iloc[i] = -1
                    position = -1
                    entry_price = df['close'].iloc[i]
            else:
                # 持倉中，檢查停損/停利
                if position == 1:
                    if (df['close'].iloc[i] <= entry_price * (1 - params['stop_loss_pct'])) or \
                       (df['close'].iloc[i] >= entry_price * (1 + params['take_profit_pct'])):
                        df['signal'].iloc[i] = -1
                        position = 0
                        entry_price = np.nan
                elif position == -1:
                    if (df['close'].iloc[i] >= entry_price * (1 + params['stop_loss_pct'])) or \
                       (df['close'].iloc[i] <= entry_price * (1 - params['take_profit_pct'])):
                        df['signal'].iloc[i] = 1
                        position = 0
                        entry_price = np.nan
            
            df['position'].iloc[i] = position
            df['entry_price'].iloc[i] = entry_price
        
        return df
    
    def strategy_2_take_profit_1pct(self, df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """
        策略2：基本策略 + 1%停利
        
        Args:
            df: 包含技術指標的DataFrame
            params: 策略參數
            
        Returns:
            包含交易訊號的DataFrame
        """
        if params is None:
            params = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'obv_threshold': 1.2,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.01,  # 1% take profit
                'max_position': 1
            }
        
        # 使用與策略1相同的邏輯，但停利設為1%
        return self.strategy_1_basic_risk_management(df, params)
    
    def strategy_3_signal_filtering(self, df: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """
        策略3：策略 + 訊號過濾
        
        Args:
            df: 包含技術指標的DataFrame
            params: 策略參數
            
        Returns:
            包含交易訊號的DataFrame
        """
        if params is None:
            params = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'obv_threshold': 1.2,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.01,
                'max_position': 1,
                'trend_filter': True,
                'volatility_filter': True
            }
        
        df = df.copy()
        df['signal'] = 0
        df['position'] = 0
        df['entry_price'] = np.nan
        
        position = 0
        entry_price = np.nan
        
        for i in range(len(df)):
            # 基本進場條件
            if position == 0:
                buy_condition = (
                    (df['close'].iloc[i] <= df['bb_lower'].iloc[i]) &
                    (df['rsi'].iloc[i] < params['rsi_oversold']) &
                    (df['obv_ratio'].iloc[i] > params['obv_threshold'])
                )
                sell_condition = (
                    (df['close'].iloc[i] >= df['bb_upper'].iloc[i]) &
                    (df['rsi'].iloc[i] > params['rsi_overbought']) &
                    (df['obv_ratio'].iloc[i] < 1/params['obv_threshold'])
                )
                
                # 趨勢過濾器
                if params.get('trend_filter', False):
                    buy_condition = buy_condition & df['trend_up'].iloc[i]
                    sell_condition = sell_condition & df['trend_down'].iloc[i]
                
                # 波動率過濾器
                if params.get('volatility_filter', False):
                    volatility_ok = df['atr_ratio'].iloc[i] > (df['atr_ratio'].rolling(window=28).mean().iloc[i] * 0.5)
                    buy_condition = buy_condition & volatility_ok
                    sell_condition = sell_condition & volatility_ok
                
                if buy_condition and position < params['max_position']:
                    df['signal'].iloc[i] = 1
                    position = 1
                    entry_price = df['close'].iloc[i]
                elif sell_condition and position > -params['max_position']:
                    df['signal'].iloc[i] = -1
                    position = -1
                    entry_price = df['close'].iloc[i]
            else:
                # 停損/停利邏輯與策略1相同
                if position == 1:
                    if (df['close'].iloc[i] <= entry_price * (1 - params['stop_loss_pct'])) or \
                       (df['close'].iloc[i] >= entry_price * (1 + params['take_profit_pct'])):
                        df['signal'].iloc[i] = -1
                        position = 0
                        entry_price = np.nan
                elif position == -1:
                    if (df['close'].iloc[i] >= entry_price * (1 + params['stop_loss_pct'])) or \
                       (df['close'].iloc[i] <= entry_price * (1 - params['take_profit_pct'])):
                        df['signal'].iloc[i] = 1
                        position = 0
                        entry_price = np.nan
            
            df['position'].iloc[i] = position
            df['entry_price'].iloc[i] = entry_price
        
        return df
    
    def calculate_performance_metrics(self, df: pd.DataFrame, strategy_name: str, params: Dict = None) -> Tuple[Dict, pd.DataFrame]:
        """
        計算策略績效指標
        
        Args:
            df: 包含交易訊號的DataFrame
            strategy_name: 策略名稱
            params: 策略參數
            
        Returns:
            (績效指標字典, 含績效欄位的DataFrame)
        """
        if params is None:
            params = {'transaction_cost': 0.0001, 'initial_capital': 1000000}  # 預設100萬初始資金
        
        df = df.copy()
        
        # 計算策略報酬
        df['price_change'] = df['close'].pct_change()
        df['strategy_return'] = df['position'].shift(1) * df['price_change']
        
        # 交易成本
        transaction_cost = params.get('transaction_cost', 0.0001)
        position_changes = df['position'].diff().abs()
        df['transaction_costs'] = position_changes * transaction_cost
        df['net_return'] = df['strategy_return'] - df['transaction_costs']
        
        # 累積報酬
        df['cumulative_return'] = (1 + df['net_return']).cumprod()
        
        # 績效指標
        total_return = df['cumulative_return'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        
        # 波動率
        daily_returns = df['net_return'].resample('D').sum()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 夏普比率
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative = df['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 勝率
        trades = df[df['signal'] != 0]
        if len(trades) > 0:
            winning_trades = trades[trades['strategy_return'] > 0]
            win_rate = len(winning_trades) / len(trades)
        else:
            win_rate = 0
        
        # 交易統計
        long_trades = trades[trades['signal'] == 1]
        short_trades = trades[trades['signal'] == -1]
        
        # 平均獲利/虧損
        if len(trades) > 0:
            avg_profit = trades[trades['strategy_return'] > 0]['strategy_return'].mean() if len(trades[trades['strategy_return'] > 0]) > 0 else 0
            avg_loss = trades[trades['strategy_return'] < 0]['strategy_return'].mean() if len(trades[trades['strategy_return'] < 0]) > 0 else 0
        else:
            avg_profit = avg_loss = 0
        
        # 計算實際損益金額
        initial_capital = params.get('initial_capital', 1000000)  # 預設100萬
        total_pnl_amount = total_return * initial_capital
        final_capital = initial_capital + total_pnl_amount
        
        metrics = {
            'strategy_name': strategy_name,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'num_long_trades': len(long_trades),
            'num_short_trades': len(short_trades),
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf'),
            'final_cumulative_return': df['cumulative_return'].iloc[-1],
            'total_pnl': df['net_return'].sum(),
            'initial_capital': initial_capital,
            'total_pnl_amount': total_pnl_amount,
            'final_capital': final_capital
        }
        return metrics, df
    
    def run_all_strategies(self) -> Dict:
        """
        執行所有策略
        
        Returns:
            所有策略結果字典
        """
        print("Running all strategies...")
        
        # 載入資料
        df = self.load_data()
        if df is None:
            return {}
        
        # 轉換為4小時資料
        df_4h = self.convert_to_4h_data(df)
        
        # 計算技術指標
        df_with_indicators = self.calculate_technical_indicators(df_4h)
        
        # 定義策略
        strategies = {
            'Strategy_1_Basic_Risk_Management': self.strategy_1_basic_risk_management,
            'Strategy_2_Take_Profit_1pct': self.strategy_2_take_profit_1pct,
            'Strategy_3_Signal_Filtering': self.strategy_3_signal_filtering
        }
        
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            print(f"Running {strategy_name}...")
            try:
                # 執行策略
                df_with_signals = strategy_func(df_with_indicators.copy())
                # 計算績效，並取得含 cumulative_return 的 DataFrame
                metrics, df_with_metrics = self.calculate_performance_metrics(df_with_signals, strategy_name)
                # 將績效欄位存回 data
                results[strategy_name] = {
                    'metrics': metrics,
                    'data': df_with_metrics  # 這裡 df_with_metrics 已含 cumulative_return
                }
                print(f"✅ {strategy_name} completed")
            except Exception as e:
                print(f"❌ {strategy_name} failed: {e}")
                results[strategy_name] = {
                    'metrics': {'strategy_name': strategy_name, 'error': str(e)},
                    'data': None
                }
        self.results = results
        return results
    
    def export_to_excel(self, filename: str = None) -> str:
        """
        將結果匯出到Excel
        
        Args:
            filename: Excel檔案名稱
            
        Returns:
            Excel檔案路徑
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.xlsx"
        
        filepath = os.path.join(self.excel_folder, filename)
        
        print(f"Exporting results to {filepath}...")
        
        # 創建Excel工作簿
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            
            # 1. 績效摘要表
            summary_data = []
            for strategy_name, result in self.results.items():
                if 'error' not in result['metrics']:
                    metrics = result['metrics']
                    summary_data.append({
                        'Strategy': strategy_name,
                        'Total Return (%)': metrics['total_return'] * 100,
                        'Annual Return (%)': metrics['annual_return'] * 100,
                        'Volatility (%)': metrics['volatility'] * 100,
                        'Sharpe Ratio': metrics['sharpe_ratio'],
                        'Max Drawdown (%)': metrics['max_drawdown'] * 100,
                        'Win Rate (%)': metrics['win_rate'] * 100,
                        'Number of Trades': metrics['num_trades'],
                        'Number of Long Trades': metrics['num_long_trades'],
                        'Number of Short Trades': metrics['num_short_trades'],
                        'Average Profit (%)': metrics['avg_profit'] * 100,
                        'Average Loss (%)': metrics['avg_loss'] * 100,
                        'Profit Factor': metrics['profit_factor'],
                        'Total PnL': metrics['total_pnl']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
            
            # 2. 詳細交易記錄
            for strategy_name, result in self.results.items():
                if result['data'] is not None:
                    # 提取交易記錄
                    trades_df = result['data'][result['data']['signal'] != 0].copy()
                    if len(trades_df) > 0:
                        trades_df['Date'] = trades_df.index.date
                        trades_df['Time'] = trades_df.index.time
                        trades_df['Strategy'] = strategy_name
                        
                        # 選擇重要欄位
                        export_columns = ['Date', 'Time', 'Strategy', 'signal', 'position', 'entry_price', 
                                        'close', 'rsi', 'bb_position', 'obv_ratio']
                        trades_df = trades_df[export_columns]
                        
                        sheet_name = f"{strategy_name[:30]}"  # Excel工作表名稱限制
                        trades_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 3. 原始資料摘要
            if self.processed_data is not None:
                data_summary = self.processed_data.describe()
                data_summary.to_excel(writer, sheet_name='Data_Summary')
            
            # 4. 策略參數設定
            params_data = []
            for strategy_name in self.results.keys():
                params_data.append({
                    'Strategy': strategy_name,
                    'RSI Oversold': 30,
                    'RSI Overbought': 70,
                    'OBV Threshold': 1.2,
                    'Stop Loss (%)': 2.0,
                    'Take Profit (%)': 1.0,
                    'Transaction Cost (%)': 0.01
                })
            
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='Strategy_Parameters', index=False)
        
        print(f"✅ Results exported to {filepath}")
        return filepath
    
    def generate_performance_charts(self, filename: str = None) -> str:
        """
        生成績效圖表（中文化）
        
        Args:
            filename: 圖表檔案名稱
            
        Returns:
            圖表檔案路徑
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_charts_{timestamp}.png"
        
        filepath = os.path.join(self.excel_folder, filename)
        
        # 創建圖表 (2x3 佈局)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('交易策略績效比較', fontsize=16, fontweight='bold')
        
        # 1. 累積報酬比較
        ax1 = axes[0, 0]
        for strategy_name, result in self.results.items():
            if result['data'] is not None:
                df = result['data'].copy()
                # 若無 cumulative_return 則補上
                if 'cumulative_return' not in df.columns and 'net_return' in df.columns:
                    df['cumulative_return'] = (1 + df['net_return']).cumprod()
                # 強制 index 轉為 DatetimeIndex（若可能）
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    print(f"[DEBUG] {strategy_name} index 轉換失敗: {e}")
                # 檢查 cumulative_return 是否有有效數值
                if 'cumulative_return' in df.columns and df['cumulative_return'].notna().sum() > 0:
                    ax1.plot(df.index, df['cumulative_return'].values, 
                            label=strategy_name, linewidth=2)
                else:
                    print(f"[DEBUG] {strategy_name} 沒有有效的 cumulative_return。欄位: {df.columns.tolist()}")
                    print(df.head())
        ax1.set_title('累積報酬比較')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('累積報酬')
        ax1.legend(title='策略')
        ax1.grid(True, alpha=0.3)
        
        # 2. 勝率比較
        ax2 = axes[0, 1]
        win_rates = []
        strategy_names = []
        for strategy_name, result in self.results.items():
            if 'error' not in result['metrics']:
                win_rates.append(result['metrics']['win_rate'] * 100)
                strategy_names.append(strategy_name)
        bars = ax2.bar(strategy_names, win_rates, color=['#2E8B57', '#4682B4', '#CD853F'])
        ax2.set_title('勝率比較')
        ax2.set_ylabel('勝率 (%)')
        ax2.set_ylim(0, 100)
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # 3. 夏普比率比較
        ax3 = axes[0, 2]
        sharpe_ratios = []
        for strategy_name, result in self.results.items():
            if 'error' not in result['metrics']:
                sharpe_ratios.append(result['metrics']['sharpe_ratio'])
        bars = ax3.bar(strategy_names, sharpe_ratios, color=['#32CD32', '#4169E1', '#FF6347'])
        ax3.set_title('夏普比率比較')
        ax3.set_ylabel('夏普比率')
        ax3.grid(True, alpha=0.3)
        for bar, ratio in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.3f}', ha='center', va='bottom')
        
        # 4. 交易次數比較
        ax4 = axes[1, 0]
        trade_counts = []
        for strategy_name, result in self.results.items():
            if 'error' not in result['metrics']:
                trade_counts.append(result['metrics']['num_trades'])
        bars = ax4.bar(strategy_names, trade_counts, color=['#FFD700', '#9370DB', '#20B2AA'])
        ax4.set_title('交易次數比較')
        ax4.set_ylabel('交易次數')
        ax4.grid(True, alpha=0.3)
        for bar, count in zip(bars, trade_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # 5. 實際損益金額比較 (萬元)
        ax5 = axes[1, 1]
        pnl_amounts = []
        for strategy_name, result in self.results.items():
            if 'error' not in result['metrics']:
                pnl_amounts.append(result['metrics']['total_pnl_amount'] / 10000)  # 轉換為萬元
        colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in pnl_amounts]  # 虧損紅色，獲利綠色
        bars = ax5.bar(strategy_names, pnl_amounts, color=colors)
        ax5.set_title('實際損益金額比較')
        ax5.set_ylabel('損益金額 (萬元)')
        ax5.grid(True, alpha=0.3)
        for bar, amount in zip(bars, pnl_amounts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + (0.5 if amount >= 0 else -1.5),
                    f'{amount:.1f}萬', ha='center', va='bottom' if amount >= 0 else 'top')
        
        # 6. 最終資金比較 (萬元)
        ax6 = axes[1, 2]
        final_capitals = []
        for strategy_name, result in self.results.items():
            if 'error' not in result['metrics']:
                final_capitals.append(result['metrics']['final_capital'] / 10000)  # 轉換為萬元
        bars = ax6.bar(strategy_names, final_capitals, color=['#87CEEB', '#98FB98', '#DDA0DD'])
        ax6.set_title('最終資金比較')
        ax6.set_ylabel('最終資金 (萬元)')
        ax6.grid(True, alpha=0.3)
        for bar, capital in zip(bars, final_capitals):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{capital:.1f}萬', ha='center', va='bottom')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 績效圖表已儲存於 {filepath}")
        return filepath
    
    def run_complete_backtest(self) -> Tuple[str, str]:
        """
        執行完整的回測流程
        
        Returns:
            (Excel檔案路徑, 圖表檔案路徑)
        """
        print("="*60)
        print("INTEGRATED BACKTEST SYSTEM - COMPLETE EXECUTION")
        print("="*60)
        
        # 執行所有策略
        results = self.run_all_strategies()
        
        if not results:
            print("❌ No strategies completed successfully")
            return None, None
        
        # 匯出到Excel
        excel_file = self.export_to_excel()
        
        # 生成圖表
        chart_file = self.generate_performance_charts()
        
        # 打印摘要
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        
        for strategy_name, result in results.items():
            if 'error' not in result['metrics']:
                metrics = result['metrics']
                print(f"\n{strategy_name}:")
                print(f"  初始資金: {metrics['initial_capital']:,.0f} 元")
                print(f"  總報酬率: {metrics['total_return']*100:.2f}%")
                print(f"  實際損益: {metrics['total_pnl_amount']:,.0f} 元")
                print(f"  最終資金: {metrics['final_capital']:,.0f} 元")
                print(f"  勝率: {metrics['win_rate']*100:.1f}%")
                print(f"  交易次數: {metrics['num_trades']}")
                print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
                print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        
        print(f"\n📊 Results exported to: {excel_file}")
        print(f"📈 Charts saved to: {chart_file}")
        
        return excel_file, chart_file


def main():
    """主函數"""
    # 創建整合回測系統
    backtest_system = IntegratedBacktestSystem()
    
    # 執行完整回測
    excel_file, chart_file = backtest_system.run_complete_backtest()
    
    if excel_file and chart_file:
        print("\n🎉 Backtest completed successfully!")
        print(f"📁 Excel file: {excel_file}")
        print(f"📁 Chart file: {chart_file}")
    else:
        print("\n❌ Backtest failed")


if __name__ == "__main__":
    main() 