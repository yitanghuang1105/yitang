"""
參數調整與穩健性測試介面
提供方便的參數調整和Excel結果匯出功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import random
from sklearn.model_selection import TimeSeriesSplit
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading

warnings.filterwarnings('ignore')

# 導入穩健性測試函數
def apply_cost_model(trades_df, fee=1.5, slippage_long=1.0, slippage_short=2.0):
    if len(trades_df) == 0:
        return pd.DataFrame()
    result_df = trades_df.copy()
    result_df['PnL'] = (result_df['ExitPrice'] - result_df['EntryPrice']) * result_df['Direction']
    result_df['Fee'] = fee * 2
    result_df['Slippage'] = slippage_long * 2
    result_df['NetPnL'] = result_df['PnL'] - result_df['Fee'] - result_df['Slippage']
    return result_df

def load_txf_data(file_path):
    return pd.read_csv(file_path)

def resample_to_4h(df):
    df_4h = df.copy()
    df_4h['Datetime'] = pd.to_datetime(df_4h['Date'] + ' ' + df_4h['Time'])
    df_4h = df_4h.set_index('Datetime')
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'TotalVolume': 'sum'}
    df_4h = df_4h.resample('4H').agg(agg_dict).dropna().reset_index()
    return df_4h

def compute_indicators(df, params, df_4h=None):
    df['BB_MID'] = df['Close'].rolling(params['bb_window']).mean()
    df['BB_STD'] = df['Close'].rolling(params['bb_window']).std()
    df['BB_UPPER'] = df['BB_MID'] + params['bb_std'] * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - params['bb_std'] * df['BB_STD']
    
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(params['rsi_period']).mean()
    avg_loss = pd.Series(loss).rolling(params['rsi_period']).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['TotalVolume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['TotalVolume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_MA'] = df['OBV'].rolling(params['obv_ma_window']).mean()
    
    if df_4h is not None:
        delta = df_4h['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(params['rsi_period']).mean()
        avg_loss = pd.Series(loss).rolling(params['rsi_period']).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df_4h['RSI_4H'] = 100 - (100 / (1 + rs))
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df_4h = df_4h.set_index('Datetime')
        df['RSI_4H'] = df['Datetime'].map(df_4h['RSI_4H'])
    
    return df

def generate_entry_signal(df, params):
    cond1 = df['Close'] < df['BB_LOWER']
    cond2 = df['RSI'] < params['rsi_oversold']
    cond3 = df['OBV'] > df['OBV_MA']
    cond4 = df['RSI_4H'] > 50
    entry = (cond1.astype(int) + cond2.astype(int) + cond3.astype(int)) >= params['entry_n']
    df['EntrySignal'] = entry & cond4
    return df

def generate_exit_signal(df, params):
    cond1 = df['Close'] > df['BB_MID']
    cond2 = df['RSI'] > params['rsi_exit']
    exit_signal = cond1 & cond2
    df['ExitSignal'] = exit_signal
    return df

def generate_trades_from_signals(df):
    trades = []
    position = 0
    entry_idx = None
    for i, row in df.iterrows():
        if position == 0 and row.get('EntrySignal', False):
            position = 1
            entry_idx = i
        elif position == 1 and row.get('ExitSignal', False):
            trade = {
                'EntryTime': df.loc[entry_idx, 'Date'] + ' ' + df.loc[entry_idx, 'Time'],
                'ExitTime': row['Date'] + ' ' + row['Time'],
                'EntryPrice': df.loc[entry_idx, 'Close'],
                'ExitPrice': row['Close'],
                'Direction': 1
            }
            trades.append(trade)
            position = 0
            entry_idx = None
    return pd.DataFrame(trades)

def walk_forward_test(df, params, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        train_df_4h = resample_to_4h(train_df)
        train_df = compute_indicators(train_df, params, train_df_4h)
        
        test_df_4h = resample_to_4h(test_df)
        test_df = compute_indicators(test_df, params, test_df_4h)
        
        train_df = generate_entry_signal(train_df, params)
        train_df = generate_exit_signal(train_df, params)
        test_df = generate_entry_signal(test_df, params)
        test_df = generate_exit_signal(test_df, params)
        
        train_trades = generate_trades_from_signals(train_df)
        test_trades = generate_trades_from_signals(test_df)
        
        if len(train_trades) > 0:
            train_result = apply_cost_model(train_trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            train_pnl = train_result['NetPnL'].sum()
        else:
            train_pnl = 0
            
        if len(test_trades) > 0:
            test_result = apply_cost_model(test_trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            test_pnl = test_result['NetPnL'].sum()
        else:
            test_pnl = 0
        
        results.append({
            'fold': fold + 1,
            'train_pnl': train_pnl,
            'test_pnl': test_pnl,
            'train_trades': len(train_trades),
            'test_trades': len(test_trades)
        })
    
    return pd.DataFrame(results)

def data_shuffle_test(df, params, n_shuffles=3):
    results = []
    
    for i in range(n_shuffles):
        shuffled_df = df.sample(frac=1, random_state=i).reset_index(drop=True)
        shuffled_df['Datetime'] = pd.to_datetime(shuffled_df['Date'] + ' ' + shuffled_df['Time'])
        shuffled_df = shuffled_df.sort_values('Datetime').reset_index(drop=True)
        
        shuffled_df_4h = resample_to_4h(shuffled_df)
        shuffled_df = compute_indicators(shuffled_df, params, shuffled_df_4h)
        shuffled_df = generate_entry_signal(shuffled_df, params)
        shuffled_df = generate_exit_signal(shuffled_df, params)
        
        trades = generate_trades_from_signals(shuffled_df)
        
        if len(trades) > 0:
            result = apply_cost_model(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            total_pnl = result['NetPnL'].sum()
        else:
            total_pnl = 0
        
        results.append({
            'shuffle': i + 1,
            'total_pnl': total_pnl,
            'trades_count': len(trades)
        })
    
    return pd.DataFrame(results)

def noise_injection_test(df, params, noise_levels=[0.001, 0.002]):
    results = []
    
    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, len(df))
        noisy_df = df.copy()
        noisy_df['Close'] = noisy_df['Close'] * (1 + noise)
        noisy_df['Open'] = noisy_df['Open'] * (1 + noise)
        noisy_df['High'] = noisy_df['High'] * (1 + noise)
        noisy_df['Low'] = noisy_df['Low'] * (1 + noise)
        
        noisy_df_4h = resample_to_4h(noisy_df)
        noisy_df = compute_indicators(noisy_df, params, noisy_df_4h)
        noisy_df = generate_entry_signal(noisy_df, params)
        noisy_df = generate_exit_signal(noisy_df, params)
        
        trades = generate_trades_from_signals(noisy_df)
        
        if len(trades) > 0:
            result = apply_cost_model(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            total_pnl = result['NetPnL'].sum()
        else:
            total_pnl = 0
        
        results.append({
            'noise_level': noise_level,
            'noise_percent': f"±{noise_level*100:.1f}%",
            'total_pnl': total_pnl,
            'trades_count': len(trades)
        })
    
    return pd.DataFrame(results)

def year_interval_test(df, params):
    year_ranges = [
        (2020, 2021, "2020-2021 疫情期間"),
        (2021, 2022, "2021-2022 復甦期"),
        (2022, 2023, "2022-2023 調整期")
    ]
    
    results = []
    
    for start_year, end_year, period_name in year_ranges:
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        period_df = df[(df['Year'] >= start_year) & (df['Year'] < end_year)].copy()
        
        if len(period_df) == 0:
            continue
        
        period_df_4h = resample_to_4h(period_df)
        period_df = compute_indicators(period_df, params, period_df_4h)
        period_df = generate_entry_signal(period_df, params)
        period_df = generate_exit_signal(period_df, params)
        
        trades = generate_trades_from_signals(period_df)
        
        if len(trades) > 0:
            result = apply_cost_model(trades, fee=1.5, slippage_long=1.0, slippage_short=2.0)
            total_pnl = result['NetPnL'].sum()
        else:
            total_pnl = 0
        
        results.append({
            'period': period_name,
            'total_pnl': total_pnl,
            'trades_count': len(trades),
            'data_points': len(period_df)
        })
    
    return pd.DataFrame(results)

class ParameterAdjustmentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("量化交易策略參數調整與穩健性測試")
        self.root.geometry("800x600")
        
        self.default_params = {
            'bb_window': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_exit': 70,
            'obv_ma_window': 20,
            'entry_n': 2
        }
        
        self.results = None
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        title_label = ttk.Label(main_frame, text="量化交易策略參數調整", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        param_frame = ttk.LabelFrame(main_frame, text="策略參數調整", padding="10")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # 布林通道參數
        ttk.Label(param_frame, text="布林通道週期:").grid(row=0, column=0, sticky=tk.W)
        self.bb_window_var = tk.StringVar(value="20")
        ttk.Entry(param_frame, textvariable=self.bb_window_var, width=10).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(param_frame, text="布林通道標準差:").grid(row=1, column=0, sticky=tk.W)
        self.bb_std_var = tk.StringVar(value="2.0")
        ttk.Entry(param_frame, textvariable=self.bb_std_var, width=10).grid(row=1, column=1, padx=(10, 0))
        
        # RSI參數
        ttk.Label(param_frame, text="RSI週期:").grid(row=2, column=0, sticky=tk.W)
        self.rsi_period_var = tk.StringVar(value="14")
        ttk.Entry(param_frame, textvariable=self.rsi_period_var, width=10).grid(row=2, column=1, padx=(10, 0))
        
        ttk.Label(param_frame, text="RSI超賣線:").grid(row=3, column=0, sticky=tk.W)
        self.rsi_oversold_var = tk.StringVar(value="30")
        ttk.Entry(param_frame, textvariable=self.rsi_oversold_var, width=10).grid(row=3, column=1, padx=(10, 0))
        
        ttk.Label(param_frame, text="RSI出場線:").grid(row=4, column=0, sticky=tk.W)
        self.rsi_exit_var = tk.StringVar(value="70")
        ttk.Entry(param_frame, textvariable=self.rsi_exit_var, width=10).grid(row=4, column=1, padx=(10, 0))
        
        # 測試設定
        test_frame = ttk.LabelFrame(main_frame, text="測試設定", padding="10")
        test_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        ttk.Label(test_frame, text="測試資料筆數:").grid(row=0, column=0, sticky=tk.W)
        self.data_size_var = tk.StringVar(value="1000")
        ttk.Entry(test_frame, textvariable=self.data_size_var, width=10).grid(row=0, column=1, padx=(10, 0))
        
        # 按鈕
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        self.run_button = ttk.Button(button_frame, text="執行穩健性測試", command=self.run_tests)
        self.run_button.grid(row=0, column=0, padx=(0, 10))
        
        self.export_button = ttk.Button(button_frame, text="匯出Excel結果", command=self.export_excel, state="disabled")
        self.export_button.grid(row=0, column=1, padx=(0, 10))
        
        # 結果顯示
        result_frame = ttk.LabelFrame(main_frame, text="測試結果", padding="10")
        result_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.result_text = tk.Text(result_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 設定權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
    def get_params(self):
        try:
            params = {
                'bb_window': int(self.bb_window_var.get()),
                'bb_std': float(self.bb_std_var.get()),
                'rsi_period': int(self.rsi_period_var.get()),
                'rsi_oversold': int(self.rsi_oversold_var.get()),
                'rsi_exit': int(self.rsi_exit_var.get()),
                'obv_ma_window': 20,
                'entry_n': 2
            }
            return params
        except ValueError as e:
            messagebox.showerror("參數錯誤", f"參數格式錯誤: {e}")
            return None
    
    def run_tests(self):
        params = self.get_params()
        if params is None:
            return
        
        self.run_button.config(state="disabled")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "正在執行穩健性測試...\n")
        self.root.update()
        
        thread = threading.Thread(target=self._run_tests_thread, args=(params,))
        thread.daemon = True
        thread.start()
    
    def _run_tests_thread(self, params):
        try:
            file_path = "TXF1_Minute_2020-01-01_2025-06-16.txt"
            df = load_txf_data(file_path)
            data_size = int(self.data_size_var.get())
            df = df.head(data_size)
            
            walk_forward_results = walk_forward_test(df, params, n_splits=3)
            shuffle_results = data_shuffle_test(df, params, n_shuffles=3)
            noise_results = noise_injection_test(df, params, noise_levels=[0.001, 0.002])
            year_results = year_interval_test(df, params)
            
            self.results = {
                'walk_forward': walk_forward_results,
                'shuffle': shuffle_results,
                'noise': noise_results,
                'year_interval': year_results,
                'params': params
            }
            
            self.root.after(0, self._update_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("測試錯誤", f"執行測試時發生錯誤: {e}"))
            self.root.after(0, lambda: self.run_button.config(state="normal"))
    
    def _update_results(self):
        self.result_text.delete(1.0, tk.END)
        
        self.result_text.insert(tk.END, "="*60 + "\n")
        self.result_text.insert(tk.END, "當前參數設定:\n")
        for key, value in self.results['params'].items():
            self.result_text.insert(tk.END, f"  {key}: {value}\n")
        self.result_text.insert(tk.END, "="*60 + "\n\n")
        
        self.result_text.insert(tk.END, "Walk-Forward 測試結果:\n")
        self.result_text.insert(tk.END, str(self.results['walk_forward']) + "\n\n")
        
        self.result_text.insert(tk.END, "資料洗牌測試結果:\n")
        self.result_text.insert(tk.END, str(self.results['shuffle']) + "\n\n")
        
        self.result_text.insert(tk.END, "噪聲注入測試結果:\n")
        self.result_text.insert(tk.END, str(self.results['noise']) + "\n\n")
        
        self.result_text.insert(tk.END, "年份區間測試結果:\n")
        self.result_text.insert(tk.END, str(self.results['year_interval']) + "\n\n")
        
        self.result_text.insert(tk.END, "測試完成！可以點擊「匯出Excel結果」按鈕匯出詳細結果。\n")
        
        self.run_button.config(state="normal")
        self.export_button.config(state="normal")
    
    def export_excel(self):
        if self.results is None:
            messagebox.showwarning("警告", "請先執行測試")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="保存Excel檔案"
            )
            
            if filename:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    params_df = pd.DataFrame(list(self.results['params'].items()), columns=['參數', '數值'])
                    params_df.to_excel(writer, sheet_name='參數設定', index=False)
                    
                    self.results['walk_forward'].to_excel(writer, sheet_name='Walk_Forward測試', index=False)
                    self.results['shuffle'].to_excel(writer, sheet_name='資料洗牌測試', index=False)
                    self.results['noise'].to_excel(writer, sheet_name='噪聲注入測試', index=False)
                    self.results['year_interval'].to_excel(writer, sheet_name='年份區間測試', index=False)
                    
                    # 測試摘要
                    summary_data = [['測試類型', '平均報酬', '標準差', '交易次數']]
                    
                    if len(self.results['walk_forward']) > 0:
                        avg_pnl = self.results['walk_forward']['test_pnl'].mean()
                        std_pnl = self.results['walk_forward']['test_pnl'].std()
                        total_trades = self.results['walk_forward']['test_trades'].sum()
                        summary_data.append(['Walk-Forward測試', f"{avg_pnl:.2f}", f"{std_pnl:.2f}", total_trades])
                    
                    if len(self.results['shuffle']) > 0:
                        avg_pnl = self.results['shuffle']['total_pnl'].mean()
                        std_pnl = self.results['shuffle']['total_pnl'].std()
                        total_trades = self.results['shuffle']['trades_count'].sum()
                        summary_data.append(['資料洗牌測試', f"{avg_pnl:.2f}", f"{std_pnl:.2f}", total_trades])
                    
                    if len(self.results['noise']) > 0:
                        avg_pnl = self.results['noise']['total_pnl'].mean()
                        std_pnl = self.results['noise']['total_pnl'].std()
                        total_trades = self.results['noise']['trades_count'].sum()
                        summary_data.append(['噪聲注入測試', f"{avg_pnl:.2f}", f"{std_pnl:.2f}", total_trades])
                    
                    if len(self.results['year_interval']) > 0:
                        avg_pnl = self.results['year_interval']['total_pnl'].mean()
                        std_pnl = self.results['year_interval']['total_pnl'].std()
                        total_trades = self.results['year_interval']['trades_count'].sum()
                        summary_data.append(['年份區間測試', f"{avg_pnl:.2f}", f"{std_pnl:.2f}", total_trades])
                    
                    summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
                    summary_df.to_excel(writer, sheet_name='測試摘要', index=False)
                
                messagebox.showinfo("成功", f"Excel檔案已保存至: {filename}")
                
        except Exception as e:
            messagebox.showerror("匯出錯誤", f"匯出Excel時發生錯誤: {e}")

def main():
    root = tk.Tk()
    app = ParameterAdjustmentGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 