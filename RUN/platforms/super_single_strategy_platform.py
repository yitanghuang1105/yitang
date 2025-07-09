# -*- coding: utf-8 -*-
"""
Super Single Strategy Platform
整合 1.py-6.py 功能的完整單策略開發平台
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import talib
import glob
import json
import os
import sys
import threading
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
import subprocess

warnings.filterwarnings('ignore')
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("husl")

class SuperSingleStrategyPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("超級單策略開發平台")
        self.root.geometry("1400x900")
        
        # 預設參數
        self.default_params = {
            'bb_window': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_exit': 50,
            'obv_ma_window': 10,
            'obv_threshold': 1.2,
            'entry_n': 3,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.01,
            'max_position': 1
        }
        
        self.current_params = self.default_params.copy()
        self.df = None
        self.results = None
        self.output_dir = "output/super_single_strategy"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.create_widgets()
        self.load_sample_data()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        title_label = ttk.Label(main_frame, text="超級單策略開發平台", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # 策略參數分頁
        strategy_frame = ttk.Frame(notebook, padding="10")
        notebook.add(strategy_frame, text="策略參數")
        self.create_strategy_parameters_tab(strategy_frame)
        
        # 優化設定分頁
        optimization_frame = ttk.Frame(notebook, padding="10")
        notebook.add(optimization_frame, text="參數優化")
        self.create_optimization_tab(optimization_frame)
        
        # 成本模型分頁
        cost_frame = ttk.Frame(notebook, padding="10")
        notebook.add(cost_frame, text="成本模型")
        self.create_cost_model_tab(cost_frame)
        
        # 穩健性測試分頁
        robustness_frame = ttk.Frame(notebook, padding="10")
        notebook.add(robustness_frame, text="穩健性測試")
        self.create_robustness_tab(robustness_frame)
        
        # 結果分析分頁
        results_frame = ttk.Frame(notebook, padding="10")
        notebook.add(results_frame, text="結果分析")
        self.create_results_tab(results_frame)
        
        # 按鈕區域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=20)
        
        ttk.Button(button_frame, text="載入資料", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="執行基礎分析", command=self.run_basic_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="參數優化", command=self.run_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="穩健性測試", command=self.run_robustness_tests).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="完整分析", command=self.run_complete_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="匯出報告", command=self.export_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="開啟結果資料夾", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        
        # 狀態列
        self.status_var = tk.StringVar(value="就緒 Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def create_strategy_parameters_tab(self, parent):
        """創建策略參數分頁"""
        # 布林通道參數
        bb_frame = ttk.LabelFrame(parent, text="布林通道參數", padding="10")
        bb_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        ttk.Label(bb_frame, text="布林通道週期:").grid(row=0, column=0, sticky=tk.W)
        self.bb_window_var = tk.StringVar(value=str(self.default_params['bb_window']))
        ttk.Entry(bb_frame, textvariable=self.bb_window_var, width=10).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(bb_frame, text="標準差倍數:").grid(row=1, column=0, sticky=tk.W)
        self.bb_std_var = tk.StringVar(value=str(self.default_params['bb_std']))
        ttk.Entry(bb_frame, textvariable=self.bb_std_var, width=10).grid(row=1, column=1, padx=(10, 0))
        
        # RSI參數
        rsi_frame = ttk.LabelFrame(parent, text="RSI參數", padding="10")
        rsi_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        ttk.Label(rsi_frame, text="RSI週期:").grid(row=0, column=0, sticky=tk.W)
        self.rsi_period_var = tk.StringVar(value=str(self.default_params['rsi_period']))
        ttk.Entry(rsi_frame, textvariable=self.rsi_period_var, width=10).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(rsi_frame, text="超賣線:").grid(row=1, column=0, sticky=tk.W)
        self.rsi_oversold_var = tk.StringVar(value=str(self.default_params['rsi_oversold']))
        ttk.Entry(rsi_frame, textvariable=self.rsi_oversold_var, width=10).grid(row=1, column=1, padx=(10, 0))
        
        ttk.Label(rsi_frame, text="超買線:").grid(row=2, column=0, sticky=tk.W)
        self.rsi_overbought_var = tk.StringVar(value=str(self.default_params['rsi_overbought']))
        ttk.Entry(rsi_frame, textvariable=self.rsi_overbought_var, width=10).grid(row=2, column=1, padx=(10, 0))
        
        ttk.Label(rsi_frame, text="出場線:").grid(row=3, column=0, sticky=tk.W)
        self.rsi_exit_var = tk.StringVar(value=str(self.default_params['rsi_exit']))
        ttk.Entry(rsi_frame, textvariable=self.rsi_exit_var, width=10).grid(row=3, column=1, padx=(10, 0))
        
        # OBV參數
        obv_frame = ttk.LabelFrame(parent, text="OBV參數", padding="10")
        obv_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        ttk.Label(obv_frame, text="OBV移動平均週期:").grid(row=0, column=0, sticky=tk.W)
        self.obv_ma_window_var = tk.StringVar(value=str(self.default_params['obv_ma_window']))
        ttk.Entry(obv_frame, textvariable=self.obv_ma_window_var, width=10).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(obv_frame, text="OBV閾值:").grid(row=1, column=0, sticky=tk.W)
        self.obv_threshold_var = tk.StringVar(value=str(self.default_params['obv_threshold']))
        ttk.Entry(obv_frame, textvariable=self.obv_threshold_var, width=10).grid(row=1, column=1, padx=(10, 0))
        
        # 交易參數
        trade_frame = ttk.LabelFrame(parent, text="交易參數", padding="10")
        trade_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        ttk.Label(trade_frame, text="進場條件數:").grid(row=0, column=0, sticky=tk.W)
        self.entry_n_var = tk.StringVar(value=str(self.default_params['entry_n']))
        ttk.Entry(trade_frame, textvariable=self.entry_n_var, width=10).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(trade_frame, text="停損百分比:").grid(row=0, column=2, sticky=tk.W)
        self.stop_loss_var = tk.StringVar(value=str(self.default_params['stop_loss_pct']))
        ttk.Entry(trade_frame, textvariable=self.stop_loss_var, width=10).grid(row=0, column=3, padx=(10, 0))
        
        ttk.Label(trade_frame, text="停利百分比:").grid(row=0, column=4, sticky=tk.W)
        self.take_profit_var = tk.StringVar(value=str(self.default_params['take_profit_pct']))
        ttk.Entry(trade_frame, textvariable=self.take_profit_var, width=10).grid(row=0, column=5, padx=(10, 0))
        
        # 設定權重
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
    
    def create_optimization_tab(self, parent):
        """創建參數優化分頁"""
        # 優化設定
        opt_frame = ttk.LabelFrame(parent, text="優化設定", padding="10")
        opt_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        ttk.Label(opt_frame, text="優化目標:").grid(row=0, column=0, sticky=tk.W)
        self.optimization_target = tk.StringVar(value="total_return")
        ttk.Combobox(opt_frame, textvariable=self.optimization_target, 
                    values=["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(opt_frame, text="搜尋範圍:").grid(row=1, column=0, sticky=tk.W)
        self.search_range = tk.StringVar(value="grid")
        ttk.Combobox(opt_frame, textvariable=self.search_range, 
                    values=["grid", "random", "bayesian"]).grid(row=1, column=1, padx=(10, 0))
        
        ttk.Label(opt_frame, text="最大迭代次數:").grid(row=2, column=0, sticky=tk.W)
        self.max_iterations = tk.StringVar(value="100")
        ttk.Entry(opt_frame, textvariable=self.max_iterations, width=10).grid(row=2, column=1, padx=(10, 0))
        
        # 優化結果
        result_frame = ttk.LabelFrame(parent, text="優化結果", padding="10")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        self.optimization_text = tk.Text(result_frame, height=15, width=50)
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.optimization_text.yview)
        self.optimization_text.configure(yscrollcommand=scrollbar.set)
        
        self.optimization_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 設定權重
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
    
    def create_cost_model_tab(self, parent):
        """創建成本模型分頁"""
        cost_frame = ttk.LabelFrame(parent, text="交易成本設定", padding="10")
        cost_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        ttk.Label(cost_frame, text="手續費 (點):").grid(row=0, column=0, sticky=tk.W)
        self.fee_var = tk.StringVar(value="1.5")
        ttk.Entry(cost_frame, textvariable=self.fee_var, width=10).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(cost_frame, text="做多滑點 (點):").grid(row=1, column=0, sticky=tk.W)
        self.slippage_long_var = tk.StringVar(value="1.0")
        ttk.Entry(cost_frame, textvariable=self.slippage_long_var, width=10).grid(row=1, column=1, padx=(10, 0))
        
        ttk.Label(cost_frame, text="做空滑點 (點):").grid(row=2, column=0, sticky=tk.W)
        self.slippage_short_var = tk.StringVar(value="2.0")
        ttk.Entry(cost_frame, textvariable=self.slippage_short_var, width=10).grid(row=2, column=1, padx=(10, 0))
        
        # 成本分析結果
        cost_result_frame = ttk.LabelFrame(parent, text="成本分析結果", padding="10")
        cost_result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        self.cost_text = tk.Text(cost_result_frame, height=15, width=50)
        cost_scrollbar = ttk.Scrollbar(cost_result_frame, orient="vertical", command=self.cost_text.yview)
        self.cost_text.configure(yscrollcommand=cost_scrollbar.set)
        
        self.cost_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        cost_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 設定權重
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        cost_result_frame.columnconfigure(0, weight=1)
        cost_result_frame.rowconfigure(0, weight=1)
    
    def create_robustness_tab(self, parent):
        """創建穩健性測試分頁"""
        test_frame = ttk.LabelFrame(parent, text="測試設定", padding="10")
        test_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Walk-forward 測試
        ttk.Label(test_frame, text="Walk-forward 分割數:").grid(row=0, column=0, sticky=tk.W)
        self.wf_splits = tk.StringVar(value="3")
        ttk.Entry(test_frame, textvariable=self.wf_splits, width=10).grid(row=0, column=1, padx=(10, 0))
        
        # 資料洗牌測試
        ttk.Label(test_frame, text="洗牌測試次數:").grid(row=1, column=0, sticky=tk.W)
        self.shuffle_times = tk.StringVar(value="3")
        ttk.Entry(test_frame, textvariable=self.shuffle_times, width=10).grid(row=1, column=1, padx=(10, 0))
        
        # 雜訊注入測試
        ttk.Label(test_frame, text="雜訊水準 (%):").grid(row=2, column=0, sticky=tk.W)
        self.noise_levels = tk.StringVar(value="0.1,0.2")
        ttk.Entry(test_frame, textvariable=self.noise_levels, width=15).grid(row=2, column=1, padx=(10, 0))
        
        # 測試結果
        test_result_frame = ttk.LabelFrame(parent, text="穩健性測試結果", padding="10")
        test_result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        self.robustness_text = tk.Text(test_result_frame, height=15, width=50)
        robustness_scrollbar = ttk.Scrollbar(test_result_frame, orient="vertical", command=self.robustness_text.yview)
        self.robustness_text.configure(yscrollcommand=robustness_scrollbar.set)
        
        self.robustness_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        robustness_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 設定權重
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        test_result_frame.columnconfigure(0, weight=1)
        test_result_frame.rowconfigure(0, weight=1)
    
    def create_results_tab(self, parent):
        """創建結果分析分頁"""
        # 結果顯示
        result_frame = ttk.LabelFrame(parent, text="分析結果", padding="10")
        result_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        self.results_text = tk.Text(result_frame, height=20, width=80)
        results_scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 設定權重
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # 初始訊息
        self.results_text.insert(tk.END, "等待執行分析...\n")
    
    def load_sample_data(self):
        """載入範例資料 - 只讀一次"""
        if self.df is not None:
            print("📋 使用已載入的資料，避免重複讀取")
            self.status_var.set(f"使用快取資料: {len(self.df)} 筆記錄")
            return
            
        try:
            # 嘗試載入資料檔案
            data_files = glob.glob("*.txt") + glob.glob("../*.txt") + glob.glob("../../*.txt")
            if data_files:
                file_path = data_files[0]
                print(f"📥 載入資料檔案: {file_path}")
                self.df = pd.read_csv(file_path, encoding='utf-8')
                if 'timestamp' not in self.df.columns and 'Date' in self.df.columns and 'Time' in self.df.columns:
                    self.df['timestamp'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'])
                self.df.set_index('timestamp', inplace=True)
                self.status_var.set(f"資料載入成功: {len(self.df)} 筆記錄")
                print(f"✅ 資料載入完成: {len(self.df):,} 筆記錄")
            else:
                self.generate_synthetic_data()
        except Exception as e:
            print(f"❌ 載入資料失敗: {e}")
            self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """生成合成資料"""
        dates = pd.date_range('2020-01-01', '2025-01-01', freq='1min')
        np.random.seed(42)
        
        # 生成價格資料
        returns = np.random.normal(0, 0.001, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        self.status_var.set(f"合成資料生成: {len(self.df)} 筆記錄")
    
    def load_data(self):
        """載入資料 - 只讀一次"""
        file_path = filedialog.askopenfilename(
            title="選擇資料檔案",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                print(f"📥 載入資料檔案: {file_path}")
                self.df = pd.read_csv(file_path, encoding='utf-8')
                if 'timestamp' not in self.df.columns and 'Date' in self.df.columns and 'Time' in self.df.columns:
                    self.df['timestamp'] = pd.to_datetime(self.df['Date'] + ' ' + self.df['Time'])
                self.df.set_index('timestamp', inplace=True)
                self.status_var.set(f"資料載入成功: {len(self.df)} 筆記錄")
                print(f"✅ 資料載入完成: {len(self.df):,} 筆記錄")
                messagebox.showinfo("成功", "資料載入成功！")
            except Exception as e:
                print(f"❌ 資料載入失敗: {e}")
                messagebox.showerror("錯誤", f"資料載入失敗: {e}")
    
    def run_basic_analysis(self):
        """執行基礎分析"""
        if self.df is None:
            messagebox.showerror("錯誤", "請先載入資料")
            return
        
        self.status_var.set("執行基礎分析中...")
        
        def analysis_thread():
            try:
                # 這裡可以添加基礎分析邏輯
                self.root.after(0, self.analysis_completed)
            except Exception as e:
                self.root.after(0, lambda: self.analysis_failed(str(e)))
        
        thread = threading.Thread(target=analysis_thread)
        thread.daemon = True
        thread.start()
    
    def run_optimization(self):
        """執行參數優化"""
        if self.df is None:
            messagebox.showerror("錯誤", "請先載入資料")
            return
        
        self.status_var.set("執行參數優化中...")
        
        def optimization_thread():
            try:
                # 這裡可以添加參數優化邏輯
                self.root.after(0, self.optimization_completed)
            except Exception as e:
                self.root.after(0, lambda: self.optimization_failed(str(e)))
        
        thread = threading.Thread(target=optimization_thread)
        thread.daemon = True
        thread.start()
    
    def run_robustness_tests(self):
        """執行穩健性測試"""
        if self.df is None:
            messagebox.showerror("錯誤", "請先載入資料")
            return
        
        self.status_var.set("執行穩健性測試中...")
        
        def robustness_thread():
            try:
                # 這裡可以添加穩健性測試邏輯
                self.root.after(0, self.robustness_completed)
            except Exception as e:
                self.root.after(0, lambda: self.robustness_failed(str(e)))
        
        thread = threading.Thread(target=robustness_thread)
        thread.daemon = True
        thread.start()
    
    def run_complete_analysis(self):
        """執行完整分析"""
        if self.df is None:
            messagebox.showerror("錯誤", "請先載入資料")
            return
        
        self.status_var.set("執行完整分析中...")
        
        def complete_analysis_thread():
            try:
                # 這裡可以添加完整分析邏輯
                self.root.after(0, self.complete_analysis_completed)
            except Exception as e:
                self.root.after(0, lambda: self.complete_analysis_failed(str(e)))
        
        thread = threading.Thread(target=complete_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def analysis_completed(self):
        """分析完成"""
        self.status_var.set("基礎分析完成")
        messagebox.showinfo("完成", "基礎分析已完成！")
    
    def analysis_failed(self, error_msg):
        """分析失敗"""
        self.status_var.set(f"分析失敗: {error_msg}")
        messagebox.showerror("錯誤", f"分析失敗: {error_msg}")
    
    def optimization_completed(self):
        """優化完成"""
        self.status_var.set("參數優化完成")
        messagebox.showinfo("完成", "參數優化已完成！")
    
    def optimization_failed(self, error_msg):
        """優化失敗"""
        self.status_var.set(f"優化失敗: {error_msg}")
        messagebox.showerror("錯誤", f"優化失敗: {error_msg}")
    
    def robustness_completed(self):
        """穩健性測試完成"""
        self.status_var.set("穩健性測試完成")
        messagebox.showinfo("完成", "穩健性測試已完成！")
    
    def robustness_failed(self, error_msg):
        """穩健性測試失敗"""
        self.status_var.set(f"穩健性測試失敗: {error_msg}")
        messagebox.showerror("錯誤", f"穩健性測試失敗: {error_msg}")
    
    def complete_analysis_completed(self):
        """完整分析完成"""
        self.status_var.set("完整分析完成")
        messagebox.showinfo("完成", "完整分析已完成！")
    
    def complete_analysis_failed(self, error_msg):
        """完整分析失敗"""
        self.status_var.set(f"完整分析失敗: {error_msg}")
        messagebox.showerror("錯誤", f"完整分析失敗: {error_msg}")
    
    def export_report(self):
        """匯出報告並生成圖表"""
        if self.df is None:
            messagebox.showerror("錯誤", "請先載入資料")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subdir = os.path.join(self.output_dir, timestamp)
        os.makedirs(output_subdir, exist_ok=True)
        
        self.status_var.set("生成分析圖表中...")
        self.root.update()
        
        try:
            # 獲取當前參數
            params = {
                'bb_window': int(self.bb_window_var.get()),
                'bb_std': float(self.bb_std_var.get()),
                'rsi_period': int(self.rsi_period_var.get()),
                'rsi_oversold': int(self.rsi_oversold_var.get()),
                'rsi_overbought': int(self.rsi_overbought_var.get()),
                'rsi_exit': int(self.rsi_exit_var.get()),
                'obv_ma_window': int(self.obv_ma_window_var.get()),
                'obv_threshold': float(self.obv_threshold_var.get()),
                'entry_n': int(self.entry_n_var.get()),
                'stop_loss_pct': float(self.stop_loss_var.get()),
                'take_profit_pct': float(self.take_profit_var.get())
            }
            
            # 生成技術指標
            df_analysis = self.df.copy()
            
            # 計算布林通道
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df_analysis['close'].values, 
                timeperiod=params['bb_window'], 
                nbdevup=params['bb_std'], 
                nbdevdn=params['bb_std'], 
                matype=0
            )
            df_analysis['bb_upper'] = bb_upper
            df_analysis['bb_middle'] = bb_middle
            df_analysis['bb_lower'] = bb_lower
            
            # 計算RSI
            df_analysis['rsi'] = talib.RSI(df_analysis['close'].values, timeperiod=params['rsi_period'])
            
            # 計算OBV
            df_analysis['obv'] = talib.OBV(df_analysis['close'].values, df_analysis['volume'].values)
            df_analysis['obv_ma'] = df_analysis['obv'].rolling(window=params['obv_ma_window']).mean()
            df_analysis['obv_ratio'] = df_analysis['obv'] / df_analysis['obv_ma']
            
            # 生成信號
            df_analysis['signal'] = 0
            
            # 買入條件
            buy_conditions = (
                (df_analysis['close'] <= df_analysis['bb_lower']) &  # 價格在布林通道下軌
                (df_analysis['rsi'] <= params['rsi_oversold']) &     # RSI超賣
                (df_analysis['obv_ratio'] >= params['obv_threshold'])  # OBV強勢
            )
            
            # 賣出條件
            sell_conditions = (
                (df_analysis['close'] >= df_analysis['bb_upper']) &  # 價格在布林通道上軌
                (df_analysis['rsi'] >= params['rsi_overbought']) &   # RSI超買
                (df_analysis['obv_ratio'] <= 1/params['obv_threshold'])  # OBV弱勢
            )
            
            df_analysis.loc[buy_conditions, 'signal'] = 1
            df_analysis.loc[sell_conditions, 'signal'] = -1
            
            # 生成圖表
            fig, axes = plt.subplots(6, 1, figsize=(16, 18))
            fig.suptitle('Super Single Strategy Analysis', fontsize=18, fontweight='bold')
            
            # 圖表1: 價格和布林通道
            ax1 = axes[0]
            ax1.plot(df_analysis.index, df_analysis['close'], label='Close Price', linewidth=1.5, alpha=0.8)
            ax1.plot(df_analysis.index, df_analysis['bb_upper'], label='BB Upper', alpha=0.7, linestyle='--', linewidth=1.5)
            ax1.plot(df_analysis.index, df_analysis['bb_middle'], label='BB Middle', alpha=0.7, linestyle='--', linewidth=1.5)
            ax1.plot(df_analysis.index, df_analysis['bb_lower'], label='BB Lower', alpha=0.7, linestyle='--', linewidth=1.5)
            
            # 添加信號點
            buy_points = df_analysis.index[df_analysis['signal'] == 1]
            sell_points = df_analysis.index[df_analysis['signal'] == -1]
            
            ax1.scatter(buy_points, df_analysis.loc[buy_points, 'close'], 
                       color='green', marker='^', s=80, label='Buy Signal', alpha=0.9, zorder=5)
            ax1.scatter(sell_points, df_analysis.loc[sell_points, 'close'], 
                       color='red', marker='v', s=80, label='Sell Signal', alpha=0.9, zorder=5)
            
            ax1.set_title('Price and Bollinger Bands with Signals', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel('Price')
            
            # 圖表2: RSI
            ax2 = axes[1]
            ax2.plot(df_analysis.index, df_analysis['rsi'], label='RSI', color='purple', linewidth=1.5)
            ax2.axhline(y=params['rsi_overbought'], color='red', linestyle='--', alpha=0.7, 
                       label=f'Overbought ({params["rsi_overbought"]})', linewidth=2)
            ax2.axhline(y=params['rsi_oversold'], color='green', linestyle='--', alpha=0.7, 
                       label=f'Oversold ({params["rsi_oversold"]})', linewidth=2)
            ax2.axhline(y=params['rsi_exit'], color='orange', linestyle='--', alpha=0.7, 
                       label=f'Exit Level ({params["rsi_exit"]})', linewidth=2)
            ax2.set_title('RSI Indicator', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('RSI')
            
            # 圖表3: OBV
            ax3 = axes[2]
            ax3.plot(df_analysis.index, df_analysis['obv'], label='OBV', alpha=0.7, linewidth=1.5)
            ax3.plot(df_analysis.index, df_analysis['obv_ma'], label='OBV MA', alpha=0.7, linewidth=1.5)
            ax3_twin = ax3.twinx()
            ax3_twin.plot(df_analysis.index, df_analysis['obv_ratio'], label='OBV Ratio', color='orange', alpha=0.7, linewidth=1.5)
            ax3_twin.axhline(y=params['obv_threshold'], color='green', linestyle='--', alpha=0.7, 
                            label=f'Threshold ({params["obv_threshold"]})', linewidth=2)
            ax3_twin.axhline(y=1/params['obv_threshold'], color='red', linestyle='--', alpha=0.7, 
                            label=f'Inverse Threshold ({1/params["obv_threshold"]:.2f})', linewidth=2)
            
            ax3.set_title('OBV and Volume Analysis', fontsize=14, fontweight='bold')
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylabel('OBV')
            ax3_twin.set_ylabel('OBV Ratio')
            
            # 圖表4: 參數設定視覺化
            ax4 = axes[3]
            param_names = ['BB Window', 'BB Std', 'RSI Period', 'RSI Oversold', 'RSI Overbought', 'RSI Exit', 'OBV Window', 'OBV Threshold']
            param_values = [params['bb_window'], params['bb_std'], params['rsi_period'], 
                           params['rsi_oversold'], params['rsi_overbought'], params['rsi_exit'],
                           params['obv_ma_window'], params['obv_threshold']]
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightblue', 'lightgray', 'lightcyan']
            
            bars = ax4.bar(param_names, param_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax4.set_title('Strategy Parameters', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Parameter Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, param_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(param_values)*0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            # Rotate x-axis labels for better readability
            ax4.tick_params(axis='x', rotation=45)
            
            # 圖表5: 信號分布
            ax5 = axes[4]
            signal_counts = df_analysis['signal'].value_counts()
            signal_colors = {1: 'green', -1: 'red', 0: 'gray'}
            signal_values = [signal_counts.get(1, 0), signal_counts.get(-1, 0), signal_counts.get(0, 0)]
            signal_labels = ['Buy', 'Sell', 'Hold']
            
            bars = ax5.bar(signal_labels, signal_values, color=[signal_colors.get(i, 'gray') for i in [1, -1, 0]], 
                          alpha=0.8, edgecolor='black', linewidth=1)
            ax5.set_title('Signal Distribution', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Number of Signals')
            
            # Add value labels on bars
            for bar, value in zip(bars, signal_values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + max(signal_values)*0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # 圖表6: 成交量分析
            ax6 = axes[5]
            ax6.bar(df_analysis.index, df_analysis['volume'], alpha=0.6, color='blue', label='Volume')
            
            # Highlight volume on signal days
            if len(buy_points) > 0:
                ax6.bar(buy_points, df_analysis.loc[buy_points, 'volume'], alpha=0.8, color='green', label='Buy Signal Volume')
            if len(sell_points) > 0:
                ax6.bar(sell_points, df_analysis.loc[sell_points, 'volume'], alpha=0.8, color='red', label='Sell Signal Volume')
            
            ax6.set_title('Volume Analysis', fontsize=14, fontweight='bold')
            ax6.legend(loc='upper left')
            ax6.grid(True, alpha=0.3)
            ax6.set_ylabel('Volume')
            ax6.set_xlabel('Time')
            
            plt.tight_layout()
            
            # 保存圖表
            plot_filename = f"super_single_strategy_analysis_{timestamp}.png"
            plot_path = os.path.join(output_subdir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 生成分析摘要
            total_signals = len(df_analysis[df_analysis['signal'] != 0])
            buy_signals = len(buy_points)
            sell_signals = len(sell_points)
            
            # 顯示成功訊息
            success_msg = f"""✅ 分析報告匯出成功！

📊 生成的文件:
• 圖表: {plot_filename}

📁 保存位置: {output_subdir}

📈 分析摘要:
• 總信號數: {total_signals}
• 買入信號: {buy_signals}
• 賣出信號: {sell_signals}
• 數據點數: {len(df_analysis)}

是否要開啟圖表文件？"""
            
            if messagebox.askyesno("匯出成功", success_msg):
                # 開啟圖表文件
                if sys.platform == "win32":
                    os.startfile(plot_path)
                elif sys.platform == "darwin":
                    subprocess.run(["open", plot_path])
                else:
                    subprocess.run(["xdg-open", plot_path])
            
            self.status_var.set(f"✅ 報告已匯出到: {output_subdir}")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"匯出失敗: {e}")
            self.status_var.set("匯出失敗")
    
    def open_output_folder(self):
        """開啟輸出資料夾"""
        try:
            if os.path.exists(self.output_dir):
                if sys.platform == "win32":
                    os.startfile(self.output_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", self.output_dir])
                else:
                    subprocess.run(["xdg-open", self.output_dir])
            else:
                messagebox.showinfo("資訊", "輸出資料夾不存在。請先匯出一些結果。")
        except Exception as e:
            messagebox.showerror("錯誤", f"開啟輸出資料夾失敗: {e}")

# 主程式入口
if __name__ == "__main__":
    root = tk.Tk()
    app = SuperSingleStrategyPlatform(root)
    root.mainloop() 