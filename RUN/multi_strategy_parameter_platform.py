"""
Multi-Strategy Parameter Platform with Complete Export Functions
A comprehensive GUI platform for adjusting multi-strategy parameters and exporting all results
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from datetime import datetime
import subprocess
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Import multi-strategy modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'multi_strategy_system'))
from multi_strategy_system.strategy_combiner import (
    run_multi_strategy_analysis, 
    get_default_params, 
    get_default_weights
)
from multi_strategy_system.performance_analyzer import analyze_strategy_performance

class MultiStrategyParameterPlatform:
    def __init__(self, root, initial_params=None):
        self.root = root
        self.root.title("Multi-Strategy Parameter Platform - Complete Edition")
        self.root.geometry("1400x900")
        
        # Load default parameters
        self.default_params = {
            # Strategy parameters
            'rsi_window': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_window': 20,
            'bb_std': 2.0,
            'obv_window': 10,
            'obv_threshold': 1.2,
            
            # Decision thresholds
            'buy_threshold': 70,
            'sell_threshold': 30,
            
            # Strategy weights
            'rsi_weight': 0.4,
            'bollinger_bands_weight': 0.35,
            'obv_weight': 0.25,
            
            # Performance settings
            'initial_capital': 100000,
            'position_size': 0.1
        }
        
        # Initialize data source tracking
        self.data_source_file = None
        if initial_params is not None:
            self.default_params.update(initial_params)
        self.current_params = self.default_params.copy()
        self.analysis_results = None
        self.df = None
        self.waiting_window = None
        
        # Parameter optimization settings
        self.optimization_history = []
        self.best_params = None
        self.best_performance = None
        
        # Create output directory
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.create_widgets()
        self.load_sample_data()
        self.update_system_info()
        
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Multi-Strategy Parameter Platform - Complete Edition", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Strategy Parameters Tab
        strategy_frame = ttk.Frame(notebook, padding="10")
        notebook.add(strategy_frame, text="Strategy Parameters")
        self.create_strategy_parameters_tab(strategy_frame)
        
        # Weights Tab
        weights_frame = ttk.Frame(notebook, padding="10")
        notebook.add(weights_frame, text="Strategy Weights")
        self.create_weights_tab(weights_frame)
        
        # Decision Thresholds Tab
        decision_frame = ttk.Frame(notebook, padding="10")
        notebook.add(decision_frame, text="Decision Thresholds")
        self.create_decision_thresholds_tab(decision_frame)
        
        # Performance Settings Tab
        performance_frame = ttk.Frame(notebook, padding="10")
        notebook.add(performance_frame, text="Performance Settings")
        self.create_performance_settings_tab(performance_frame)
        
        # Results Preview Tab
        results_frame = ttk.Frame(notebook, padding="10")
        notebook.add(results_frame, text="Results Preview")
        self.create_results_preview_tab(results_frame)
        
        # Parameter Optimization Tab
        optimization_frame = ttk.Frame(notebook, padding="10")
        notebook.add(optimization_frame, text="Parameter Optimization")
        self.create_optimization_tab(optimization_frame)
        
        # System Info Tab
        info_frame = ttk.Frame(notebook, padding="10")
        notebook.add(info_frame, text="System Info")
        self.create_system_info_tab(info_frame)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=20)
        
        # Buttons
        ttk.Button(button_frame, text="Load Parameters", command=self.load_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Parameters", command=self.save_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export All Results", command=self.export_all_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Optimize Parameters", command=self.optimize_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Send to Demo", command=self.send_params_to_demo).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Open Output Folder", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_strategy_parameters_tab(self, parent):
        """Create strategy parameters tab"""
        # RSI Parameters
        rsi_frame = ttk.LabelFrame(parent, text="RSI Strategy Parameters", padding="10")
        rsi_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        rsi_params = {
            'rsi_window': ('RSI Window', 5, 30),
            'rsi_oversold': ('RSI Oversold Level', 10, 40),
            'rsi_overbought': ('RSI Overbought Level', 60, 90)
        }
        
        self.create_param_widgets(rsi_frame, rsi_params, 0, 0)
        
        # Bollinger Bands Parameters
        bb_frame = ttk.LabelFrame(parent, text="Bollinger Bands Strategy Parameters", padding="10")
        bb_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        bb_params = {
            'bb_window': ('BB Window', 5, 50),
            'bb_std': ('BB Standard Deviation', 1.0, 4.0)
        }
        
        self.create_param_widgets(bb_frame, bb_params, 0, 0)
        
        # OBV Parameters
        obv_frame = ttk.LabelFrame(parent, text="OBV Strategy Parameters", padding="10")
        obv_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        obv_params = {
            'obv_window': ('OBV Window', 5, 30),
            'obv_threshold': ('OBV Threshold', 1.0, 2.0)
        }
        
        self.create_param_widgets(obv_frame, obv_params, 0, 0)
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
    def create_weights_tab(self, parent):
        """Create strategy weights tab"""
        # Weights frame
        weights_frame = ttk.LabelFrame(parent, text="Strategy Weights (Must Sum to 1.0)", padding="20")
        weights_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=20)
        
        # Weight parameters
        weight_params = {
            'rsi_weight': ('RSI Weight', 0.0, 1.0),
            'bollinger_bands_weight': ('Bollinger Bands Weight', 0.0, 1.0),
            'obv_weight': ('OBV Weight', 0.0, 1.0)
        }
        
        self.create_param_widgets(weights_frame, weight_params, 0, 0)
        
        # Weight validation
        validation_frame = ttk.Frame(weights_frame)
        validation_frame.grid(row=len(weight_params), column=0, columnspan=3, pady=(20, 0))
        
        self.weight_sum_var = tk.StringVar(value="Total: 1.00")
        ttk.Label(validation_frame, textvariable=self.weight_sum_var, font=("Arial", 12, "bold")).pack()
        
        # Auto-balance button
        ttk.Button(validation_frame, text="Auto-Balance Weights", command=self.auto_balance_weights).pack(pady=(10, 0))
        
        # Bind weight changes to validation
        for param in weight_params.keys():
            if param in self.param_widgets:
                self.param_widgets[param][0].trace('w', self.validate_weights)
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
    def create_decision_thresholds_tab(self, parent):
        """Create decision thresholds tab"""
        # Thresholds frame
        thresholds_frame = ttk.LabelFrame(parent, text="Decision Thresholds", padding="20")
        thresholds_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=20)
        
        # Threshold parameters
        threshold_params = {
            'buy_threshold': ('Buy Threshold (0-100)', 50, 90),
            'sell_threshold': ('Sell Threshold (0-100)', 10, 50)
        }
        
        self.create_param_widgets(thresholds_frame, threshold_params, 0, 0)
        
        # Threshold validation
        validation_frame = ttk.Frame(thresholds_frame)
        validation_frame.grid(row=len(threshold_params), column=0, columnspan=3, pady=(20, 0))
        
        self.threshold_info_var = tk.StringVar(value="Buy threshold should be higher than sell threshold")
        ttk.Label(validation_frame, textvariable=self.threshold_info_var, font=("Arial", 10)).pack()
        
        # Bind threshold changes to validation
        for param in threshold_params.keys():
            if param in self.param_widgets:
                self.param_widgets[param][0].trace('w', self.validate_thresholds)
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
    def create_performance_settings_tab(self, parent):
        """Create performance settings tab"""
        # Performance frame
        performance_frame = ttk.LabelFrame(parent, text="Performance Analysis Settings", padding="20")
        performance_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=20)
        
        # Performance parameters
        performance_params = {
            'initial_capital': ('Initial Capital ($)', 10000, 1000000),
            'position_size': ('Position Size (0.01-1.0)', 0.01, 1.0)
        }
        
        self.create_param_widgets(performance_frame, performance_params, 0, 0)
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
    def create_results_preview_tab(self, parent):
        """Create results preview tab"""
        # Preview frame
        preview_frame = ttk.LabelFrame(parent, text="Analysis Results Preview", padding="10")
        preview_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Create text widget for results
        self.results_text = tk.Text(preview_frame, height=20, width=80, font=("Courier", 9))
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Initial message
        self.results_text.insert(tk.END, "No analysis results yet. Run analysis to see results here.\n")
        
    def create_system_info_tab(self, parent):
        """Create system information tab"""
        # Data source frame
        data_frame = ttk.LabelFrame(parent, text="Data Source Information", padding="10")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        # Data source info
        self.data_source_text = tk.Text(data_frame, height=8, width=60, wrap=tk.WORD)
        self.data_source_text.pack(fill='both', expand=True)
        
        # Initial parameters frame
        params_frame = ttk.LabelFrame(parent, text="Initial Parameters", padding="10")
        params_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0), pady=(0, 10))
        
        # Initial parameters info
        self.initial_params_text = tk.Text(params_frame, height=8, width=60, wrap=tk.WORD)
        self.initial_params_text.pack(fill='both', expand=True)
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Update info after data loading
        self.update_system_info()
    
    def update_system_info(self):
        """Update system information display"""
        # Update data source info
        self.data_source_text.delete(1.0, tk.END)
        if hasattr(self, 'data_source_file') and self.data_source_file:
            if self.data_source_file == "SYNTHETIC_DATA":
                info = f"""Data Type: Synthetic Demo Data
Rows: {len(self.df) if self.df is not None else 'N/A'}
Date Range: {self.df.index[0].strftime('%Y-%m-%d') if self.df is not None else 'N/A'} to {self.df.index[-1].strftime('%Y-%m-%d') if self.df is not None else 'N/A'}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            else:
                info = f"""Data File: {self.data_source_file}
Rows: {len(self.df) if self.df is not None else 'N/A'}
Date Range: {self.df.index[0].strftime('%Y-%m-%d %H:%M') if self.df is not None else 'N/A'} to {self.df.index[-1].strftime('%Y-%m-%d %H:%M') if self.df is not None else 'N/A'}
Loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        else:
            info = "No data loaded"
        
        self.data_source_text.insert(tk.END, info)
        
        # Update initial parameters info
        self.initial_params_text.delete(1.0, tk.END)
        params_info = "Initial Parameters:\n\n"
        for key, value in self.default_params.items():
            params_info += f"{key}: {value}\n"
        self.initial_params_text.insert(tk.END, params_info)
    
    def create_optimization_tab(self, parent):
        """Create parameter optimization tab"""
        # Optimization settings frame
        settings_frame = ttk.LabelFrame(parent, text="Optimization Settings", padding="10")
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        # Optimization parameters
        opt_params = {
            'optimization_iterations': ('Iterations', 5, 20),
            'parameter_step_size': ('Step Size', 0.1, 1.0),
            'performance_threshold': ('Performance Threshold', 0.5, 0.9)
        }
        
        self.create_param_widgets(settings_frame, opt_params, 0, 0)
        
        # Optimization history frame
        history_frame = ttk.LabelFrame(parent, text="Optimization History", padding="10")
        history_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        
        # History text widget
        self.optimization_text = tk.Text(history_frame, height=15, width=50)
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.optimization_text.yview)
        self.optimization_text.configure(yscrollcommand=scrollbar.set)
        
        self.optimization_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Best performance frame
        best_frame = ttk.LabelFrame(parent, text="Best Performance", padding="10")
        best_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.best_performance_var = tk.StringVar(value="No optimization run yet")
        ttk.Label(best_frame, textvariable=self.best_performance_var, font=("Arial", 12, "bold")).pack()
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
    def create_param_widgets(self, parent, params, start_row, start_col):
        """Create parameter widgets for a given set of parameters"""
        row = start_row
        
        for param, (label, min_val, max_val) in params.items():
            ttk.Label(parent, text=label).grid(row=row, column=start_col, sticky=tk.W, padx=(0, 10), pady=5)
            
            # Use default value from current_params, fallback to min_val if not found
            default_value = self.current_params.get(param, min_val)
            var = tk.DoubleVar(value=default_value)
            scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=start_col+1, sticky=(tk.W, tk.E), padx=(0, 10), pady=5)
            
            entry = ttk.Entry(parent, textvariable=var, width=10)
            entry.grid(row=row, column=start_col+2, padx=(0, 20), pady=5)
            # 強制同步 Entry 顯示初始值
            entry.delete(0, 'end')
            entry.insert(0, str(default_value))
            
            if not hasattr(self, 'param_widgets'):
                self.param_widgets = {}
            self.param_widgets[param] = (var, scale, entry)
            
            row += 1
        
        # Configure column weights
        parent.columnconfigure(start_col+1, weight=1)
        
    def load_sample_data(self):
        """Load sample data for analysis"""
        try:
            # Try to load TXF data from multiple possible locations
            possible_paths = [
                "multi_strategy_system/TXF1_Minute_2020-01-01_2025-07-04.txt",
                "TXF1_Minute_2020-01-01_2025-07-04.txt",
                "multi_strategy_system/TXF1_Minute_2020-01-01_2025-06-16.txt",
                "TXF1_Minute_2020-01-01_2025-06-16.txt"
            ]
            
            self.data_source_file = None
            for file_path in possible_paths:
                if os.path.exists(file_path):
                    try:
                        # Try comma-separated first, then tab-separated
                        try:
                            df = pd.read_csv(file_path)
                        except:
                            df = pd.read_csv(file_path, sep='\t')
                        
                        # Rename columns
                        column_mapping = {
                            'Date': 'date', 'Time': 'time', 'Open': 'open',
                            'High': 'high', 'Low': 'low', 'Close': 'close', 
                            'Volume': 'volume', 'TotalVolume': 'volume'
                        }
                        df = df.rename(columns=column_mapping)
                        
                        # Combine date and time
                        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                        df = df.set_index('datetime')
                        df = df[['open', 'high', 'low', 'close', 'volume']]
                        
                        # Take last 1000 rows
                        self.df = df.tail(1000).copy()
                        self.data_source_file = file_path
                        self.status_var.set(f"Loaded {len(self.df)} rows from: {file_path}")
                        return
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")
                        continue
            
            # If no data file found, generate synthetic data
            self.generate_synthetic_data()
                
        except Exception as e:
            self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """Generate synthetic data for demo"""
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
        np.random.seed(42)
        
        returns = np.random.normal(0, 0.001, 1000)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.0005, 1000)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, 1000))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, 1000))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        self.data_source_file = "SYNTHETIC_DATA"
        self.status_var.set(f"Generated {len(self.df)} rows of synthetic data")
    
    def validate_weights(self, *args):
        """Validate that weights sum to 1.0"""
        try:
            rsi_weight = self.param_widgets['rsi_weight'][0].get()
            bb_weight = self.param_widgets['bollinger_bands_weight'][0].get()
            obv_weight = self.param_widgets['obv_weight'][0].get()
            
            total = rsi_weight + bb_weight + obv_weight
            
            if abs(total - 1.0) < 0.01:
                self.weight_sum_var.set("Total: 1.00 ✓")
            else:
                self.weight_sum_var.set(f"Total: {total:.2f} ✗ (Should be 1.00)")
                
        except (KeyError, tk.TclError):
            pass
    
    def validate_thresholds(self, *args):
        """Validate that buy threshold is higher than sell threshold"""
        try:
            buy_threshold = self.param_widgets['buy_threshold'][0].get()
            sell_threshold = self.param_widgets['sell_threshold'][0].get()
            
            if buy_threshold > sell_threshold:
                self.threshold_info_var.set("✓ Valid thresholds")
            else:
                self.threshold_info_var.set("✗ Buy threshold should be higher than sell threshold")
                
        except (KeyError, tk.TclError):
            pass
    
    def auto_balance_weights(self):
        """Automatically balance weights to sum to 1.0"""
        try:
            rsi_weight = self.param_widgets['rsi_weight'][0].get()
            bb_weight = self.param_widgets['bollinger_bands_weight'][0].get()
            obv_weight = self.param_widgets['obv_weight'][0].get()
            
            total = rsi_weight + bb_weight + obv_weight
            
            if total > 0:
                # Normalize weights
                self.param_widgets['rsi_weight'][0].set(rsi_weight / total)
                self.param_widgets['bollinger_bands_weight'][0].set(bb_weight / total)
                self.param_widgets['obv_weight'][0].set(obv_weight / total)
                
                self.status_var.set("Weights auto-balanced")
            else:
                # Set equal weights if all are zero
                self.param_widgets['rsi_weight'][0].set(0.33)
                self.param_widgets['bollinger_bands_weight'][0].set(0.33)
                self.param_widgets['obv_weight'][0].set(0.34)
                
                self.status_var.set("Weights set to equal distribution")
                
        except (KeyError, tk.TclError) as e:
            messagebox.showerror("Error", f"Failed to balance weights: {e}")
    
    def get_current_parameters(self):
        """Get current parameters from widgets"""
        params = {}
        for param, (var, scale, entry) in self.param_widgets.items():
            try:
                value = var.get()
                # Ensure integer parameters are properly converted
                if 'window' in param:
                    params[param] = int(value)
                else:
                    params[param] = value
            except tk.TclError:
                params[param] = self.default_params.get(param, 0)
        return params
    
    def run_analysis(self):
        """Run analysis in background thread"""
        def run_analysis_thread():
            try:
                # Show waiting window
                self.show_waiting_window()
                
                self.status_var.set("Running analysis...")
                
                # Get current parameters
                params = self.get_current_parameters()
                
                # Validate parameters
                if not self.validate_analysis_parameters(params):
                    self.hide_waiting_window()
                    return
                
                # Run analysis
                self.analysis_results = run_multi_strategy_analysis(self.df, params)
                
                # Update results preview
                self.update_results_preview()
                
                # 產生多策略分析圖和績效分析圖
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.output_dir
                self.plot_strategy_analysis(output_dir, timestamp)
                self.plot_performance_analysis(output_dir, timestamp)
                
                self.status_var.set("Analysis completed successfully (plots saved)")
                
                # Hide waiting window
                self.hide_waiting_window()
                
            except Exception as e:
                self.status_var.set(f"Analysis failed: {e}")
                messagebox.showerror("Error", f"Analysis failed: {e}")
                # Hide waiting window on error
                self.hide_waiting_window()
        
        # Run in background thread
        thread = threading.Thread(target=run_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def show_waiting_window(self):
        """Show a waiting window"""
        self.waiting_window = tk.Toplevel(self.root)
        self.waiting_window.title("請稍候")
        self.waiting_window.geometry("300x150")
        self.waiting_window.resizable(False, False)
        
        # Center the window
        self.waiting_window.transient(self.root)
        self.waiting_window.grab_set()
        
        # Add waiting message
        waiting_frame = tk.Frame(self.waiting_window)
        waiting_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        waiting_label = tk.Label(waiting_frame, text="請稍候...", font=("Arial", 14, "bold"))
        waiting_label.pack(pady=10)
        
        progress_label = tk.Label(waiting_frame, text="正在執行策略分析", font=("Arial", 10))
        progress_label.pack(pady=5)
        
        # Add a simple progress bar
        self.progress_bar = tk.Frame(waiting_frame, height=4, bg='lightgray')
        self.progress_bar.pack(fill='x', pady=10)
        
        # Start progress animation
        self.animate_progress()
        
        # Center window on screen
        self.waiting_window.update_idletasks()
        x = (self.waiting_window.winfo_screenwidth() // 2) - (300 // 2)
        y = (self.waiting_window.winfo_screenheight() // 2) - (150 // 2)
        self.waiting_window.geometry(f"300x150+{x}+{y}")
    
    def hide_waiting_window(self):
        """Hide the waiting window"""
        if hasattr(self, 'waiting_window') and self.waiting_window:
            try:
                self.waiting_window.destroy()
                self.waiting_window = None
            except:
                pass
    
    def animate_progress(self):
        """Animate the progress bar"""
        if hasattr(self, 'waiting_window') and self.waiting_window:
            try:
                # Create a moving progress indicator
                progress_indicator = tk.Frame(self.progress_bar, width=50, height=4, bg='blue')
                progress_indicator.pack(side='left')
                
                def move_progress():
                    if hasattr(self, 'waiting_window') and self.waiting_window:
                        try:
                            # Move the indicator
                            progress_indicator.pack_forget()
                            progress_indicator.pack(side='right')
                            self.waiting_window.after(500, lambda: progress_indicator.pack(side='left'))
                            self.waiting_window.after(1000, move_progress)
                        except:
                            pass
                
                move_progress()
            except:
                pass
    
    def validate_analysis_parameters(self, params):
        """Validate analysis parameters"""
        # Validate weights
        total_weight = params['rsi_weight'] + params['bollinger_bands_weight'] + params['obv_weight']
        if abs(total_weight - 1.0) > 0.01:
            messagebox.showwarning("Warning", f"Weights sum to {total_weight:.2f}, not 1.0. Consider auto-balancing.")
        
        # Validate thresholds
        if params['buy_threshold'] <= params['sell_threshold']:
            messagebox.showerror("Error", "Buy threshold must be higher than sell threshold")
            return False
        
        return True
    
    def update_results_preview(self):
        """Update results preview tab"""
        if self.analysis_results is None:
            return
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Add results summary
        summary = self.generate_results_summary()
        self.results_text.insert(tk.END, summary)
    
    def generate_results_summary(self):
        """Generate results summary text"""
        if self.analysis_results is None:
            return "No analysis results available.\n"
        
        results = self.analysis_results
        decisions = results['decisions']
        
        summary = f"""
MULTI-STRATEGY ANALYSIS RESULTS
{'='*50}

DATA SOURCE:
{'-'*15}
"""
        
        # Add data source information
        if hasattr(self, 'data_source_file') and self.data_source_file:
            if self.data_source_file == "SYNTHETIC_DATA":
                summary += f"  Data Type: Synthetic Demo Data\n"
                summary += f"  Rows: {len(self.df)}\n"
                summary += f"  Date Range: {self.df.index[0].strftime('%Y-%m-%d')} to {self.df.index[-1].strftime('%Y-%m-%d')}\n"
            else:
                summary += f"  Data File: {self.data_source_file}\n"
                summary += f"  Rows: {len(self.df)}\n"
                summary += f"  Date Range: {self.df.index[0].strftime('%Y-%m-%d %H:%M')} to {self.df.index[-1].strftime('%Y-%m-%d %H:%M')}\n"
        else:
            summary += f"  Data Source: Unknown\n"
            summary += f"  Rows: {len(self.df)}\n"
        
        summary += f"""
PARAMETERS USED:
{'-'*20}
"""
        
        # Add parameters
        params = results['params_used']
        for key, value in params.items():
            summary += f"  {key}: {value}\n"
        
        summary += f"""
STRATEGY WEIGHTS:
{'-'*20}
"""
        
        # Add weights
        weights = results['weights_used']
        for strategy, weight in weights.items():
            summary += f"  {strategy.replace('_', ' ').title()}: {weight:.1%}\n"
        
        summary += f"""
DECISION STATISTICS:
{'-'*20}
"""
        
        # Add decision statistics
        decision_counts = decisions.value_counts()
        for decision, count in decision_counts.items():
            percentage = (count / len(decisions)) * 100
            summary += f"  {decision}: {count} ({percentage:.1f}%)\n"
        
        summary += f"""
SCORE STATISTICS:
{'-'*20}
"""
        
        # Add score statistics
        combined_score = results['combined_score']
        summary += f"  Combined Score Mean: {combined_score.mean():.2f}\n"
        summary += f"  Combined Score Std: {combined_score.std():.2f}\n"
        summary += f"  Combined Score Min: {combined_score.min():.2f}\n"
        summary += f"  Combined Score Max: {combined_score.max():.2f}\n"
        
        summary += f"""
INDIVIDUAL STRATEGY STATISTICS:
{'-'*30}
"""
        
        # Add individual strategy statistics
        individual_scores = results['individual_scores']
        for name, score in individual_scores.items():
            summary += f"  {name.replace('_', ' ').title()}:\n"
            summary += f"    Mean: {score.mean():.2f}\n"
            summary += f"    Std: {score.std():.2f}\n"
        
        return summary
    
    def export_all_results(self):
        """Export all analysis results"""
        if self.analysis_results is None:
            messagebox.showwarning("Warning", "No analysis results to export. Please run analysis first.")
            return
        
        try:
            self.status_var.set("Exporting results...")
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create output directory
            output_subdir = os.path.join(self.output_dir, timestamp)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            # Export parameters
            self.export_parameters(output_subdir, timestamp)
            
            # Export Excel results
            self.export_excel_results(output_subdir, timestamp)
            
            # Export charts
            self.export_charts(output_subdir, timestamp)
            
            # Export performance analysis
            self.export_performance_analysis(output_subdir, timestamp)
            
            # Export summary report
            self.export_summary_report(output_subdir, timestamp)
            
            self.status_var.set(f"All results exported to {output_subdir}")
            messagebox.showinfo("Success", f"All results exported successfully!\nLocation: {output_subdir}")
            
        except Exception as e:
            self.status_var.set(f"Export failed: {e}")
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def export_parameters(self, output_dir, timestamp):
        """Export parameters to JSON"""
        params = self.get_current_parameters()
        params['_metadata'] = {
            'exported_at': datetime.now().isoformat(),
            'platform': 'Multi-Strategy Parameter Platform',
            'data_points': len(self.df)
        }
        
        file_path = os.path.join(output_dir, f"strategy_params_{timestamp}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
    
    def export_excel_results(self, output_dir, timestamp):
        """Export analysis results to Excel"""
        results_df = self.df.copy()
        
        # Add individual scores
        for name, score in self.analysis_results['individual_scores'].items():
            results_df[f'{name}_score'] = score
        
        # Add combined score and decisions
        results_df['combined_score'] = self.analysis_results['combined_score']
        results_df['decision'] = self.analysis_results['decisions']
        
        # Add parameters info
        params = self.analysis_results['params_used']
        weights = self.analysis_results['weights_used']
        
        # Create Excel writer
        file_path = os.path.join(output_dir, f"multi_strategy_results_{timestamp}.xlsx")
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Main results
            results_df.to_excel(writer, sheet_name='Analysis_Results', index=True)
            
            # Parameters summary
            params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            # Weights summary
            weights_df = pd.DataFrame(list(weights.items()), columns=['Strategy', 'Weight'])
            weights_df.to_excel(writer, sheet_name='Weights', index=False)
            
            # Decision statistics
            decision_stats = self.analysis_results['decisions'].value_counts()
            decision_stats_df = pd.DataFrame(decision_stats).reset_index()
            decision_stats_df.columns = ['Decision', 'Count']
            decision_stats_df['Percentage'] = decision_stats_df['Count'] / len(self.analysis_results['decisions']) * 100
            decision_stats_df.to_excel(writer, sheet_name='Decision_Stats', index=False)
    
    def export_charts(self, output_dir, timestamp):
        """Export analysis charts"""
        # Strategy analysis chart
        self.plot_strategy_analysis(output_dir, timestamp)
        
        # Performance analysis chart
        self.plot_performance_analysis(output_dir, timestamp)
    
    def plot_strategy_analysis(self, output_dir, timestamp):
        """Plot strategy analysis results"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle('Multi-Strategy Analysis Results', fontsize=16)
        
        # Plot 1: Price and decisions
        ax1 = axes[0]
        ax1.plot(self.df.index, self.df['close'], label='Close Price', alpha=0.7)
        
        # Add decision markers
        decisions = self.analysis_results['decisions']
        # Ensure decisions align with df index
        decisions = pd.Series(decisions.values, index=self.df.index)
        buy_points = self.df.index[decisions == 'Buy']
        sell_points = self.df.index[decisions == 'Sell']
        
        ax1.scatter(buy_points, self.df.loc[buy_points, 'close'], 
                   color='green', marker='^', s=50, label='Buy Signal', alpha=0.8)
        ax1.scatter(sell_points, self.df.loc[sell_points, 'close'], 
                   color='red', marker='v', s=50, label='Sell Signal', alpha=0.8)
        
        ax1.set_title('Price and Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual strategy scores
        ax2 = axes[1]
        individual_scores = self.analysis_results['individual_scores']
        for name, score in individual_scores.items():
            ax2.plot(self.df.index, score, label=name.replace('_', ' ').title(), alpha=0.8)
        
        ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
        ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
        ax2.set_title('Individual Strategy Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Combined score
        ax3 = axes[2]
        combined_score = self.analysis_results['combined_score']
        ax3.plot(self.df.index, combined_score, label='Combined Score', linewidth=2, color='purple')
        ax3.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
        ax3.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
        ax3.fill_between(self.df.index, 70, 100, alpha=0.2, color='green', label='Buy Zone')
        ax3.fill_between(self.df.index, 0, 30, alpha=0.2, color='red', label='Sell Zone')
        ax3.set_title('Combined Strategy Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Plot 4: Volume
        ax4 = axes[3]
        ax4.bar(self.df.index, self.df['volume'], alpha=0.6, color='blue', label='Volume')
        ax4.set_title('Volume')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        file_path = os.path.join(output_dir, f"multi_strategy_analysis_{timestamp}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_analysis(self, output_dir, timestamp):
        """Plot performance analysis"""
        # 資料檢查
        if self.df is None or len(self.df) < 2:
            print("[Error] No valid data for performance analysis! df shape:", None if self.df is None else self.df.shape)
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No valid data for performance analysis!', ha='center', va='center', fontsize=16)
            plt.title('Performance Analysis Error')
            plt.axis('off')
            file_path = os.path.join(output_dir, f"performance_analysis_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.show()
            return
        if self.analysis_results is None or 'decisions' not in self.analysis_results or len(self.analysis_results['decisions']) < 2:
            print("[Error] No valid decisions for performance analysis!")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No valid decisions for performance analysis!', ha='center', va='center', fontsize=16)
            plt.title('Performance Analysis Error')
            plt.axis('off')
            file_path = os.path.join(output_dir, f"performance_analysis_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.show()
            return
        # Calculate performance metrics
        params = self.get_current_parameters()
        performance_metrics = analyze_strategy_performance(
            self.df, 
            self.analysis_results['decisions'],
            params['initial_capital'],
            params['position_size']
        )
        # Create performance analyzer and plot
        from multi_strategy_system.performance_analyzer import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer(params['initial_capital'])
        file_path = os.path.join(output_dir, f"performance_analysis_{timestamp}.png")
        analyzer.plot_performance_analysis(self.df, performance_metrics, file_path)
    
    def export_performance_analysis(self, output_dir, timestamp):
        """Export performance analysis to Excel"""
        params = self.get_current_parameters()
        performance_metrics = analyze_strategy_performance(
            self.df, 
            self.analysis_results['decisions'],
            params['initial_capital'],
            params['position_size']
        )
        
        # Create performance summary
        performance_summary = {
            'Metric': [
                'Total Return (%)',
                'Maximum Drawdown (%)',
                'Sharpe Ratio',
                'Calmar Ratio',
                'Win Rate (%)',
                'Annual Volatility (%)',
                'Average Win (%)',
                'Average Loss (%)',
                'Final Equity ($)',
                'Max Equity ($)',
                'Min Equity ($)',
                'Peak Date',
                'Trough Date'
            ],
            'Value': [
                performance_metrics['total_return_pct'],
                performance_metrics['mdd_pct'],
                performance_metrics['sharpe_ratio'],
                performance_metrics['calmar_ratio'],
                performance_metrics['win_rate_pct'],
                performance_metrics['annual_volatility_pct'],
                performance_metrics['avg_win_pct'],
                performance_metrics['avg_loss_pct'],
                performance_metrics['final_equity'],
                performance_metrics['max_equity'],
                performance_metrics['min_equity'],
                performance_metrics['peak_date'].strftime('%Y-%m-%d'),
                performance_metrics['trough_date'].strftime('%Y-%m-%d')
            ]
        }
        
        performance_df = pd.DataFrame(performance_summary)
        
        # Save to Excel
        file_path = os.path.join(output_dir, f"performance_analysis_{timestamp}.xlsx")
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            performance_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
            
            # Add equity curve
            equity_df = pd.DataFrame({
                'Date': performance_metrics['equity_curve'].index,
                'Equity': performance_metrics['equity_curve'].values
            })
            equity_df.to_excel(writer, sheet_name='Equity_Curve', index=False)
    
    def export_summary_report(self, output_dir, timestamp):
        """Export summary report"""
        # Add data source information at the beginning
        report_header = f"""MULTI-STRATEGY ANALYSIS SUMMARY REPORT
{'='*50}

GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SOURCE INFORMATION:
{'-'*30}
"""
        
        if hasattr(self, 'data_source_file') and self.data_source_file:
            if self.data_source_file == "SYNTHETIC_DATA":
                report_header += f"""Data Type: Synthetic Demo Data
Rows: {len(self.df)}
Date Range: {self.df.index[0].strftime('%Y-%m-%d')} to {self.df.index[-1].strftime('%Y-%m-%d')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            else:
                report_header += f"""Data File: {self.data_source_file}
Rows: {len(self.df)}
Date Range: {self.df.index[0].strftime('%Y-%m-%d %H:%M')} to {self.df.index[-1].strftime('%Y-%m-%d %H:%M')}
Loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:
            report_header += "Data Source: Unknown\n"
        
        report_header += "\n"
        
        summary = self.generate_results_summary()
        
        # Add performance metrics if available
        try:
            params = self.get_current_parameters()
            performance_metrics = analyze_strategy_performance(
                self.df, 
                self.analysis_results['decisions'],
                params['initial_capital'],
                params['position_size']
            )
            
            summary += f"""
PERFORMANCE METRICS:
{'-'*20}
Total Return: {performance_metrics['total_return_pct']:.2f}%
Maximum Drawdown: {performance_metrics['mdd_pct']:.2f}%
Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}
Calmar Ratio: {performance_metrics['calmar_ratio']:.2f}
Win Rate: {performance_metrics['win_rate_pct']:.1f}%
Annual Volatility: {performance_metrics['annual_volatility_pct']:.2f}%
Final Equity: ${performance_metrics['final_equity']:,.0f}
"""
        except:
            pass
        
        # Save to text file
        file_path = os.path.join(output_dir, f"analysis_summary_{timestamp}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_header + summary)
    
    def load_parameters(self):
        """Load parameters from file"""
        file_path = filedialog.askopenfilename(
            title="Load Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                # Update current parameters
                self.current_params.update(params)
                
                # Update widgets
                for param, value in params.items():
                    if param in self.param_widgets:
                        self.param_widgets[param][0].set(value)
                
                self.status_var.set(f"Parameters loaded from {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters: {e}")
    
    def save_parameters(self):
        """Save parameters to file"""
        params = self.get_current_parameters()
        
        # Add metadata
        params['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'platform': 'Multi-Strategy Parameter Platform'
        }
        
        file_path = filedialog.asksaveasfilename(
            title="Save Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=2, ensure_ascii=False)
                
                self.status_var.set(f"Parameters saved to {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save parameters: {e}")
    
    def reset_parameters(self):
        """Reset parameters to default values"""
        if messagebox.askyesno("Reset Parameters", "Are you sure you want to reset all parameters to default values?"):
            self.current_params = self.default_params.copy()
            # Update widgets
            for param, value in self.default_params.items():
                if param in self.param_widgets:
                    var, scale, entry = self.param_widgets[param]
                    var.set(value)
                    try:
                        scale.set(value)
                    except Exception:
                        pass
                    try:
                        entry.delete(0, 'end')
                        entry.insert(0, str(value))
                    except Exception:
                        pass
            self.status_var.set("Parameters reset to default values")
    
    def open_output_folder(self):
        """Open output folder in file explorer"""
        try:
            if os.path.exists(self.output_dir):
                if sys.platform == "win32":
                    os.startfile(self.output_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", self.output_dir])
                else:
                    subprocess.run(["xdg-open", self.output_dir])
            else:
                messagebox.showinfo("Info", "Output folder does not exist yet. Export some results first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open output folder: {e}")

    def optimize_parameters(self):
        """Optimize parameters based on performance analysis"""
        if self.analysis_results is None:
            messagebox.showwarning("Warning", "Please run analysis first before optimizing parameters.")
            return
        
        def optimize_parameters_thread():
            try:
                self.status_var.set("Optimizing parameters...")
                
                # Get optimization settings
                opt_params = self.get_current_parameters()
                iterations = int(opt_params.get('optimization_iterations', 5))
                step_size = opt_params.get('parameter_step_size', 0.2)
                threshold = opt_params.get('performance_threshold', 0.7)
                
                # Get current performance
                current_performance = self.calculate_performance_metric()
                
                # Store current parameters as baseline
                baseline_params = self.get_current_parameters().copy()
                best_params = baseline_params.copy()
                best_performance = current_performance
                
                # Add to history
                self.add_optimization_history("Baseline", baseline_params, current_performance)
                
                # Optimization loop
                for i in range(iterations):
                    self.status_var.set(f"Optimization iteration {i+1}/{iterations}...")
                    
                    # Generate parameter variations
                    variations = self.generate_parameter_variations(baseline_params, step_size)
                    
                    # Test each variation
                    for var_params in variations:
                        try:
                            # Run analysis with variation
                            var_results = run_multi_strategy_analysis(self.df, var_params)
                            var_performance = self.calculate_performance_from_results(var_results)
                            
                            # Update best if better
                            if var_performance > best_performance:
                                best_performance = var_performance
                                best_params = var_params.copy()
                                self.add_optimization_history(f"Iteration {i+1}", var_params, var_performance)
                        
                        except Exception as e:
                            print(f"Warning: Failed to test parameter variation: {e}")
                            continue
                    
                    # Update baseline for next iteration
                    baseline_params = best_params.copy()
                
                # Apply best parameters
                self.apply_parameters(best_params)
                self.best_params = best_params
                self.best_performance = best_performance
                
                # Update best performance display
                self.best_performance_var.set(f"Best Performance: {best_performance:.4f}")
                
                self.status_var.set(f"Optimization completed. Best performance: {best_performance:.4f}")
                messagebox.showinfo("Success", f"Parameter optimization completed!\nBest performance: {best_performance:.4f}")
                
            except Exception as e:
                self.status_var.set(f"Optimization failed: {e}")
                messagebox.showerror("Error", f"Optimization failed: {e}")
        
        # Run in background thread
        thread = threading.Thread(target=optimize_parameters_thread)
        thread.daemon = True
        thread.start()
    
    def calculate_performance_metric(self):
        """Calculate performance metric from current analysis results"""
        if self.analysis_results is None:
            return 0.0
        
        # Calculate Sharpe ratio or other performance metric
        decisions = self.analysis_results['decisions']
        combined_score = self.analysis_results['combined_score']
        
        # Simple performance metric based on score consistency and decision distribution
        score_std = combined_score.std()
        buy_ratio = (decisions == 'Buy').mean()
        sell_ratio = (decisions == 'Sell').mean()
        
        # Higher score consistency and balanced decisions = better performance
        performance = (1.0 / (1.0 + score_std)) * (buy_ratio + sell_ratio) * 0.5
        
        return performance
    
    def calculate_performance_from_results(self, results):
        """Calculate performance metric from analysis results"""
        decisions = results['decisions']
        combined_score = results['combined_score']
        
        # Simple performance metric
        score_std = combined_score.std()
        buy_ratio = (decisions == 'Buy').mean()
        sell_ratio = (decisions == 'Sell').mean()
        
        performance = (1.0 / (1.0 + score_std)) * (buy_ratio + sell_ratio) * 0.5
        
        return performance
    
    def generate_parameter_variations(self, base_params, step_size):
        """Generate parameter variations for optimization"""
        variations = []
        
        # Define parameter ranges
        param_ranges = {
            'rsi_window': (10, 20),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'bb_window': (15, 25),
            'bb_std': (1.5, 2.5),
            'obv_window': (8, 15),
            'obv_threshold': (1.0, 1.5),
            'buy_threshold': (65, 75),
            'sell_threshold': (25, 35),
            'rsi_weight': (0.3, 0.5),
            'bollinger_bands_weight': (0.25, 0.45),
            'obv_weight': (0.15, 0.35)
        }
        
        # Generate variations
        for param, (min_val, max_val) in param_ranges.items():
            if param in base_params:
                current_val = base_params[param]
                
                # Generate step up and step down
                step_up = min(max_val, current_val + step_size * (max_val - min_val) * 0.1)
                step_down = max(min_val, current_val - step_size * (max_val - min_val) * 0.1)
                
                # Create variations
                for new_val in [step_up, step_down]:
                    if new_val != current_val:
                        var_params = base_params.copy()
                        var_params[param] = new_val
                        
                        # Normalize weights if needed
                        if param.endswith('_weight'):
                            var_params = self.normalize_weights(var_params)
                        
                        variations.append(var_params)
        
        return variations
    
    def normalize_weights(self, params):
        """Normalize strategy weights to sum to 1.0"""
        weight_params = ['rsi_weight', 'bollinger_bands_weight', 'obv_weight']
        total_weight = sum(params.get(w, 0) for w in weight_params)
        
        if total_weight > 0:
            for w in weight_params:
                if w in params:
                    params[w] = params[w] / total_weight
        
        return params
    
    def add_optimization_history(self, iteration, params, performance):
        """Add optimization result to history"""
        history_entry = {
            'iteration': iteration,
            'params': params.copy(),
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(history_entry)
        
        # Update display
        self.optimization_text.insert(tk.END, f"{iteration}: {performance:.4f}\n")
        self.optimization_text.see(tk.END)
    
    def apply_parameters(self, params):
        """Apply parameters to widgets"""
        for param, value in params.items():
            if param in self.param_widgets:
                var, scale, entry = self.param_widgets[param]
                var.set(value)
    
    def send_params_to_demo(self):
        """Send optimized parameters to demo.py for execution"""
        if self.best_params is None:
            messagebox.showwarning("Warning", "No optimized parameters available. Please run optimization first.")
            return
        
        try:
            # Create parameters file for demo.py
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            params_file = f"optimized_params_{timestamp}.json"
            params_path = os.path.join(self.output_dir, params_file)
            
            # Save optimized parameters
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(self.best_params, f, indent=2, ensure_ascii=False)
            
            # Create demo execution script
            demo_script = f"""
import subprocess
import sys
import os

# Path to demo.py
demo_path = os.path.join(os.path.dirname(__file__), 'multi_strategy_system', 'demo.py')
params_path = os.path.join(os.path.dirname(__file__), 'output', '{params_file}')

# Run demo.py with optimized parameters
cmd = [sys.executable, demo_path, '--params', params_path]
subprocess.run(cmd)
"""
            
            # Save demo execution script
            script_path = os.path.join(self.output_dir, f"run_demo_{timestamp}.py")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(demo_script)
            
            # Execute demo.py
            demo_path = os.path.join(os.path.dirname(__file__), 'multi_strategy_system', 'demo.py')
            cmd = [sys.executable, demo_path, '--params', params_path]
            
            self.status_var.set("Executing demo.py with optimized parameters...")
            
            # Run in background
            def run_demo():
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
                    if result.returncode == 0:
                        self.status_var.set("Demo execution completed successfully")
                        messagebox.showinfo("Success", "Demo.py executed successfully with optimized parameters!")
                    else:
                        self.status_var.set(f"Demo execution failed: {result.stderr}")
                        messagebox.showerror("Error", f"Demo execution failed:\n{result.stderr}")
                except Exception as e:
                    self.status_var.set(f"Demo execution failed: {e}")
                    messagebox.showerror("Error", f"Demo execution failed: {e}")
            
            thread = threading.Thread(target=run_demo)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.status_var.set(f"Failed to send parameters: {e}")
            messagebox.showerror("Error", f"Failed to send parameters: {e}")

def main():
    """Main function"""
    root = tk.Tk()
    app = MultiStrategyParameterPlatform(root)
    root.mainloop()

if __name__ == "__main__":
    main() 