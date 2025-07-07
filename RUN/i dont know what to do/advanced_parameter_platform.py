"""
Advanced Parameter Adjustment Platform
Enhanced GUI platform with real-time preview and parameter validation
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import os
from datetime import datetime
import subprocess
import sys
import threading
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class AdvancedParameterPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Trading Strategy Parameter Platform")
        self.root.geometry("1200x800")
        
        # Load default parameters
        self.default_params = {
            'basic_optimization': {
                'bb_window': 20,
                'bb_std': 2.0,
                'rsi_window': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'obv_threshold': 1.2,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.01,
                'max_position': 1
            },
            'take_profit_optimization': {
                'bb_window': 20,
                'bb_std': 2.0,
                'rsi_window': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'obv_threshold': 1.2,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.01,
                'max_position': 1
            },
            'filter_optimization': {
                'bb_window': 20,
                'bb_std': 2.0,
                'rsi_window': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'obv_threshold': 1.2,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.01,
                'max_position': 1,
                'trend_short_ma': 10,
                'trend_long_ma': 50,
                'atr_window': 14,
                'atr_multiplier': 0.5,
                'volume_window': 20,
                'volume_threshold': 1.2,
                'min_hold_periods': 5
            }
        }
        
        self.current_params = self.default_params.copy()
        self.parameter_history = []
        self.create_widgets()
        
    def create_widgets(self):
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_parameter_tab()
        self.create_preview_tab()
        self.create_batch_tab()
        self.create_history_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def create_parameter_tab(self):
        """Create parameter adjustment tab"""
        param_frame = ttk.Frame(self.notebook)
        self.notebook.add(param_frame, text="Parameters")
        
        # Left panel for parameters
        left_panel = ttk.Frame(param_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Strategy selection
        strategy_frame = ttk.LabelFrame(left_panel, text="Strategy Selection", padding="10")
        strategy_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(strategy_frame, text="Select Strategy:").pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value="basic_optimization")
        strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.strategy_var, 
                                     values=["basic_optimization", "take_profit_optimization", "filter_optimization"],
                                     state="readonly", width=25)
        strategy_combo.pack(side=tk.LEFT, padx=(10, 0))
        strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_change)
        
        # Parameter frame
        self.param_frame = ttk.LabelFrame(left_panel, text="Parameters", padding="10")
        self.param_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create parameter widgets
        self.param_widgets = {}
        self.create_parameter_widgets()
        
        # Right panel for actions
        right_panel = ttk.Frame(param_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Action buttons
        action_frame = ttk.LabelFrame(right_panel, text="Actions", padding="10")
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Load Parameters", command=self.load_parameters).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Save Parameters", command=self.save_parameters).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Reset to Default", command=self.reset_parameters).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Validate Parameters", command=self.validate_parameters).pack(fill=tk.X, pady=2)
        
        # Run buttons
        run_frame = ttk.LabelFrame(right_panel, text="Run Strategy", padding="10")
        run_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(run_frame, text="Run Single Strategy", command=self.run_single_strategy).pack(fill=tk.X, pady=2)
        ttk.Button(run_frame, text="Run All Strategies", command=self.run_all_strategies).pack(fill=tk.X, pady=2)
        ttk.Button(run_frame, text="Export Excel", command=self.export_excel).pack(fill=tk.X, pady=2)
        
        # Quick actions
        quick_frame = ttk.LabelFrame(right_panel, text="Quick Actions", padding="10")
        quick_frame.pack(fill=tk.X)
        
        ttk.Button(quick_frame, text="Save Current Config", command=self.save_current_config).pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="Load Last Config", command=self.load_last_config).pack(fill=tk.X, pady=2)
        
    def create_preview_tab(self):
        """Create preview tab with parameter visualization"""
        preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(preview_frame, text="Preview")
        
        # Parameter summary
        summary_frame = ttk.LabelFrame(preview_frame, text="Parameter Summary", padding="10")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=8, width=80)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Parameter visualization
        viz_frame = ttk.LabelFrame(preview_frame, text="Parameter Visualization", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update preview
        self.update_preview()
        
    def create_batch_tab(self):
        """Create batch processing tab"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="Batch Processing")
        
        # Batch configuration
        config_frame = ttk.LabelFrame(batch_frame, text="Batch Configuration", padding="10")
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Parameter ranges
        range_frame = ttk.Frame(config_frame)
        range_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(range_frame, text="Parameter Ranges:").pack(side=tk.LEFT)
        
        # Add parameter range inputs
        self.range_widgets = {}
        self.create_range_widgets(range_frame)
        
        # Batch actions
        action_frame = ttk.Frame(batch_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(action_frame, text="Generate Parameter Combinations", command=self.generate_combinations).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Run Batch Test", command=self.run_batch_test).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Stop Batch Test", command=self.stop_batch_test).pack(side=tk.LEFT, padx=5)
        
        # Batch results
        results_frame = ttk.LabelFrame(batch_frame, text="Batch Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.batch_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.batch_text.pack(fill=tk.BOTH, expand=True)
        
    def create_history_tab(self):
        """Create history tab"""
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="History")
        
        # History controls
        control_frame = ttk.Frame(history_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Load History", command=self.load_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear History", command=self.clear_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export History", command=self.export_history).pack(side=tk.LEFT, padx=5)
        
        # History display
        history_display_frame = ttk.LabelFrame(history_frame, text="Parameter History", padding="10")
        history_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview for history
        columns = ('Date', 'Strategy', 'Parameters', 'Performance')
        self.history_tree = ttk.Treeview(history_display_frame, columns=columns, show='headings')
        
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=150)
        
        self.history_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(history_display_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
    def create_parameter_widgets(self):
        """Create parameter input widgets"""
        # Clear existing widgets
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        # Create scrollable frame
        canvas = tk.Canvas(self.param_frame)
        scrollbar = ttk.Scrollbar(self.param_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Parameter definitions
        common_params = {
            'bb_window': ('Bollinger Bands Window', 5, 50, 'int'),
            'bb_std': ('Bollinger Bands Std Dev', 1.0, 4.0, 'float'),
            'rsi_window': ('RSI Window', 5, 30, 'int'),
            'rsi_oversold': ('RSI Oversold Level', 10, 40, 'int'),
            'rsi_overbought': ('RSI Overbought Level', 60, 90, 'int'),
            'obv_threshold': ('OBV Threshold', 1.0, 2.0, 'float'),
            'stop_loss_pct': ('Stop Loss (%)', 0.005, 0.05, 'float'),
            'take_profit_pct': ('Take Profit (%)', 0.005, 0.05, 'float'),
            'max_position': ('Max Position', 1, 5, 'int')
        }
        
        filter_params = {
            'trend_short_ma': ('Short MA Window', 5, 20, 'int'),
            'trend_long_ma': ('Long MA Window', 30, 100, 'int'),
            'atr_window': ('ATR Window', 10, 30, 'int'),
            'atr_multiplier': ('ATR Multiplier', 0.1, 1.0, 'float'),
            'volume_window': ('Volume Window', 10, 30, 'int'),
            'volume_threshold': ('Volume Threshold', 1.0, 2.0, 'float'),
            'min_hold_periods': ('Min Hold Periods', 1, 10, 'int')
        }
        
        # Create parameter widgets
        row = 0
        
        # Common parameters
        ttk.Label(scrollable_frame, text="Common Parameters", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 10))
        row += 1
        
        for param, (label, min_val, max_val, param_type) in common_params.items():
            ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 10))
            
            if param_type == 'int':
                var = tk.IntVar(value=self.current_params[self.strategy_var.get()].get(param, int(min_val)))
            else:
                var = tk.DoubleVar(value=self.current_params[self.strategy_var.get()].get(param, min_val))
            
            scale = ttk.Scale(scrollable_frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
            
            entry = ttk.Entry(scrollable_frame, textvariable=var, width=10)
            entry.grid(row=row, column=2, padx=(0, 10))
            
            # Add validation label
            validation_label = ttk.Label(scrollable_frame, text="", foreground="green")
            validation_label.grid(row=row, column=3, padx=(0, 10))
            
            self.param_widgets[param] = (var, scale, entry, validation_label)
            
            # Bind validation
            var.trace('w', lambda name, index, mode, p=param: self.validate_parameter(p))
            
            row += 1
        
        # Filter-specific parameters
        if self.strategy_var.get() == "filter_optimization":
            ttk.Label(scrollable_frame, text="Filter Parameters", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(10, 10))
            row += 1
            
            for param, (label, min_val, max_val, param_type) in filter_params.items():
                ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 10))
                
                if param_type == 'int':
                    var = tk.IntVar(value=self.current_params[self.strategy_var.get()].get(param, int(min_val)))
                else:
                    var = tk.DoubleVar(value=self.current_params[self.strategy_var.get()].get(param, min_val))
                
                scale = ttk.Scale(scrollable_frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
                scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
                
                entry = ttk.Entry(scrollable_frame, textvariable=var, width=10)
                entry.grid(row=row, column=2, padx=(0, 10))
                
                validation_label = ttk.Label(scrollable_frame, text="", foreground="green")
                validation_label.grid(row=row, column=3, padx=(0, 10))
                
                self.param_widgets[param] = (var, scale, entry, validation_label)
                
                var.trace('w', lambda name, index, mode, p=param: self.validate_parameter(p))
                
                row += 1
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_range_widgets(self, parent):
        """Create parameter range widgets for batch processing"""
        # This would be implemented for batch parameter ranges
        pass
        
    def on_strategy_change(self, event=None):
        """Handle strategy change"""
        strategy = self.strategy_var.get()
        if strategy not in self.current_params:
            self.current_params[strategy] = self.default_params[strategy].copy()
        self.create_parameter_widgets()
        self.update_preview()
        self.status_var.set(f"Switched to {strategy}")
        
    def validate_parameter(self, param):
        """Validate individual parameter"""
        if param in self.param_widgets:
            var, scale, entry, validation_label = self.param_widgets[param]
            value = var.get()
            
            # Get parameter definition
            common_params = {
                'bb_window': (5, 50), 'bb_std': (1.0, 4.0), 'rsi_window': (5, 30),
                'rsi_oversold': (10, 40), 'rsi_overbought': (60, 90), 'obv_threshold': (1.0, 2.0),
                'stop_loss_pct': (0.005, 0.05), 'take_profit_pct': (0.005, 0.05), 'max_position': (1, 5)
            }
            
            filter_params = {
                'trend_short_ma': (5, 20), 'trend_long_ma': (30, 100), 'atr_window': (10, 30),
                'atr_multiplier': (0.1, 1.0), 'volume_window': (10, 30), 'volume_threshold': (1.0, 2.0),
                'min_hold_periods': (1, 10)
            }
            
            all_params = {**common_params, **filter_params}
            
            if param in all_params:
                min_val, max_val = all_params[param]
                if min_val <= value <= max_val:
                    validation_label.config(text="✓", foreground="green")
                else:
                    validation_label.config(text="✗", foreground="red")
            else:
                validation_label.config(text="", foreground="black")
    
    def validate_parameters(self):
        """Validate all parameters"""
        invalid_params = []
        for param in self.param_widgets:
            var, scale, entry, validation_label = self.param_widgets[param]
            value = var.get()
            
            # Check parameter ranges
            common_params = {
                'bb_window': (5, 50), 'bb_std': (1.0, 4.0), 'rsi_window': (5, 30),
                'rsi_oversold': (10, 40), 'rsi_overbought': (60, 90), 'obv_threshold': (1.0, 2.0),
                'stop_loss_pct': (0.005, 0.05), 'take_profit_pct': (0.005, 0.05), 'max_position': (1, 5)
            }
            
            filter_params = {
                'trend_short_ma': (5, 20), 'trend_long_ma': (30, 100), 'atr_window': (10, 30),
                'atr_multiplier': (0.1, 1.0), 'volume_window': (10, 30), 'volume_threshold': (1.0, 2.0),
                'min_hold_periods': (1, 10)
            }
            
            all_params = {**common_params, **filter_params}
            
            if param in all_params:
                min_val, max_val = all_params[param]
                if not (min_val <= value <= max_val):
                    invalid_params.append(f"{param}: {value} (should be {min_val}-{max_val})")
        
        if invalid_params:
            messagebox.showerror("Parameter Validation", f"Invalid parameters:\n" + "\n".join(invalid_params))
        else:
            messagebox.showinfo("Parameter Validation", "All parameters are valid!")
    
    def get_current_parameters(self):
        """Get current parameter values"""
        params = {}
        for param, (var, scale, entry, validation_label) in self.param_widgets.items():
            params[param] = var.get()
        return params
    
    def update_preview(self):
        """Update preview tab"""
        # Update summary text
        strategy = self.strategy_var.get()
        params = self.get_current_parameters()
        
        summary = f"Strategy: {strategy}\n"
        summary += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        summary += "Parameters:\n"
        
        for param, value in params.items():
            summary += f"  {param}: {value}\n"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
        
        # Update visualization
        self.update_visualization(params)
    
    def update_visualization(self, params):
        """Update parameter visualization"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Create parameter bar chart
        param_names = list(params.keys())
        param_values = list(params.values())
        
        self.ax1.bar(range(len(param_names)), param_values)
        self.ax1.set_title("Parameter Values")
        self.ax1.set_ylabel("Value")
        self.ax1.set_xticks(range(len(param_names)))
        self.ax1.set_xticklabels(param_names, rotation=45, ha='right')
        
        # Create parameter distribution (simulated)
        self.ax2.hist(np.random.normal(0, 1, 1000), bins=30, alpha=0.7)
        self.ax2.set_title("Parameter Distribution (Sample)")
        self.ax2.set_xlabel("Value")
        self.ax2.set_ylabel("Frequency")
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def load_parameters(self):
        """Load parameters from file"""
        filename = filedialog.askopenfilename(
            title="Load Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    loaded_params = json.load(f)
                
                strategy = self.strategy_var.get()
                if strategy in loaded_params:
                    self.current_params[strategy] = loaded_params[strategy]
                    self.create_parameter_widgets()
                    self.update_preview()
                    self.status_var.set(f"Loaded parameters from {filename}")
                else:
                    messagebox.showerror("Error", f"No parameters found for {strategy}")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters: {e}")
    
    def save_parameters(self):
        """Save parameters to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                strategy = self.strategy_var.get()
                self.current_params[strategy] = self.get_current_parameters()
                
                with open(filename, 'w') as f:
                    json.dump(self.current_params, f, indent=2)
                
                self.status_var.set(f"Saved parameters to {filename}")
                messagebox.showinfo("Success", f"Parameters saved to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save parameters: {e}")
    
    def reset_parameters(self):
        """Reset parameters to default"""
        strategy = self.strategy_var.get()
        self.current_params[strategy] = self.default_params[strategy].copy()
        self.create_parameter_widgets()
        self.update_preview()
        self.status_var.set(f"Reset {strategy} parameters to default")
    
    def save_current_config(self):
        """Save current configuration"""
        try:
            strategy = self.strategy_var.get()
            params = self.get_current_parameters()
            
            config = {
                'strategy': strategy,
                'parameters': params,
                'timestamp': datetime.now().isoformat()
            }
            
            config_file = f"strategy_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.status_var.set(f"Saved current config to {config_file}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
    
    def load_last_config(self):
        """Load last saved configuration"""
        try:
            config_files = [f for f in os.listdir("RUN") if f.startswith("strategy_params_")]
            if config_files:
                latest_file = max(config_files, key=lambda x: os.path.getctime(os.path.join("RUN", x)))
                config_path = os.path.join("RUN", latest_file)
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.strategy_var.set(config['strategy'])
                self.current_params[config['strategy']] = config['parameters']
                self.create_parameter_widgets()
                self.update_preview()
                
                self.status_var.set(f"Loaded config from {latest_file}")
            else:
                messagebox.showinfo("Info", "No saved configurations found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
    
    def run_single_strategy(self):
        """Run single strategy with current parameters"""
        def run_strategy():
            try:
                strategy = self.strategy_var.get()
                params = self.get_current_parameters()
                self.current_params[strategy] = params
                param_file = f"strategy_params_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(param_file, 'w') as f:
                    json.dump({strategy: params}, f, indent=2)
                self.status_var.set(f"Running {strategy}...")
                if strategy == "basic_optimization":
                    cmd = [sys.executable, "basic_opt/1.py"]
                elif strategy == "take_profit_optimization":
                    cmd = [sys.executable, "takeprofit_opt/2.py"]
                elif strategy == "filter_optimization":
                    cmd = [sys.executable, "filter_opt/3.py"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    self.status_var.set(f"{strategy} completed successfully")
                    messagebox.showinfo("Success", f"{strategy} completed successfully!")
                else:
                    self.status_var.set(f"{strategy} failed")
                    messagebox.showerror("Error", f"{strategy} failed:\n{result.stderr}")
            except Exception as e:
                self.status_var.set(f"Error running {strategy}")
                messagebox.showerror("Error", f"Failed to run strategy: {e}")
        thread = threading.Thread(target=run_strategy)
        thread.daemon = True
        thread.start()
    
    def run_all_strategies(self):
        """Run all strategies with current parameters"""
        def run_all():
            try:
                self.status_var.set("Running all strategies...")
                strategies = ["basic_optimization", "take_profit_optimization", "filter_optimization"]
                results = {}
                for strategy in strategies:
                    self.strategy_var.set(strategy)
                    params = self.get_current_parameters()
                    param_file = f"strategy_params_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(param_file, 'w') as f:
                        json.dump({strategy: params}, f, indent=2)
                    if strategy == "basic_optimization":
                        cmd = [sys.executable, "basic_opt/1.py"]
                    elif strategy == "take_profit_optimization":
                        cmd = [sys.executable, "takeprofit_opt/2.py"]
                    elif strategy == "filter_optimization":
                        cmd = [sys.executable, "filter_opt/3.py"]
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                    results[strategy] = result.returncode == 0
                successful = [s for s, r in results.items() if r]
                failed = [s for s, r in results.items() if not r]
                if successful:
                    self.status_var.set(f"Completed: {', '.join(successful)}")
                if failed:
                    self.status_var.set(f"Failed: {', '.join(failed)}")
                messagebox.showinfo("Batch Complete", f"Successful: {len(successful)}\nFailed: {len(failed)}")
            except Exception as e:
                self.status_var.set("Error running strategies")
                messagebox.showerror("Error", f"Failed to run strategies: {e}")
        thread = threading.Thread(target=run_all)
        thread.daemon = True
        thread.start()
    
    def export_excel(self):
        """Export Excel with current parameters"""
        def export():
            try:
                self.status_var.set("Exporting Excel...")
                
                cmd = [sys.executable, "complete_excel_export.py"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                
                if result.returncode == 0:
                    self.status_var.set("Excel exported successfully")
                    messagebox.showinfo("Success", "Excel file exported successfully!")
                else:
                    self.status_var.set("Excel export failed")
                    messagebox.showerror("Error", f"Excel export failed:\n{result.stderr}")
                    
            except Exception as e:
                self.status_var.set("Error exporting Excel")
                messagebox.showerror("Error", f"Failed to export Excel: {e}")
        
        # Run in separate thread
        thread = threading.Thread(target=export)
        thread.daemon = True
        thread.start()
    
    def generate_combinations(self):
        """Generate parameter combinations for batch testing"""
        # This would generate parameter combinations for batch testing
        pass
    
    def run_batch_test(self):
        """Run batch parameter testing"""
        # This would run batch parameter testing
        pass
    
    def stop_batch_test(self):
        """Stop batch parameter testing"""
        # This would stop batch parameter testing
        pass
    
    def load_history(self):
        """Load parameter history"""
        # This would load parameter history
        pass
    
    def clear_history(self):
        """Clear parameter history"""
        # This would clear parameter history
        pass
    
    def export_history(self):
        """Export parameter history"""
        # This would export parameter history
        pass

def main():
    """Main function"""
    root = tk.Tk()
    app = AdvancedParameterPlatform(root)
    root.mainloop()

if __name__ == "__main__":
    main() 