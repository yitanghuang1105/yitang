"""
Simple Parameter Adjustment Platform
A streamlined GUI for adjusting trading strategy parameters
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from datetime import datetime
import subprocess
import sys
import glob

class SimpleParameterPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Parameter Platform")
        self.root.geometry("900x700")
        
        # Default parameters for each strategy
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
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_container, text="Trading Strategy Parameter Platform", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Strategy selection frame
        strategy_frame = ttk.LabelFrame(main_container, text="Strategy Selection", padding="10")
        strategy_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(strategy_frame, text="Select Strategy:").pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value="basic_optimization")
        strategy_combo = ttk.Combobox(strategy_frame, textvariable=self.strategy_var, 
                                     values=["basic_optimization", "take_profit_optimization", "filter_optimization"],
                                     state="readonly", width=30)
        strategy_combo.pack(side=tk.LEFT, padx=(10, 0))
        strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_change)
        
        # Parameter frame
        param_frame = ttk.LabelFrame(main_container, text="Parameters", padding="10")
        param_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create scrollable parameter area
        self.create_parameter_area(param_frame)
        
        # Action buttons frame
        button_frame = ttk.Frame(main_container)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)
        
        ttk.Button(left_buttons, text="Load Parameters", command=self.load_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(left_buttons, text="Save Parameters", command=self.save_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(left_buttons, text="Reset to Default", command=self.reset_parameters).pack(side=tk.LEFT, padx=5)
        
        # Right buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT)
        
        ttk.Button(right_buttons, text="Run Strategy", command=self.run_strategy).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_buttons, text="Run All Strategies", command=self.run_all_strategies).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_buttons, text="Export Excel", command=self.export_excel).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_container, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        
    def create_parameter_area(self, parent):
        """Create scrollable parameter area"""
        # Create canvas and scrollbar
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
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
        self.param_widgets = {}
        row = 0
        
        # Common parameters
        ttk.Label(scrollable_frame, text="Common Parameters", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 10))
        row += 1
        
        for param, (label, min_val, max_val, param_type) in common_params.items():
            ttk.Label(scrollable_frame, text=label, width=25).grid(row=row, column=0, sticky=tk.W, padx=(0, 10))
            
            if param_type == 'int':
                var = tk.IntVar(value=self.current_params[self.strategy_var.get()].get(param, int(min_val)))
            else:
                var = tk.DoubleVar(value=self.current_params[self.strategy_var.get()].get(param, min_val))
            
            scale = ttk.Scale(scrollable_frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
            
            entry = ttk.Entry(scrollable_frame, textvariable=var, width=10)
            entry.grid(row=row, column=2, padx=(0, 10))
            
            # Value label
            value_label = ttk.Label(scrollable_frame, text=f"({min_val}-{max_val})", foreground="gray")
            value_label.grid(row=row, column=3, padx=(0, 10))
            
            self.param_widgets[param] = (var, scale, entry, value_label)
            row += 1
        
        # Filter-specific parameters
        if self.strategy_var.get() == "filter_optimization":
            ttk.Label(scrollable_frame, text="Filter Parameters", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(10, 10))
            row += 1
            
            for param, (label, min_val, max_val, param_type) in filter_params.items():
                ttk.Label(scrollable_frame, text=label, width=25).grid(row=row, column=0, sticky=tk.W, padx=(0, 10))
                
                if param_type == 'int':
                    var = tk.IntVar(value=self.current_params[self.strategy_var.get()].get(param, int(min_val)))
                else:
                    var = tk.DoubleVar(value=self.current_params[self.strategy_var.get()].get(param, min_val))
                
                scale = ttk.Scale(scrollable_frame, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
                scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
                
                entry = ttk.Entry(scrollable_frame, textvariable=var, width=10)
                entry.grid(row=row, column=2, padx=(0, 10))
                
                value_label = ttk.Label(scrollable_frame, text=f"({min_val}-{max_val})", foreground="gray")
                value_label.grid(row=row, column=3, padx=(0, 10))
                
                self.param_widgets[param] = (var, scale, entry, value_label)
                row += 1
        
        # Configure grid weights
        scrollable_frame.columnconfigure(1, weight=1)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def on_strategy_change(self, event=None):
        """Handle strategy change"""
        strategy = self.strategy_var.get()
        if strategy not in self.current_params:
            self.current_params[strategy] = self.default_params[strategy].copy()
        
        # Recreate parameter widgets
        param_frame = self.root.winfo_children()[0].winfo_children()[2]
        for widget in param_frame.winfo_children():
            widget.destroy()
        self.create_parameter_area(param_frame)
        
        self.status_var.set(f"Switched to {strategy}")
    
    def get_current_parameters(self):
        """Get current parameter values"""
        params = {}
        for param, (var, scale, entry, value_label) in self.param_widgets.items():
            params[param] = var.get()
        return params
    
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
                    
                    # Update widgets
                    param_frame = self.root.winfo_children()[0].winfo_children()[2]
                    for widget in param_frame.winfo_children():
                        widget.destroy()
                    self.create_parameter_area(param_frame)
                    
                    self.status_var.set(f"Loaded parameters from {filename}")
                    messagebox.showinfo("Success", f"Parameters loaded from {filename}")
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
        
        # Update widgets
        param_frame = self.root.winfo_children()[0].winfo_children()[2]
        for widget in param_frame.winfo_children():
            widget.destroy()
        self.create_parameter_area(param_frame)
        
        self.status_var.set(f"Reset {strategy} parameters to default")
        messagebox.showinfo("Success", f"Reset {strategy} parameters to default")
    
    def run_strategy(self):
        """Run single strategy with current parameters"""
        try:
            strategy = self.strategy_var.get()
            params = self.get_current_parameters()
            self.current_params[strategy] = params
            param_file = f"strategy_params_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(param_file, 'w') as f:
                json.dump({strategy: params}, f, indent=2)
            self.status_var.set(f"Running {strategy}...")
            self.root.update()
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
    
    def run_all_strategies(self):
        """Run all strategies with current parameters"""
        try:
            self.status_var.set("Running all strategies...")
            self.root.update()
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
    
    def export_excel(self):
        """Export Excel with current parameters"""
        try:
            self.status_var.set("Exporting Excel...")
            self.root.update()
            
            # Run Excel export
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

def load_latest_params():
    param_files = sorted(glob.glob("strategy_params_basic_optimization_*.json"), reverse=True)
    if not param_files:
        print("No parameter file found! Using default parameters.")
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'obv_threshold': 1.2,
            'adx_threshold': 25,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.01
        }
    with open(param_files[0], "r") as f:
        data = json.load(f)
        return data.get("basic_optimization", data)

def main():
    """Main function"""
    root = tk.Tk()
    app = SimpleParameterPlatform(root)
    root.mainloop()

    params = load_latest_params()
    print(f"Enhanced strategy parameters: {params}")

if __name__ == "__main__":
    main() 