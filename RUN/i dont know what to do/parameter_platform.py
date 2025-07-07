"""
Parameter Adjustment Platform
A GUI platform for adjusting trading strategy parameters
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from datetime import datetime
import subprocess
import sys
import pandas as pd

class ParameterPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Strategy Parameter Platform")
        self.root.geometry("800x600")
        
        # Load default parameters
        self.default_params = {
            'basic_optimization': {
                'bb_window': 20,
                'bb_std': 2.0,
                'rsi_window': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'obv_threshold': 1.2,
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.015,
                'max_position': 3
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
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Trading Strategy Parameter Platform", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Strategy selection
        ttk.Label(main_frame, text="Select Strategy:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.strategy_var = tk.StringVar(value="basic_optimization")
        strategy_combo = ttk.Combobox(main_frame, textvariable=self.strategy_var, 
                                     values=["basic_optimization", "take_profit_optimization", "filter_optimization"],
                                     state="readonly", width=30)
        strategy_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_change)
        
        # Parameter frame
        param_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        param_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        param_frame.columnconfigure(1, weight=1)
        
        # Create parameter widgets
        self.param_widgets = {}
        self.create_parameter_widgets(param_frame)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        # Buttons
        ttk.Button(button_frame, text="Load Parameters", command=self.load_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Parameters", command=self.save_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Run Strategy", command=self.run_strategy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Excel", command=self.export_excel).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_parameter_widgets(self, parent):
        # Clear existing widgets
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Common parameters
        common_params = {
            'bb_window': ('Bollinger Bands Window', 5, 50),
            'bb_std': ('Bollinger Bands Std Dev', 1.0, 4.0),
            'rsi_window': ('RSI Window', 5, 30),
            'rsi_oversold': ('RSI Oversold Level', 10, 40),
            'rsi_overbought': ('RSI Overbought Level', 60, 90),
            'obv_threshold': ('OBV Threshold', 1.0, 2.0),
            'stop_loss_pct': ('Stop Loss (%)', 0.005, 0.05),
            'take_profit_pct': ('Take Profit (%)', 0.005, 0.05),
            'max_position': ('Max Position', 1, 5)
        }
        
        # Filter-specific parameters
        filter_params = {
            'trend_short_ma': ('Short MA Window', 5, 20),
            'trend_long_ma': ('Long MA Window', 30, 100),
            'atr_window': ('ATR Window', 10, 30),
            'atr_multiplier': ('ATR Multiplier', 0.1, 1.0),
            'volume_window': ('Volume Window', 10, 30),
            'volume_threshold': ('Volume Threshold', 1.0, 2.0),
            'min_hold_periods': ('Min Hold Periods', 1, 10)
        }
        
        # Create parameter widgets
        row = 0
        col = 0
        
        # Common parameters
        ttk.Label(parent, text="Common Parameters", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        row += 1
        
        for param, (label, min_val, max_val) in common_params.items():
            ttk.Label(parent, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 10))
            
            var = tk.DoubleVar(value=self.current_params[self.strategy_var.get()].get(param, min_val))
            scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=col+1, sticky=(tk.W, tk.E), padx=(0, 10))
            
            entry = ttk.Entry(parent, textvariable=var, width=10)
            entry.grid(row=row, column=col+2, padx=(0, 20))
            
            self.param_widgets[param] = (var, scale, entry)
            
            row += 1
            if row > 5:  # Start new column
                row = 1
                col += 3
        
        # Filter-specific parameters
        if self.strategy_var.get() == "filter_optimization":
            row = 0
            col += 3
            
            ttk.Label(parent, text="Filter Parameters", font=("Arial", 12, "bold")).grid(row=row, column=col, columnspan=3, sticky=tk.W, pady=(0, 10))
            row += 1
            
            for param, (label, min_val, max_val) in filter_params.items():
                ttk.Label(parent, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 10))
                
                var = tk.DoubleVar(value=self.current_params[self.strategy_var.get()].get(param, min_val))
                scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
                scale.grid(row=row, column=col+1, sticky=(tk.W, tk.E), padx=(0, 10))
                
                entry = ttk.Entry(parent, textvariable=var, width=10)
                entry.grid(row=row, column=col+2, padx=(0, 20))
                
                self.param_widgets[param] = (var, scale, entry)
                
                row += 1
                if row > 5:  # Start new column
                    row = 1
                    col += 3
    
    def on_strategy_change(self, event=None):
        """Handle strategy change"""
        strategy = self.strategy_var.get()
        self.current_params[strategy] = self.default_params[strategy].copy()
        self.create_parameter_widgets(self.root.winfo_children()[0].winfo_children()[2])
        self.status_var.set(f"Switched to {strategy}")
    
    def get_current_parameters(self):
        """Get current parameter values"""
        params = {}
        for param, (var, scale, entry) in self.param_widgets.items():
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
                    self.create_parameter_widgets(self.root.winfo_children()[0].winfo_children()[2])
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
        self.create_parameter_widgets(self.root.winfo_children()[0].winfo_children()[2])
        self.status_var.set(f"Reset {strategy} parameters to default")
    
    def run_strategy(self):
        """Run the selected strategy with current parameters"""
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
    
    def export_excel(self):
        """Export Excel with current parameters"""
        try:
            self.status_var.set("Exporting Excel...")
            self.root.update()
            
            # Run Excel export
            cmd = [sys.executable, "批次測試與Excel匯出.py"]
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

def export_equity_curve_to_excel(df, writer, initial_capital=100000):
    equity_curve = df['cumulative_return'] * initial_capital
    out_df = pd.DataFrame({
        'Datetime': df.index,
        'Equity': equity_curve
    })
    out_df.to_excel(writer, sheet_name='Equity_Curve', index=False)

def main():
    """Main function"""
    root = tk.Tk()
    app = ParameterPlatform(root)
    root.mainloop()

if __name__ == "__main__":
    main() 