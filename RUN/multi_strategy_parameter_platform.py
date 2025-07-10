"""
Multi-Strategy Parameter Adjustment Platform
A GUI platform for adjusting multi-strategy parameters and weights
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from datetime import datetime
import subprocess
import sys

class MultiStrategyParameterPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Strategy Parameter Platform")
        self.root.geometry("1200x800")
        
        # Load default parameters from multi-strategy system
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
            
            # Reverse modes
            'reverse_mode': False,
            'reverse2_mode': False
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
        title_label = ttk.Label(main_frame, text="Multi-Strategy Parameter Platform", 
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
        
        # Reverse Mode Tab
        reverse_frame = ttk.Frame(notebook, padding="10")
        notebook.add(reverse_frame, text="Reverse Mode")
        self.create_reverse_mode_tab(reverse_frame)
        
        # Reverse2 Mode Tab
        reverse2_frame = ttk.Frame(notebook, padding="10")
        notebook.add(reverse2_frame, text="Reverse2 Mode")
        self.create_reverse2_mode_tab(reverse2_frame)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=20)
        
        # Buttons
        ttk.Button(button_frame, text="Load Parameters", command=self.load_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Parameters", command=self.save_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Run Multi-Strategy Analysis", command=self.run_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        
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
        
    def create_reverse_mode_tab(self, parent):
        """Create reverse mode tab"""
        # Reverse mode frame
        reverse_frame = ttk.LabelFrame(parent, text="Reverse Mode Settings", padding="20")
        reverse_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=20)
        
        # Reverse mode checkbox
        self.reverse_mode_var = tk.BooleanVar(value=False)
        reverse_checkbox = ttk.Checkbutton(
            reverse_frame, 
            text="Enable Reverse Mode", 
            variable=self.reverse_mode_var,
            command=self.on_reverse_mode_change
        )
        reverse_checkbox.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Reverse mode description
        description_text = """
Reverse Mode Explanation:
• Normal Mode: High scores → Buy signals, Low scores → Sell signals
• Reverse Mode: High scores → Sell signals, Low scores → Buy signals

This allows you to trade against the strategy signals, 
useful for contrarian trading approaches.
        """
        description_label = ttk.Label(
            reverse_frame, 
            text=description_text, 
            font=("Arial", 10),
            justify=tk.LEFT
        )
        description_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status indicator
        self.reverse_status_var = tk.StringVar(value="Mode: Normal")
        status_label = ttk.Label(
            reverse_frame, 
            textvariable=self.reverse_status_var, 
            font=("Arial", 12, "bold"),
            foreground="green"
        )
        status_label.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
    def on_reverse_mode_change(self):
        """Handle reverse mode checkbox change"""
        self.update_combined_mode_display()
        
    def create_reverse2_mode_tab(self, parent):
        """Create reverse2 mode tab"""
        # Reverse2 mode frame
        reverse2_frame = ttk.LabelFrame(parent, text="Reverse2 Mode Settings", padding="20")
        reverse2_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=20)
        
        # Reverse2 mode checkbox
        self.reverse2_mode_var = tk.BooleanVar(value=False)
        reverse2_checkbox = ttk.Checkbutton(
            reverse2_frame, 
            text="Enable Reverse2 Mode", 
            variable=self.reverse2_mode_var,
            command=self.on_reverse2_mode_change
        )
        reverse2_checkbox.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Reverse2 mode description
        description_text = """
Reverse2 Mode Explanation:
• Normal Mode: High scores → Buy signals, Low scores → Sell signals
• Reverse2 Mode: Separately reverses entry and exit signals

Reverse2 Logic:
• Reverse Entry: High scores → Sell signals, Low scores → Buy signals
• Reverse Exit: Exit conditions are reversed (opposite thresholds)

Can work together with Reverse Mode:
• Reverse + Reverse2: Both modes active simultaneously
• Allows fine-grained control over entry and exit logic
        """
        description_label = ttk.Label(
            reverse2_frame, 
            text=description_text, 
            font=("Arial", 10),
            justify=tk.LEFT
        )
        description_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status indicator
        self.reverse2_status_var = tk.StringVar(value="Mode: Normal")
        status_label = ttk.Label(
            reverse2_frame, 
            textvariable=self.reverse2_status_var, 
            font=("Arial", 12, "bold"),
            foreground="green"
        )
        status_label.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Mode combination indicator
        self.combined_mode_var = tk.StringVar(value="Combined Mode: Normal")
        combined_label = ttk.Label(
            reverse2_frame, 
            textvariable=self.combined_mode_var, 
            font=("Arial", 11, "bold"),
            foreground="blue"
        )
        combined_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
    def on_reverse2_mode_change(self):
        """Handle reverse2 mode checkbox change"""
        self.update_combined_mode_display()
        
    def update_combined_mode_display(self):
        """Update the combined mode display"""
        reverse = self.reverse_mode_var.get()
        reverse2 = self.reverse2_mode_var.get()
        
        if reverse2:
            self.reverse2_status_var.set("Mode: Reverse2")
        else:
            self.reverse2_status_var.set("Mode: Normal")
        
        # Update combined mode display
        mode_parts = []
        if reverse:
            mode_parts.append('Reverse')
        if reverse2:
            mode_parts.append('Reverse2')
        if not mode_parts:
            mode_parts.append('Normal')
        
        combined_mode = '+'.join(mode_parts)
        self.combined_mode_var.set(f"Combined Mode: {combined_mode}")
        
        # Also update reverse mode display
        if reverse:
            self.reverse_status_var.set("Mode: Reverse")
        else:
            self.reverse_status_var.set("Mode: Normal")
    
    def create_param_widgets(self, parent, params, start_row, start_col):
        """Create parameter widgets for a given set of parameters"""
        row = start_row
        
        for param, (label, min_val, max_val) in params.items():
            ttk.Label(parent, text=label).grid(row=row, column=start_col, sticky=tk.W, padx=(0, 10), pady=5)
            
            var = tk.DoubleVar(value=self.current_params.get(param, min_val))
            scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=start_col+1, sticky=(tk.W, tk.E), padx=(0, 10), pady=5)
            
            entry = ttk.Entry(parent, textvariable=var, width=10)
            entry.grid(row=row, column=start_col+2, padx=(0, 20), pady=5)
            
            if not hasattr(self, 'param_widgets'):
                self.param_widgets = {}
            self.param_widgets[param] = (var, scale, entry)
            
            row += 1
        
        # Configure column weights
        parent.columnconfigure(start_col+1, weight=1)
        
    def validate_weights(self, *args):
        """Validate that weights sum to 1.0"""
        try:
            rsi_weight = self.param_widgets['rsi_weight'][0].get()
            bb_weight = self.param_widgets['bollinger_bands_weight'][0].get()
            obv_weight = self.param_widgets['obv_weight'][0].get()
            
            total = rsi_weight + bb_weight + obv_weight
            
            if abs(total - 1.0) < 0.01:
                self.weight_sum_var.set(f"Total: {total:.2f} ✓")
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
                params[param] = var.get()
            except tk.TclError:
                params[param] = self.default_params.get(param, 0)
        
        # Add reverse mode parameters
        params['reverse_mode'] = self.reverse_mode_var.get()
        params['reverse2_mode'] = self.reverse2_mode_var.get()
        
        return params
    
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
                    elif param == 'reverse_mode':
                        self.reverse_mode_var.set(value)
                    elif param == 'reverse2_mode':
                        self.reverse2_mode_var.set(value)
                
                # Update combined mode display
                self.update_combined_mode_display()
                
                self.status_var.set(f"Parameters loaded from {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters: {e}")
    
    def save_parameters(self):
        """Save parameters to file"""
        # Get current parameters
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
                    self.param_widgets[param][0].set(value)
            
            # Reset reverse mode
            self.reverse_mode_var.set(False)
            self.reverse2_mode_var.set(False)
            
            # Update combined mode display
            self.update_combined_mode_display()
            
            self.status_var.set("Parameters reset to default values")
    
    def run_analysis(self):
        """Run multi-strategy analysis"""
        try:
            # Get current parameters
            params = self.get_current_parameters()
            
            # Validate weights
            total_weight = params['rsi_weight'] + params['bollinger_bands_weight'] + params['obv_weight']
            if abs(total_weight - 1.0) > 0.01:
                messagebox.showwarning("Warning", f"Weights sum to {total_weight:.2f}, not 1.0. Consider auto-balancing.")
            
            # Validate thresholds
            if params['buy_threshold'] <= params['sell_threshold']:
                messagebox.showerror("Error", "Buy threshold must be higher than sell threshold")
                return
            
            # Save parameters to temporary file
            temp_file = f"temp_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2)
            
            # Run analysis script
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "multi_strategy_system", "demo.py"))
            if os.path.exists(script_path):
                subprocess.run([sys.executable, script_path, "--params", temp_file])
                self.status_var.set("Multi-strategy analysis completed")
            else:
                messagebox.showerror("Error", f"Analysis script not found: {script_path}")
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run analysis: {e}")
    
    def export_results(self):
        """Export analysis results"""
        try:
            # Get current parameters
            params = self.get_current_parameters()
            
            # Create results summary
            results = {
                'parameters': params,
                'weights_summary': {
                    'rsi': f"{params['rsi_weight']:.1%}",
                    'bollinger_bands': f"{params['bollinger_bands_weight']:.1%}",
                    'obv': f"{params['obv_weight']:.1%}"
                },
                'decision_thresholds': {
                    'buy': params['buy_threshold'],
                    'sell': params['sell_threshold']
                },
                'exported_at': datetime.now().isoformat()
            }
            
            file_path = filedialog.asksaveasfilename(
                title="Export Results",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                self.status_var.set(f"Results exported to {os.path.basename(file_path)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")

def main():
    """Main function"""
    print("Starting Multi-Strategy Parameter Platform...")
    print("Creating main window...")
    
    root = tk.Tk()
    root.title("Multi-Strategy Parameter Platform")
    root.geometry("1200x800")
    
    # Ensure window is on top
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    print("Creating application...")
    app = MultiStrategyParameterPlatform(root)
    
    print("GUI created successfully. Starting mainloop...")
    print("If you don't see the window, check if it's minimized or behind other windows.")
    
    root.mainloop()
    print("GUI closed.")

if __name__ == "__main__":
    main() 