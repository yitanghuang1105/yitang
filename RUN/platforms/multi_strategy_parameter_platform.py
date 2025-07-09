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
            'obv_weight': 0.25
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
        
        # Progress bar for loading
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=300)
        self.progress.grid(row=4, column=0, columnspan=4, pady=(10, 0))
        self.progress.grid_remove()
        
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
            
            self.status_var.set("Parameters reset to default values")
    
    def run_analysis(self):
        """Run multi-strategy analysis"""
        try:
            # 新增提示訊息
            messagebox.showinfo("Info", "Run successfully and wait a moment?")
            # 顯示並啟動進度條
            self.progress.grid()
            self.progress.start(10)
            self.status_var.set("Running analysis...")
            self.root.update()
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
            script_path = os.path.join("multi_strategy_system", "demo.py")
            if os.path.exists(script_path):
                subprocess.run([sys.executable, script_path, "--params", temp_file])
                self.status_var.set("Multi-strategy analysis completed")
            else:
                messagebox.showerror("Error", f"Analysis script not found: {script_path}")
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            # 分析結束後隱藏進度條
            self.progress.stop()
            self.progress.grid_remove()
                
        except Exception as e:
            self.progress.stop()
            self.progress.grid_remove()
            messagebox.showerror("Error", f"Failed to run analysis: {e}")
    
    def export_results(self):
        """Export analysis results with plots"""
        try:
            # Get current parameters
            params = self.get_current_parameters()
            
            # Validate parameters before export
            total_weight = params['rsi_weight'] + params['bollinger_bands_weight'] + params['obv_weight']
            if abs(total_weight - 1.0) > 0.01:
                messagebox.showwarning("Warning", f"Weights sum to {total_weight:.2f}, not 1.0. Consider auto-balancing.")
            
            if params['buy_threshold'] <= params['sell_threshold']:
                messagebox.showerror("Error", "Buy threshold must be higher than sell threshold")
                return
            
            # Create output directory
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate timestamp for file naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Run analysis and generate plots
            self.status_var.set("Generating analysis and plots...")
            self.root.update()
            
            # Import required modules for analysis
            try:
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import sys
                import os
                
                # Add multi_strategy_system to path
                sys.path.append('multi_strategy_system')
                from multi_strategy_system.strategy_combiner import run_multi_strategy_analysis
                from multi_strategy_system.demo import load_sample_data, plot_results
                from multi_strategy_system.performance_analyzer import PerformanceAnalyzer
                
                # Load sample data
                df = load_sample_data()
                
                # Prepare parameters for analysis
                analysis_params = {
                    'rsi_window': params['rsi_window'],
                    'rsi_oversold': params['rsi_oversold'],
                    'rsi_overbought': params['rsi_overbought'],
                    'bb_window': params['bb_window'],
                    'bb_std': params['bb_std'],
                    'obv_window': params['obv_window'],
                    'obv_threshold': params['obv_threshold'],
                    'buy_threshold': params['buy_threshold'],
                    'sell_threshold': params['sell_threshold']
                }
                
                # Prepare weights
                weights = {
                    'rsi': params['rsi_weight'],
                    'bollinger_bands': params['bollinger_bands_weight'],
                    'obv': params['obv_weight']
                }
                
                # Run analysis
                results = run_multi_strategy_analysis(df, analysis_params, weights)
                
                # Generate plot
                plot_filename = f"multi_strategy_analysis_{timestamp}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                
                # Create the plot with more comprehensive analysis (including equity curve)
                fig, axes = plt.subplots(8, 1, figsize=(16, 24))
                fig.suptitle('Multi-Strategy Analysis Results with Equity Curve', fontsize=18, fontweight='bold')
                
                # Plot 1: Price and decisions
                ax1 = axes[0]
                ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, alpha=0.8)
                
                # Add decision markers
                decisions = results['decisions']
                decisions = decisions.reindex(df.index, method='ffill')
                buy_points = df.index[decisions == 'Buy']
                sell_points = df.index[decisions == 'Sell']
                
                ax1.scatter(buy_points, df.loc[buy_points, 'close'], 
                           color='green', marker='^', s=80, label='Buy Signal', alpha=0.9, zorder=5)
                ax1.scatter(sell_points, df.loc[sell_points, 'close'], 
                           color='red', marker='v', s=80, label='Sell Signal', alpha=0.9, zorder=5)
                
                ax1.set_title('Price and Trading Signals', fontsize=14, fontweight='bold')
                ax1.legend(loc='upper left')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylabel('Price')
                
                # Plot 2: Individual strategy scores
                ax2 = axes[1]
                individual_scores = results['individual_scores']
                colors = ['blue', 'orange', 'green']
                for i, (name, score) in enumerate(individual_scores.items()):
                    ax2.plot(df.index, score, label=name.replace('_', ' ').title(), 
                            alpha=0.8, linewidth=1.5, color=colors[i])
                
                ax2.axhline(y=params['buy_threshold'], color='green', linestyle='--', alpha=0.7, 
                           label=f'Buy Threshold ({params["buy_threshold"]})', linewidth=2)
                ax2.axhline(y=params['sell_threshold'], color='red', linestyle='--', alpha=0.7, 
                           label=f'Sell Threshold ({params["sell_threshold"]})', linewidth=2)
                ax2.set_title('Individual Strategy Scores', fontsize=14, fontweight='bold')
                ax2.legend(loc='upper left')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
                ax2.set_ylabel('Score (0-100)')
                
                # Plot 3: Combined score with zones
                ax3 = axes[2]
                combined_score = results['combined_score']
                ax3.plot(df.index, combined_score, label='Combined Score', linewidth=2.5, color='purple')
                ax3.axhline(y=params['buy_threshold'], color='green', linestyle='--', alpha=0.7, 
                           label=f'Buy Threshold ({params["buy_threshold"]})', linewidth=2)
                ax3.axhline(y=params['sell_threshold'], color='red', linestyle='--', alpha=0.7, 
                           label=f'Sell Threshold ({params["sell_threshold"]})', linewidth=2)
                ax3.fill_between(df.index, params['buy_threshold'], 100, alpha=0.3, color='green', label='Buy Zone')
                ax3.fill_between(df.index, 0, params['sell_threshold'], alpha=0.3, color='red', label='Sell Zone')
                ax3.fill_between(df.index, params['sell_threshold'], params['buy_threshold'], alpha=0.3, color='yellow', label='Hold Zone')
                ax3.set_title('Combined Strategy Score with Decision Zones', fontsize=14, fontweight='bold')
                ax3.legend(loc='upper left')
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, 100)
                ax3.set_ylabel('Combined Score')
                
                # Plot 4: Strategy weights visualization
                ax4 = axes[3]
                weights = results['weights_used']
                strategy_names = [name.replace('_', ' ').title() for name in weights.keys()]
                weight_values = list(weights.values())
                colors = ['skyblue', 'lightcoral', 'lightgreen']
                
                bars = ax4.bar(strategy_names, weight_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                ax4.set_title('Strategy Weights Distribution', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Weight')
                ax4.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, weight_values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
                
                # Plot 5: Signal distribution
                ax5 = axes[4]
                signal_counts = decisions.value_counts()
                signal_colors = {'Buy': 'green', 'Sell': 'red', 'Hold': 'gray'}
                signal_values = [signal_counts.get('Buy', 0), signal_counts.get('Sell', 0), signal_counts.get('Hold', 0)]
                signal_labels = ['Buy', 'Sell', 'Hold']
                
                bars = ax5.bar(signal_labels, signal_values, color=[signal_colors[label] for label in signal_labels], 
                              alpha=0.8, edgecolor='black', linewidth=1)
                ax5.set_title('Signal Distribution', fontsize=14, fontweight='bold')
                ax5.set_ylabel('Number of Signals')
                
                # Add value labels on bars
                for bar, value in zip(bars, signal_values):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + max(signal_values)*0.01,
                            f'{value}', ha='center', va='bottom', fontweight='bold')
                
                # Plot 6: Volume analysis
                ax6 = axes[5]
                ax6.bar(df.index, df['volume'], alpha=0.6, color='blue', label='Volume')
                
                # Highlight volume on signal days
                if len(buy_points) > 0:
                    ax6.bar(buy_points, df.loc[buy_points, 'volume'], alpha=0.8, color='green', label='Buy Signal Volume')
                if len(sell_points) > 0:
                    ax6.bar(sell_points, df.loc[sell_points, 'volume'], alpha=0.8, color='red', label='Sell Signal Volume')
                
                ax6.set_title('Volume Analysis', fontsize=14, fontweight='bold')
                ax6.legend(loc='upper left')
                ax6.grid(True, alpha=0.3)
                ax6.set_ylabel('Volume')
                
                # Plot 7: Equity Curve vs Buy & Hold
                ax7 = axes[6]
                
                # Calculate equity curve for strategy
                analyzer = PerformanceAnalyzer(initial_capital=100000)
                equity_curve = analyzer.calculate_equity_curve(df, decisions, position_size=0.1)
                
                # Calculate Buy & Hold equity curve
                initial_price = df['close'].iloc[0]
                buy_hold_equity = 100000 * (df['close'] / initial_price)
                
                # Plot both equity curves
                ax7.plot(df.index, equity_curve, label='Strategy Equity', linewidth=2.5, color='blue')
                ax7.plot(df.index, buy_hold_equity, label='Buy & Hold', linewidth=2.5, color='red', linestyle='--')
                
                # Calculate and display performance metrics
                strategy_return = (equity_curve.iloc[-1] - 100000) / 100000 * 100
                buy_hold_return = (buy_hold_equity.iloc[-1] - 100000) / 100000 * 100
                
                ax7.set_title(f'Equity Curve Comparison\nStrategy: {strategy_return:.2f}% vs Buy & Hold: {buy_hold_return:.2f}%', 
                             fontsize=14, fontweight='bold')
                ax7.legend(loc='upper left')
                ax7.grid(True, alpha=0.3)
                ax7.set_ylabel('Equity ($)')
                
                # Plot 8: Performance Metrics Summary
                ax8 = axes[7]
                
                # Calculate additional performance metrics
                mdd, peak_date, trough_date = analyzer.calculate_mdd(equity_curve)
                returns = analyzer.calculate_returns(equity_curve)
                sharpe_ratio = analyzer.calculate_sharpe_ratio(returns)
                win_rate = analyzer.calculate_win_rate(equity_curve)
                
                # Create performance summary table
                metrics_data = [
                    ['Total Return', f'{strategy_return:.2f}%'],
                    ['Buy & Hold Return', f'{buy_hold_return:.2f}%'],
                    ['Max Drawdown', f'{mdd:.2f}%'],
                    ['Sharpe Ratio', f'{sharpe_ratio:.2f}'],
                    ['Win Rate', f'{win_rate:.1f}%'],
                    ['Time Period', f'{len(df)} minutes ({len(df)/60:.1f} hours)']
                ]
                
                # Create table
                table = ax8.table(cellText=metrics_data, 
                                 colLabels=['Metric', 'Value'],
                                 cellLoc='center',
                                 loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1, 2)
                
                # Style the table
                for i in range(len(metrics_data) + 1):
                    for j in range(2):
                        cell = table[(i, j)]
                        if i == 0:  # Header row
                            cell.set_facecolor('#4CAF50')
                            cell.set_text_props(weight='bold', color='white')
                        else:
                            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                
                ax8.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
                ax8.axis('off')
                ax8.set_xlabel('Time')
                
                plt.tight_layout()
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create results summary
                results_summary = {
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
                    'analysis_results': {
                        'total_signals': len(decisions[decisions != 'Hold']),
                        'buy_signals': len(buy_points),
                        'sell_signals': len(sell_points),
                        'avg_combined_score': float(results['combined_score'].mean()),
                        'score_std': float(results['combined_score'].std())
                    },
                    'files_generated': {
                        'plot': plot_filename,
                        'timestamp': timestamp
                    },
                    'exported_at': datetime.now().isoformat()
                }
                
                # Show success message with file locations
                success_msg = f"""✅ 分析完成並匯出成功！

📊 生成的文件:
• 圖表: {plot_filename}

📁 保存位置: {output_dir}

📈 分析摘要:
• 總信號數: {results_summary['analysis_results']['total_signals']}
• 買入信號: {results_summary['analysis_results']['buy_signals']}
• 賣出信號: {results_summary['analysis_results']['sell_signals']}
• 平均綜合分數: {results_summary['analysis_results']['avg_combined_score']:.2f}

是否要開啟圖表文件？"""
                
                if messagebox.askyesno("匯出成功", success_msg):
                    # Open the plot file
                    if sys.platform == "win32":
                        os.startfile(plot_path)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", plot_path])
                    else:
                        subprocess.run(["xdg-open", plot_path])
                
                self.status_var.set(f"✅ 分析完成 - 圖表和結果已保存到 {output_dir}")
                
            except ImportError as e:
                messagebox.showerror("錯誤", f"缺少必要模組: {e}\n請確保已安裝 pandas, numpy, matplotlib")
            except Exception as e:
                messagebox.showerror("錯誤", f"生成圖表時發生錯誤: {e}")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"匯出失敗: {e}")

def main():
    """Main function"""
    root = tk.Tk()
    app = MultiStrategyParameterPlatform(root)
    root.mainloop()

if __name__ == "__main__":
    main() 