"""
Test script to run strategies and export results to Excel
Handles data file path issues and creates Excel report
"""

import pandas as pd
import numpy as np
import subprocess
import sys
import os
import time
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def run_strategy_file(file_path):
    """Run a single strategy file and return the result, 並嘗試取得 cumulative_return 欄位的 DataFrame"""
    try:
        print(f"Starting {file_path}...")
        start_time = time.time()
        strategy_dir = os.path.dirname(file_path)
        original_cwd = os.getcwd()
        os.chdir(strategy_dir)
        result = subprocess.run([sys.executable, os.path.basename(file_path)], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        os.chdir(original_cwd)
        end_time = time.time()
        execution_time = end_time - start_time
        res = {
            'file': file_path,
            'status': 'success' if result.returncode == 0 else 'failed',
            'execution_time': execution_time,
            'output': result.stdout,
            'error': result.stderr
        }
        # 嘗試自動讀取策略產生的 cumulative_return DataFrame
        # 這裡以 takeprofit_opt/2.py 為例，假設它會輸出一個 csv 檔案
        csv_path = os.path.join(strategy_dir, 'cumulative_return.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                res['cumulative_return'] = df
            except Exception as e:
                print(f"讀取 cumulative_return.csv 失敗: {e}")
        return res
    except Exception as e:
        print(f"❌ {file_path} encountered an error: {str(e)}")
        return {
            'file': file_path,
            'status': 'error',
            'execution_time': 0,
            'output': '',
            'error': str(e)
        }

def extract_performance_metrics(output_text):
    """Extract performance metrics from strategy output"""
    metrics = {
        'total_return': None,
        'num_trades': None,
        'winning_trades': None,
        'losing_trades': None,
        'win_rate': None,
        'avg_win': None,
        'avg_loss': None,
        'profit_factor': None,
        'max_drawdown': None
    }
    
    lines = output_text.split('\n')
    for line in lines:
        line = line.strip()
        if 'Total Return:' in line:
            try:
                value = float(line.split(':')[1].split()[0])
                metrics['total_return'] = value
            except:
                pass
        elif 'Number of Trades:' in line:
            try:
                value = int(line.split(':')[1].strip())
                metrics['num_trades'] = value
            except:
                pass
        elif 'Winning Trades:' in line:
            try:
                value = int(line.split(':')[1].strip())
                metrics['winning_trades'] = value
            except:
                pass
        elif 'Losing Trades:' in line:
            try:
                value = int(line.split(':')[1].strip())
                metrics['losing_trades'] = value
            except:
                pass
        elif 'Win Rate:' in line:
            try:
                value = float(line.split(':')[1].split()[0])
                metrics['win_rate'] = value
            except:
                pass
        elif 'Average Win:' in line:
            try:
                value = float(line.split(':')[1].split()[0])
                metrics['avg_win'] = value
            except:
                pass
        elif 'Average Loss:' in line:
            try:
                value = float(line.split(':')[1].split()[0])
                metrics['avg_loss'] = value
            except:
                pass
        elif 'Profit Factor:' in line:
            try:
                value = float(line.split(':')[1].strip())
                metrics['profit_factor'] = value
            except:
                pass
        elif 'Maximum Drawdown:' in line:
            try:
                value = float(line.split(':')[1].split()[0])
                metrics['max_drawdown'] = value
            except:
                pass
    
    return metrics

def create_execution_summary(results):
    """Create execution summary DataFrame"""
    summary_data = []
    
    for result in results:
        summary_data.append({
            'Strategy File': result['file'],
            'Status': result['status'],
            'Execution Time (seconds)': result['execution_time'],
            'Error Message': result['error'] if result['error'] else ''
        })
    
    return pd.DataFrame(summary_data)

def create_performance_summary(results):
    """Create performance summary DataFrame"""
    performance_data = []
    
    for result in results:
        if result['status'] == 'success':
            metrics = extract_performance_metrics(result['output'])
            performance_data.append({
                'Strategy File': result['file'],
                'Total Return (%)': metrics['total_return'] * 100 if metrics['total_return'] is not None else None,
                'Number of Trades': metrics['num_trades'],
                'Winning Trades': metrics['winning_trades'],
                'Losing Trades': metrics['losing_trades'],
                'Win Rate (%)': metrics['win_rate'] * 100 if metrics['win_rate'] is not None else None,
                'Average Win (%)': metrics['avg_win'] * 100 if metrics['avg_win'] is not None else None,
                'Average Loss (%)': metrics['avg_loss'] * 100 if metrics['avg_loss'] is not None else None,
                'Profit Factor': metrics['profit_factor'],
                'Maximum Drawdown (%)': metrics['max_drawdown'] * 100 if metrics['max_drawdown'] is not None else None,
                'Execution Time (seconds)': result['execution_time']
            })
    
    return pd.DataFrame(performance_data)

def create_strategy_comparison(performance_df):
    """Create strategy comparison DataFrame with rankings"""
    if performance_df.empty:
        return pd.DataFrame()
    
    # Create a copy for ranking
    comparison_df = performance_df.copy()
    
    # Add rankings for key metrics
    comparison_df['Return Rank'] = comparison_df['Total Return (%)'].rank(ascending=False, na_option='bottom')
    comparison_df['Win Rate Rank'] = comparison_df['Win Rate (%)'].rank(ascending=False, na_option='bottom')
    comparison_df['Profit Factor Rank'] = comparison_df['Profit Factor'].rank(ascending=False, na_option='bottom')
    comparison_df['Drawdown Rank'] = comparison_df['Maximum Drawdown (%)'].rank(ascending=True, na_option='bottom')
    
    # Calculate overall score (lower is better)
    comparison_df['Overall Score'] = (
        comparison_df['Return Rank'] + 
        comparison_df['Win Rate Rank'] + 
        comparison_df['Profit Factor Rank'] + 
        comparison_df['Drawdown Rank']
    )
    comparison_df['Overall Rank'] = comparison_df['Overall Score'].rank(ascending=True)
    
    # Reorder columns
    cols = ['Strategy File', 'Overall Rank', 'Total Return (%)', 'Win Rate (%)', 
            'Profit Factor', 'Maximum Drawdown (%)', 'Number of Trades', 
            'Winning Trades', 'Losing Trades', 'Average Win (%)', 'Average Loss (%)',
            'Execution Time (seconds)', 'Return Rank', 'Win Rate Rank', 
            'Profit Factor Rank', 'Drawdown Rank', 'Overall Score']
    
    return comparison_df[cols]

def export_to_excel(results, output_file):
    """Export all results to Excel with multiple sheets"""
    print(f"\nExporting results to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Execution Summary
        execution_df = create_execution_summary(results)
        execution_df.to_excel(writer, sheet_name='Execution Summary', index=False)
        # Sheet 2: Performance Summary
        performance_df = create_performance_summary(results)
        if not performance_df.empty:
            performance_df.to_excel(writer, sheet_name='Performance Summary', index=False)
            # Sheet 3: Strategy Comparison
            comparison_df = create_strategy_comparison(performance_df)
            if not comparison_df.empty:
                comparison_df.to_excel(writer, sheet_name='Strategy Comparison', index=False)
        # Sheet 4: Raw Output
        raw_data = []
        for result in results:
            raw_data.append({
                'Strategy File': result['file'],
                'Status': result['status'],
                'Output': result['output'],
                'Error': result['error']
            })
        raw_df = pd.DataFrame(raw_data)
        raw_df.to_excel(writer, sheet_name='Raw Output', index=False)
        # Sheet 5: Test Configuration
        config_data = {
            'Test Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Total Strategies': [len(results)],
            'Successful': [len([r for r in results if r['status'] == 'success'])],
            'Failed': [len([r for r in results if r['status'] != 'success'])],
            'Success Rate': [f"{len([r for r in results if r['status'] == 'success'])/len(results)*100:.1f}%"],
            'Total Execution Time': [f"{sum(r['execution_time'] for r in results):.2f} seconds"]
        }
        config_df = pd.DataFrame(config_data)
        config_df.to_excel(writer, sheet_name='Test Configuration', index=False)
        # 新增：權益曲線 sheet
        export_equity_curve_to_excel(results, writer, initial_capital=100000)
    print(f"✅ Excel file exported successfully: {output_file}")
    # 自動開啟 Excel 檔案（僅限 Windows）
    try:
        abs_path = os.path.abspath(output_file)
        os.startfile(abs_path)
        print("✅ Excel file opened automatically!")
    except Exception as e:
        print(f"⚠️ Could not auto-open Excel file: {e}")

def export_equity_curve_to_excel(results, writer, initial_capital=100000):
    """將每個策略的權益曲線匯出到 Excel（每個策略一個 sheet）"""
    for result in results:
        # 檢查是否有 cumulative_return DataFrame
        if result.get('status') == 'success' and isinstance(result.get('cumulative_return'), pd.DataFrame):
            df = result['cumulative_return']
            if 'cumulative_return' in df.columns:
                equity_curve = df['cumulative_return'] * initial_capital
                out_df = pd.DataFrame({
                    'Datetime': df.index,
                    'Equity': equity_curve
                })
                sheet_name = f"Equity_{Path(result['file']).stem}"
                out_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

def main():
    """Main function to run all strategies and export to Excel"""
    print("="*80)
    print("TEST WITH EXCEL EXPORT: RUN Strategy Test Results")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define the strategy files to run
    strategy_files = [
        "RUN/basic_opt/1.py",
        "RUN/takeprofit_opt/2.py",
        "RUN/filter_opt/3.py",
        "RUN/strategy_demo/4.py",
        "RUN/cost_calc/5.py",
        "RUN/param_test/6.py"
    ]
    
    # Check which files exist
    existing_files = []
    for file_path in strategy_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"Found: {file_path}")
        else:
            print(f"Missing: {file_path}")
    
    if not existing_files:
        print("No strategy files found!")
        return
    
    print(f"\nRunning {len(existing_files)} strategy files...")
    print("-" * 80)
    
    # Run all files in parallel using multiprocessing
    start_time = time.time()
    
    with mp.Pool(processes=min(len(existing_files), mp.cpu_count())) as pool:
        results = pool.map(run_strategy_file, existing_files)
    
    total_time = time.time() - start_time
    
    # Display results summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    successful = 0
    failed = 0
    
    for result in results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"{status_icon} {result['file']}")
        print(f"   Status: {result['status']}")
        print(f"   Time: {result['execution_time']:.2f} seconds")
        
        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1
            if result['error']:
                print(f"   Error: {result['error'][:200]}...")
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
    
    # Export to Excel
    output_file = f"strategy_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    export_to_excel(results, output_file)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main() 