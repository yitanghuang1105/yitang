"""
Batch execution script for all strategy files in RUN directory
Runs all numbered files (1.py, 2.py, 3.py, 4.py, 5.py, 6.py) simultaneously
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import multiprocessing as mp
from pathlib import Path

def run_strategy_file(file_path):
    """Run a single strategy file and return the result"""
    try:
        print(f"Starting {file_path}...")
        start_time = time.time()
        
        # Run the Python file
        result = subprocess.run([sys.executable, file_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {file_path} completed successfully in {execution_time:.2f} seconds")
            return {
                'file': file_path,
                'status': 'success',
                'execution_time': execution_time,
                'output': result.stdout,
                'error': result.stderr
            }
        else:
            print(f"❌ {file_path} failed with return code {result.returncode}")
            return {
                'file': file_path,
                'status': 'failed',
                'execution_time': execution_time,
                'output': result.stdout,
                'error': result.stderr
            }
            
    except Exception as e:
        print(f"❌ {file_path} encountered an error: {str(e)}")
        return {
            'file': file_path,
            'status': 'error',
            'execution_time': 0,
            'output': '',
            'error': str(e)
        }

def main():
    """Main function to run all strategy files"""
    print("="*80)
    print("BATCH EXECUTION: All Strategy Files")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define the strategy files to run
    strategy_files = [
        "basic_opt/1.py",
        "takeprofit_opt/2.py", 
        "filter_opt/3.py",
        "strategy_demo/4.py",
        "cost_calc/5.py",
        "param_test/6.py"
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
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main() 