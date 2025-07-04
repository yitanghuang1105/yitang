import subprocess
import sys
import os
import time
from datetime import datetime

def run_system(system_name, system_path):
    """Run a single system and return the result"""
    print(f"\n{'='*60}")
    print(f"STARTING {system_name}")
    print(f"{'='*60}")
    
    try:
        # Run the system
        start_time = time.time()
        result = subprocess.run([sys.executable, system_path], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minutes timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {system_name} COMPLETED SUCCESSFULLY")
            print(f"⏱️  Execution time: {duration:.2f} seconds")
            print("\nOutput:")
            print(result.stdout)
            return True, result.stdout, duration
        else:
            print(f"❌ {system_name} FAILED")
            print(f"Error: {result.stderr}")
            return False, result.stderr, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {system_name} TIMEOUT (5 minutes)")
        return False, "Timeout", 300
    except Exception as e:
        print(f"❌ {system_name} ERROR: {e}")
        return False, str(e), 0

def run_all_systems_sequential():
    """Run all systems sequentially"""
    print("QUANTITATIVE TRADING SYSTEMS - SEQUENTIAL EXECUTION")
    print("Running all three systems with full dataset")
    
    systems = [
        ("System 1: Basic Strategy + Risk Management", "RUN/1/1.py"),
        ("System 2: Take Profit Optimization", "RUN/takeprofit_opt/2"),
        ("System 3: Signal Filtering Optimization", "RUN/filter_opt/3"),
        ("System 4: Strategy Demo", "RUN/strategy_demo/4"),
        ("System 5: Cost Calculation", "RUN/cost_calc/5"),
        ("System 6: Parameter Testing", "RUN/param_test/6")
    ]
    
    results = []
    
    for system_name, system_path in systems:
        success, output, duration = run_system(system_name, system_path)
        results.append({
            'name': system_name,
            'success': success,
            'output': output,
            'duration': duration
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    total_time = sum(r['duration'] for r in results)
    successful_systems = sum(1 for r in results if r['success'])
    
    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} {result['name']} ({result['duration']:.2f}s)")
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Successful systems: {successful_systems}/{len(systems)}")
    
    if successful_systems == len(systems):
        print("🎉 ALL SYSTEMS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️  SOME SYSTEMS FAILED")
    
    return results

def run_all_systems_parallel():
    """Run all systems in parallel (experimental)"""
    print("QUANTITATIVE TRADING SYSTEMS - PARALLEL EXECUTION")
    print("Running all three systems in parallel with full dataset")
    
    systems = [
        ("System 1: Basic Strategy + Risk Management", "RUN/1/1.py"),
        ("System 2: Take Profit Optimization", "RUN/takeprofit_opt/2"),
        ("System 3: Signal Filtering Optimization", "RUN/filter_opt/3"),
        ("System 4: Strategy Demo", "RUN/strategy_demo/4"),
        ("System 5: Cost Calculation", "RUN/cost_calc/5"),
        ("System 6: Parameter Testing", "RUN/param_test/6")
    ]
    
    # Start all processes
    processes = []
    for system_name, system_path in systems:
        print(f"Starting {system_name}...")
        process = subprocess.Popen([sys.executable, system_path], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        processes.append((system_name, process))
    
    # Wait for all processes to complete
    results = []
    start_time = time.time()
    
    for system_name, process in processes:
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5 minutes timeout
            duration = time.time() - start_time
            
            if process.returncode == 0:
                print(f"✅ {system_name} COMPLETED")
                results.append({
                    'name': system_name,
                    'success': True,
                    'output': stdout,
                    'duration': duration
                })
            else:
                print(f"❌ {system_name} FAILED")
                results.append({
                    'name': system_name,
                    'success': False,
                    'output': stderr,
                    'duration': duration
                })
                
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"⏰ {system_name} TIMEOUT")
            results.append({
                'name': system_name,
                'success': False,
                'output': "Timeout",
                'duration': 300
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("PARALLEL EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    successful_systems = sum(1 for r in results if r['success'])
    
    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} {result['name']} ({result['duration']:.2f}s)")
    
    print(f"\nSuccessful systems: {successful_systems}/{len(systems)}")
    
    return results

def main():
    """Main function"""
    print("Choose execution mode:")
    print("1. Sequential execution (recommended)")
    print("2. Parallel execution (experimental)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        results = run_all_systems_sequential()
    elif choice == "2":
        results = run_all_systems_parallel()
    else:
        print("Invalid choice. Running sequential execution...")
        results = run_all_systems_sequential()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"RUN/execution_results_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("QUANTITATIVE TRADING SYSTEMS EXECUTION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in results:
            f.write(f"{'='*40}\n")
            f.write(f"{result['name']}\n")
            f.write(f"{'='*40}\n")
            f.write(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            f.write(f"Duration: {result['duration']:.2f} seconds\n")
            f.write(f"Output:\n{result['output']}\n\n")
    
    print(f"\nResults saved to: {output_file}")
    return results

if __name__ == "__main__":
    main() 