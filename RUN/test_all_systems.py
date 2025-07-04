import sys
import os
import importlib.util
import pandas as pd
import numpy as np

def test_system_1():
    """Test RUN/1/1 - Basic strategy with risk management"""
    print("="*60)
    print("TESTING SYSTEM 1: Basic Strategy + Risk Management")
    print("="*60)
    
    try:
        # Import system 1
        module_path = os.path.join(os.path.dirname(__file__), '1', '1')
        spec = importlib.util.spec_from_file_location("system1", module_path)
        system1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system1)
        
        # Test data loading and conversion
        print("Testing data loading and 4-hour conversion...")
        df_1min = system1.load_txf_data("TXF1_Minute_2020-01-01_2025-06-16.txt")
        df_4h = system1.convert_to_4h_data(df_1min)
        print(f"✓ Data conversion successful: {len(df_1min)} 1-min → {len(df_4h)} 4-hour")
        
        # Test indicators calculation
        print("Testing technical indicators...")
        df_bb = system1.calculate_bollinger_bands(df_4h, window=20, num_std=2.0)
        df_rsi = system1.calculate_rsi(df_bb, window=14)
        df_obv = system1.calculate_obv(df_rsi)
        print("✓ Technical indicators calculated successfully")
        
        # Test signal generation
        print("Testing signal generation...")
        params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'obv_threshold': 1.2,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.01
        }
        df_signals = system1.generate_signals(df_obv, params)
        signals = df_signals['signal'].value_counts()
        print(f"✓ Signals generated: {signals.to_dict()}")
        
        # Test performance calculation
        print("Testing performance calculation...")
        metrics = system1.calculate_returns(df_signals, params)
        print("✓ Performance metrics calculated successfully")
        
        print("SYSTEM 1: ALL TESTS PASSED ✓")
        return True
        
    except Exception as e:
        print(f"SYSTEM 1: ERROR - {e}")
        return False

def test_system_2():
    """Test RUN/2/2 - Basic strategy with 1% take profit"""
    print("\n" + "="*60)
    print("TESTING SYSTEM 2: Basic Strategy + 1% Take Profit")
    print("="*60)
    
    try:
        # Import system 2
        module_path = os.path.join(os.path.dirname(__file__), '2', '2')
        spec = importlib.util.spec_from_file_location("system2", module_path)
        system2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system2)
        
        # Test data loading and conversion
        print("Testing data loading and 4-hour conversion...")
        df_1min = system2.load_txf_data("TXF1_Minute_2020-01-01_2025-06-16.txt")
        df_4h = system2.convert_to_4h_data(df_1min)
        print(f"✓ Data conversion successful: {len(df_1min)} 1-min → {len(df_4h)} 4-hour")
        
        # Test indicators calculation
        print("Testing technical indicators...")
        df_bb = system2.calculate_bollinger_bands(df_4h, window=20, num_std=2.0)
        df_rsi = system2.calculate_rsi(df_bb, window=14)
        df_obv = system2.calculate_obv(df_rsi)
        print("✓ Technical indicators calculated successfully")
        
        # Test signal generation with 1% take profit
        print("Testing signal generation with 1% take profit...")
        params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'obv_threshold': 1.2,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.01  # 1% take profit
        }
        df_signals = system2.generate_signals(df_obv, params)
        signals = df_signals['signal'].value_counts()
        print(f"✓ Signals generated: {signals.to_dict()}")
        
        # Test performance calculation
        print("Testing performance calculation...")
        metrics = system2.calculate_returns(df_signals, params)
        print("✓ Performance metrics calculated successfully")
        
        print("SYSTEM 2: ALL TESTS PASSED ✓")
        return True
        
    except Exception as e:
        print(f"SYSTEM 2: ERROR - {e}")
        return False

def test_system_3():
    """Test RUN/3/3 - Strategy with signal filtering"""
    print("\n" + "="*60)
    print("TESTING SYSTEM 3: Strategy + Signal Filtering")
    print("="*60)
    
    try:
        # Import system 3
        module_path = os.path.join(os.path.dirname(__file__), '3', '3')
        spec = importlib.util.spec_from_file_location("system3", module_path)
        system3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system3)
        
        # Test data loading and conversion
        print("Testing data loading and 4-hour conversion...")
        df_1min = system3.load_txf_data("TXF1_Minute_2020-01-01_2025-06-16.txt")
        df_4h = system3.convert_to_4h_data(df_1min)
        print(f"✓ Data conversion successful: {len(df_1min)} 1-min → {len(df_4h)} 4-hour")
        
        # Test indicators calculation
        print("Testing technical indicators...")
        df_bb = system3.calculate_bollinger_bands(df_4h, window=20, num_std=2.0)
        df_rsi = system3.calculate_rsi(df_bb, window=14)
        df_obv = system3.calculate_obv(df_rsi)
        print("✓ Technical indicators calculated successfully")
        
        # Test signal filters
        print("Testing signal filters...")
        params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'obv_threshold': 1.2,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.01,
            'trend_short_ma': 10,
            'trend_long_ma': 50,
            'atr_window': 14,
            'atr_multiplier': 0.5,
            'volume_window': 20,
            'volume_threshold': 1.2
        }
        df_filters = system3.apply_signal_filters(df_obv, params)
        print("✓ Signal filters applied successfully")
        
        # Test signal generation with filters
        print("Testing signal generation with filters...")
        df_signals = system3.generate_signals(df_filters, params)
        signals = df_signals['signal'].value_counts()
        print(f"✓ Signals generated: {signals.to_dict()}")
        
        # Test performance calculation
        print("Testing performance calculation...")
        metrics = system3.calculate_returns(df_signals, params)
        print("✓ Performance metrics calculated successfully")
        
        print("SYSTEM 3: ALL TESTS PASSED ✓")
        return True
        
    except Exception as e:
        print(f"SYSTEM 3: ERROR - {e}")
        return False

def test_data_consistency():
    """Test that all systems produce consistent data"""
    print("\n" + "="*60)
    print("TESTING DATA CONSISTENCY ACROSS SYSTEMS")
    print("="*60)
    
    try:
        # Import all systems
        module_path1 = os.path.join(os.path.dirname(__file__), '1', '1')
        spec1 = importlib.util.spec_from_file_location("system1", module_path1)
        system1 = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(system1)
        
        module_path2 = os.path.join(os.path.dirname(__file__), '2', '2')
        spec2 = importlib.util.spec_from_file_location("system2", module_path2)
        system2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(system2)
        
        module_path3 = os.path.join(os.path.dirname(__file__), '3', '3')
        spec3 = importlib.util.spec_from_file_location("system3", module_path3)
        system3 = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(system3)
        
        # Load and convert data with all systems
        print("Loading data with all systems...")
        df_1min_1 = system1.load_txf_data("TXF1_Minute_2020-01-01_2025-06-16.txt")
        df_1min_2 = system2.load_txf_data("TXF1_Minute_2020-01-01_2025-06-16.txt")
        df_1min_3 = system3.load_txf_data("TXF1_Minute_2020-01-01_2025-06-16.txt")
        
        df_4h_1 = system1.convert_to_4h_data(df_1min_1)
        df_4h_2 = system2.convert_to_4h_data(df_1min_2)
        df_4h_3 = system3.convert_to_4h_data(df_1min_3)
        
        # Check consistency
        print("Checking data consistency...")
        assert len(df_4h_1) == len(df_4h_2) == len(df_4h_3), "Data lengths should be identical"
        assert df_4h_1.index[0] == df_4h_2.index[0] == df_4h_3.index[0], "Start dates should be identical"
        assert df_4h_1.index[-1] == df_4h_2.index[-1] == df_4h_3.index[-1], "End dates should be identical"
        
        print(f"✓ Data consistency verified: {len(df_4h_1)} records from {df_4h_1.index[0]} to {df_4h_1.index[-1]}")
        
        # Test indicator consistency
        print("Testing indicator consistency...")
        df_bb_1 = system1.calculate_bollinger_bands(df_4h_1, window=20, num_std=2.0)
        df_bb_2 = system2.calculate_bollinger_bands(df_4h_2, window=20, num_std=2.0)
        df_bb_3 = system3.calculate_bollinger_bands(df_4h_3, window=20, num_std=2.0)
        
        # Check if indicators are identical (within numerical precision)
        assert np.allclose(df_bb_1['bb_middle'], df_bb_2['bb_middle'], rtol=1e-10), "BB indicators should be identical"
        assert np.allclose(df_bb_1['bb_middle'], df_bb_3['bb_middle'], rtol=1e-10), "BB indicators should be identical"
        
        print("✓ Indicator consistency verified")
        
        print("DATA CONSISTENCY: ALL TESTS PASSED ✓")
        return True
        
    except Exception as e:
        print(f"DATA CONSISTENCY: ERROR - {e}")
        return False

def main():
    """Run all tests"""
    print("QUANTITATIVE TRADING SYSTEMS - COMPREHENSIVE TEST")
    print("Testing all three systems for connectivity and consistency")
    
    results = []
    
    # Test each system
    results.append(test_system_1())
    results.append(test_system_2())
    results.append(test_system_3())
    results.append(test_data_consistency())
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all(results):
        print("🎉 ALL SYSTEMS PASSED ALL TESTS!")
        print("✅ System 1: Basic Strategy + Risk Management")
        print("✅ System 2: Basic Strategy + 1% Take Profit")
        print("✅ System 3: Strategy + Signal Filtering")
        print("✅ Data Consistency Across All Systems")
        print("\nAll systems are ready for production use!")
    else:
        print("❌ SOME TESTS FAILED")
        failed_tests = [i+1 for i, result in enumerate(results) if not result]
        print(f"Failed tests: {failed_tests}")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 