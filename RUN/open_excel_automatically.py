"""
Automatically open Excel files after creation
"""

import os
import glob
import subprocess
import time
from datetime import datetime

def find_latest_excel_file():
    """Find the most recent Excel file in the current directory"""
    excel_files = glob.glob("complete_trading_results_with_charts_*.xlsx")
    
    if not excel_files:
        print("No Excel files found!")
        return None
    
    # Sort by modification time (newest first)
    latest_file = max(excel_files, key=os.path.getmtime)
    return latest_file

def open_excel_file(file_path):
    """Open Excel file using default application"""
    try:
        # Get absolute path
        abs_path = os.path.abspath(file_path)
        print(f"Opening Excel file: {abs_path}")
        
        # Open with default application
        if os.name == 'nt':  # Windows
            os.startfile(abs_path)
        else:  # macOS/Linux
            subprocess.run(['open', abs_path])
        
        print("✅ Excel file opened successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error opening Excel file: {e}")
        return False

def main():
    """Main function to find and open Excel file"""
    print("="*50)
    print("AUTO OPEN EXCEL FILE")
    print("="*50)
    
    # Find the latest Excel file
    excel_file = find_latest_excel_file()
    
    if excel_file:
        print(f"Found Excel file: {excel_file}")
        
        # Wait a moment to ensure file is fully written
        print("Waiting for file to be ready...")
        time.sleep(2)
        
        # Open the Excel file
        success = open_excel_file(excel_file)
        
        if success:
            print("\n📊 Excel file should now be open!")
            print("If it didn't open automatically, you can:")
            print(f"1. Navigate to: {os.getcwd()}")
            print(f"2. Double-click: {excel_file}")
        else:
            print("\n❌ Failed to open Excel file automatically")
            print("Please open it manually:")
            print(f"File location: {os.getcwd()}")
            print(f"File name: {excel_file}")
    else:
        print("No Excel files found in current directory")
        print("Please run the Excel export script first:")
        print("python complete_excel_export.py")

if __name__ == "__main__":
    main() 