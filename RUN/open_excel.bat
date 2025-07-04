@echo off
echo ========================================
echo Auto Open Excel File
echo ========================================
echo.

REM Change to the RUN directory
cd /d "%~dp0"

REM Run the auto open Excel script
python open_excel_automatically.py

echo.
echo ========================================
echo Excel file should be open now
echo ========================================
pause 