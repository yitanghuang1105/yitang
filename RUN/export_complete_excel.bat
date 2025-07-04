@echo off
echo ========================================
echo Complete Excel Export with Charts
echo ========================================
echo.

REM Change to the RUN directory
cd /d "%~dp0"

REM Create RUN directory if it doesn't exist
if not exist "RUN" mkdir RUN

REM Run the complete Excel export script with charts
python complete_excel_export.py

echo.
echo ========================================
echo Complete Excel export with charts completed
echo ========================================
pause 