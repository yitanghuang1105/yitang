@echo off
echo ========================================
echo Smart Excel Export
echo ========================================
echo.

REM Change to the RUN directory
cd /d "%~dp0"

REM Run the smart Excel export script
python smart_excel_export.py

echo.
echo ========================================
echo Smart Excel export completed
echo ========================================
pause 