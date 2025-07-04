@echo off
echo ========================================
echo Excel Export for RUN Strategy Tests
echo ========================================
echo.

cd /d "%~dp0"
python excel_export.py

echo.
echo Press any key to exit...
pause >nul 