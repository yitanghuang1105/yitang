@echo off
echo ========================================
echo Test with Excel Export for RUN Strategies
echo ========================================
echo.

cd /d "%~dp0"
python test_with_excel_export.py

echo.
echo Press any key to exit...
pause >nul 