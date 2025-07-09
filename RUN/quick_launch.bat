@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   多策略交易平台快速啟動器
echo ========================================
echo.
echo 正在啟動平台啟動器...
echo.

python platform_launcher.py

if errorlevel 1 (
    echo.
    echo 啟動失敗，請檢查Python環境
    pause
) 