@echo off
echo ========================================
echo BATCH EXECUTION: All Strategy Files
echo ========================================
echo Start time: %date% %time%
echo.

echo Starting all strategy files in parallel...
echo.

start "Strategy 1" cmd /c "cd basic_opt && python 1.py && pause"
start "Strategy 2" cmd /c "cd takeprofit_opt && python 2.py && pause"
start "Strategy 3" cmd /c "cd filter_opt && python 3.py && pause"
start "Strategy 4" cmd /c "cd strategy_demo && python 4.py && pause"
start "Strategy 5" cmd /c "cd cost_calc && python 5.py && pause"
start "Strategy 6" cmd /c "cd param_test && python 6.py && pause"

echo.
echo All strategy files started in separate windows.
echo Each window will pause when complete.
echo.
echo End time: %date% %time%
echo ========================================
pause 