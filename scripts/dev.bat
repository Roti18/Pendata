@echo off
echo Starting Development Environment...
echo [TIP] If Windows Firewall asks, please select 'Allow Access' for Python.

if not exist venv\Scripts\python.exe (
    echo [ERROR] Virtual environment ^(venv^) not found. Please run run.bat install first.
    pause
    exit /b 1
)

:: Set MODE and run server
set MODE=DEVELOPMENT
call venv\Scripts\python.exe --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Virtual environment is broken. Please run run.bat install to rebuild venv.
    pause
    exit /b 1
)

call venv\Scripts\python.exe scripts\dev_server.py
pause
