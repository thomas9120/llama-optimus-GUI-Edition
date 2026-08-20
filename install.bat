@echo off
REM llama-optimus: install script (Windows)
REM Installs llama-optimus from this repository into the current Python environment.
REM Requires Python 3.10+ (https://www.python.org/downloads/).

setlocal
cd /d "%~dp0"

echo.
echo === llama-optimus installer ===
echo.

REM --- check that Python is available ---
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found on PATH.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         and make sure "Add Python to PATH" is checked during setup.
    pause
    exit /b 1
)

echo Using Python:
python --version
echo.

echo Installing llama-optimus (this may take a minute)...
python -m pip install --upgrade .

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed. See the messages above.
    pause
    exit /b 1
)

echo.
echo === Installation complete ===
echo.
echo  - Command line :  llama-optimus --help
echo  - GUI          :  llama-optimus-gui      (or double-click start_gui.bat)
echo.
echo  Tip: drop a llama.cpp build into %%USERPROFILE%%\.llama-optimus\llama\bin
echo       and llama-optimus will find it automatically.
echo.
pause
