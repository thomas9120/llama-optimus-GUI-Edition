@echo off
REM llama-optimus: GUI launcher (Windows)
REM Runs llama-optimus-gui. If you just cloned the repo, run install.bat first.

setlocal

REM --- prefer the installed entry point, fall back to running the module ---
where llama-optimus-gui >nul 2>nul
if %errorlevel% equ 0 (
    start "" llama-optimus-gui
    exit /b 0
)

where python >nul 2>nul
if %errorlevel% equ 0 (
    python -m llama_optimus.gui
    if %errorlevel% equ 0 exit /b 0
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -m llama_optimus.gui
    if %errorlevel% equ 0 exit /b 0
)

echo [ERROR] Could not launch the llama-optimus GUI.
echo.
echo Is llama-optimus installed? Run install.bat first (requires Python 3.10+).
echo Once installed, the GUI is also available as the "llama-optimus-gui" command.
pause
exit /b 1
