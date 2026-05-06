@echo off
REM SubForge Launcher - double-click to start.
REM First run: checks for Python, runs setup, then launches.
REM Subsequent runs: launches immediately.

cd /d "%~dp0"

REM Add local tools/ to PATH so bundled ffmpeg is found
if exist "tools\ffmpeg.exe" set "PATH=%~dp0tools;%PATH%"

REM --- Fast path: setup completed previously, just launch ---
if exist ".setup_done" (
    start "" "venv\Scripts\pythonw.exe" -m subforge
    exit /b 0
)

REM --- Env not ready, need setup ---
echo.
echo ==================================================
echo   SubForge - First launch, setting up environment
echo   This may take a few minutes. Please wait.
echo ==================================================
echo.

REM Check if Python is available
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo.
    echo Please install Python 3.11+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo Then double-click SubForge.bat again.
    echo.
    pause
    exit /b 1
)

REM Run setup
if not exist "scripts\setup.bat" (
    echo [ERROR] Cannot find scripts\setup.bat
    echo Make sure SubForge.bat is in the project root.
    echo.
    pause
    exit /b 1
)

call scripts\setup.bat
if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed. Check the messages above and try again.
    echo.
    pause
    exit /b 1
)

REM Verify setup worked
if not exist "venv\Scripts\subforge.exe" (
    echo.
    echo [ERROR] Setup completed but subforge was not installed correctly.
    echo Try running manually:
    echo   venv\Scripts\activate
    echo   pip install -e ".[full]"
    echo.
    pause
    exit /b 1
)

echo.
echo Setup complete! Starting SubForge ...
start "" "venv\Scripts\pythonw.exe" -m subforge
exit /b 0
