@echo off
setlocal EnableDelayedExpansion

REM One-click installer for auto-subtitle on Windows.
REM Creates a venv, installs torch with CUDA support, then installs the
REM base requirements and the spaCy English model. Safe to re-run.

set VENV_DIR=venv
set TORCH_INDEX=https://download.pytorch.org/whl/cu124

echo.
echo === auto-subtitle Windows setup ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python was not found on PATH.
    echo Install Python 3.11+ from https://www.python.org/downloads/ and re-run.
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [1/5] Creating virtual environment in %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo [1/5] Reusing existing virtual environment in %VENV_DIR%.
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

echo [2/5] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [3/5] Installing torch + torchaudio (CUDA 12.4 build)...
pip install torch torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 exit /b 1

REM Bundle CUDA runtime libs through pip so ctranslate2 (used by faster-whisper)
REM finds cuBLAS / cuDNN regardless of the system-wide CUDA install.
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
if errorlevel 1 exit /b 1

echo [4/5] Installing base requirements...
pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [5/5] Downloading spaCy English model...
python -m spacy download en_core_web_sm
if errorlevel 1 exit /b 1

echo.
echo === Setup complete ===
echo.
echo Activate the environment in a new shell with:
echo     %VENV_DIR%\Scripts\activate
echo.
echo Then launch the app:
echo     python gui.py
echo     python youtube_subtitle_app\main.py
echo.
echo To install the experimental NeMo / Parakeet backend:
echo     pip install -r requirements-experimental.txt
echo.

endlocal
