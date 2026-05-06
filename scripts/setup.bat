@echo off
setlocal EnableDelayedExpansion

REM One-click installer for SubForge on Windows.
REM Creates a venv, installs torch with CUDA support, then installs the
REM base requirements and the spaCy English model. Safe to re-run.

set VENV_DIR=venv
set TORCH_INDEX=https://download.pytorch.org/whl/cu124
set FFMPEG_DIR=tools
set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip

echo.
echo === SubForge Windows setup ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python was not found on PATH.
    echo Install Python 3.11+ from https://www.python.org/downloads/ and re-run.
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [1/7] Creating virtual environment in %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo [1/7] Reusing existing virtual environment in %VENV_DIR%.
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

echo [2/7] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [3/7] Installing torch + torchaudio (CUDA 12.4 build)...
pip install torch torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 exit /b 1

REM Bundle CUDA runtime libs through pip so ctranslate2 (used by faster-whisper)
REM finds cuBLAS / cuDNN regardless of the system-wide CUDA install.
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
if errorlevel 1 exit /b 1

echo [4/7] Installing base requirements...
pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [5/7] Installing subforge (editable)...
pip install -e ".[full]"
if errorlevel 1 exit /b 1

echo [6/7] Downloading spaCy English model...
python -m spacy download en_core_web_sm
if errorlevel 1 exit /b 1

REM --- ffmpeg ---
REM Check system PATH first, then local tools/ directory.
set "PATH=%~dp0..\%FFMPEG_DIR%;%PATH%"
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [7/7] Downloading ffmpeg...
    if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"
    powershell -Command "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_DIR%\ffmpeg.zip'"
    if errorlevel 1 (
        echo [ERROR] Failed to download ffmpeg.
        exit /b 1
    )
    powershell -Command "Expand-Archive -Path '%FFMPEG_DIR%\ffmpeg.zip' -DestinationPath '%FFMPEG_DIR%' -Force"
    if errorlevel 1 (
        echo [ERROR] Failed to extract ffmpeg.
        exit /b 1
    )
    REM Move binaries up to tools/ and clean up
    for /d %%D in ("%FFMPEG_DIR%\ffmpeg-*") do (
        copy "%%D\bin\ffmpeg.exe" "%FFMPEG_DIR%\" >nul
        copy "%%D\bin\ffprobe.exe" "%FFMPEG_DIR%\" >nul
        rd /s /q "%%D"
    )
    del "%FFMPEG_DIR%\ffmpeg.zip"
    echo ffmpeg installed to %FFMPEG_DIR%\
) else (
    echo [7/7] ffmpeg already available, skipping.
)

REM Mark setup as complete so SubForge.bat knows to skip setup next time
echo.> .setup_done

echo.
echo === Setup complete ===
echo.
echo Activate the environment in a new shell with:
echo     %VENV_DIR%\Scripts\activate
echo.
echo Then launch the app:
echo     subforge
echo     subforge-cli --url "https://..."
echo.
echo To install the experimental NeMo / Parakeet backend:
echo     pip install -r requirements-experimental.txt
echo.

endlocal
