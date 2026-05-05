#!/usr/bin/env bash
# One-click installer for auto-subtitle on macOS / Linux.
# Creates a venv, installs the right torch build for the host (CUDA on Linux,
# MPS on macOS), then installs the base requirements and the spaCy model.
# Safe to re-run.

set -e

VENV_DIR="venv"
TORCH_INDEX_LINUX="https://download.pytorch.org/whl/cu124"

echo
echo "=== auto-subtitle setup ==="
echo

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "[ERROR] python3 was not found on PATH."
    echo "Install Python 3.11+ and re-run."
    exit 1
fi

OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM=mac ;;
    Linux)  PLATFORM=linux ;;
    *)      PLATFORM=other ;;
esac
echo "Detected platform: $PLATFORM ($OS)"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[1/5] Creating virtual environment in $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "[1/5] Reusing existing virtual environment in $VENV_DIR."
fi

# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"

echo "[2/5] Upgrading pip..."
python -m pip install --upgrade pip

echo "[3/5] Installing torch + torchaudio..."
if [ "$PLATFORM" = "mac" ]; then
    # macOS wheels ship with MPS support built in; no special index needed.
    pip install torch torchaudio
else
    # Linux: pull the CUDA 12.4 build of torch.
    pip install torch torchaudio --index-url "$TORCH_INDEX_LINUX"
    # Bundle CUDA runtime libs through pip so ctranslate2 (used by
    # faster-whisper) finds cuBLAS / cuDNN regardless of the system
    # CUDA install.
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
fi

echo "[4/5] Installing base requirements..."
pip install -r requirements.txt

echo "[5/5] Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo
echo "=== Setup complete ==="
echo
echo "Activate the environment in a new shell with:"
echo "    source $VENV_DIR/bin/activate"
echo
echo "Then launch the app:"
echo "    python gui.py"
echo "    python youtube_subtitle_app/main.py"
echo
echo "To install the experimental NeMo / Parakeet backend:"
echo "    pip install -r requirements-experimental.txt"
echo
