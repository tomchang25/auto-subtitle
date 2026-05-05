# SubForge

Generate clean, accurate subtitles from any YouTube link — powered by state-of-the-art speech recognition and smart post-processing.

---

## What It Does

- Paste a YouTube link — the app downloads audio and transcribes it
- Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with built-in punctuation
- Splits overly long ASR output into sentence-like segments using NLP
- Fixes common loop/noise gibberish with background cleanup
- Outputs a ready-to-use `.srt` subtitle file

---

## Why It Helps

ASR models are powerful, but often give you:

- One massive block of text — hard to subtitle or read
- Garbled loop segments when background noise kicks in

This app cleans it up for you:

- Chunks long lines using sentence logic (with spaCy + custom rules)
- Runs optional noise removal with Demucs for cleaner input
- Gives you clean, natural subtitles with proper timecodes

---

## Coming Soon

- Auto-translate and generate dual-language subtitles
- Japanese speech recognition model support

---

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/tomchang25/subforge.git
cd subforge
```

### One-click install

The setup scripts create a `venv/`, install the right `torch` build for your
platform (CUDA 12.4 on Windows/Linux, MPS on macOS), install the base
dependencies, and download the spaCy English model. Re-running the script on
an existing venv is safe.

```bat
:: Windows
scripts/setup.bat
```

```bash
# macOS / Linux
./scripts/setup.sh
```

### Manual install

If you prefer to manage the environment yourself:

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install torch + torchaudio for your platform first.
# Linux / Windows with NVIDIA GPU:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# macOS:
# pip install torch torchaudio

pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

---

## Usage

```bash
# GUI
subforge

# CLI
subforge-cli --url "https://www.youtube.com/watch?v=..."
subforge-cli --url "..." --model large-v3 --download-video --video-quality 720p
```

The program will:

1. Download the audio
2. Transcribe it (with punctuation from faster-whisper)
3. Break into aligned subtitle chunks
4. Export an `.srt` file under `~/Documents/SubForge/<video title>/output.srt`

---

## Run Tests

```bash
pytest tests/ -v
```

You'll need test assets under `tests/data/` for some integration tests.

---

## Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) – speech recognition
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) – YouTube audio downloading
- [spaCy](https://spacy.io) – NLP sentence processing

---

## License

MIT License
