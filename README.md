# Auto Subtitle

Easily generate clean, accurate subtitles from any YouTube link — powered by state-of-the-art speech recognition and smart post-processing.

---

## 🎯 What It Does

- 🎧 Just paste a YouTube link — the app downloads audio and transcribes it
- 🧠 Backed by NVIDIA’s top ASR model [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- ✍️ Automatically restores punctuation for natural reading
- 🧾 Splits overly long ASR output into sentence-like segments using NLP
- 🔇 Fixes common loop/noise gibberish with background cleanup
- 📝 Outputs a ready-to-use `.srt` subtitle file

---

## 🧠 Why It Helps

ASR models are powerful, but often give you:

- One **massive block** of text — hard to subtitle or read
- Garbled loop segments when background noise kicks in

This app cleans it up for you:

- 🔪 Chunks long lines using sentence logic (with spaCy + custom rules)
- 🎚️ Runs optional noise removal with Demucs for cleaner input
- 💬 Gives you clean, natural subtitles with proper timecodes

---

## 🚧 Coming Soon

- 🌍 Auto-translate and generate **dual-language subtitles**
- 🇯🇵 Japanese speech recognition model support

---

## 🛠️ Setup

Requires Python 3.11+.

```bash
# Clone this repo
git clone https://github.com/tomchang25/auto-subtitle.git
cd auto-subtitle
```

### One-click install

The setup scripts create a `venv/`, install the right `torch` build for your
platform (CUDA 12.4 on Windows/Linux, MPS on macOS), install the base
dependencies, and download the spaCy English model. Re-running the script on
an existing venv is safe.

```bat
:: Windows
setup.bat
```

```bash
# macOS / Linux
./setup.sh
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
```

### Experimental: NeMo / Parakeet backend

The default backend is `faster-whisper`. To opt into the larger NVIDIA NeMo
Parakeet backend:

```bash
pip install -r requirements-experimental.txt
# or, when installing the project itself:
pip install -e ".[experimental]"
```

---

## 🚀 Usage

```bash
python youtube_subtitle_app/main.py
```

You’ll be prompted to paste a YouTube URL. The program will:

1. Download the audio
2. Transcribe it
3. Restore punctuation
4. Break into aligned subtitle chunks
5. Export an `.srt` file under `~/Documents/AutoSubtitle/<video title>/output.srt`

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

You’ll need test assets under `tests/data/` for some integration tests.

---

## 🧠 Credits

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) – speech recognition
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) – YouTube audio downloading
- [spaCy](https://spacy.io) – NLP sentence processing
- [DeepMultilingualPunctuation](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large)

---

## 🪪 License

MIT License
