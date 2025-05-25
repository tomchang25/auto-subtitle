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

```bash
# Clone this repo
git clone https://github.com/tomchang25/auto-subtitle.git
cd auto-subtitle

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # on Windows
# source venv/bin/activate  # on macOS/Linux

# Install dependencies
pip install -e .
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
