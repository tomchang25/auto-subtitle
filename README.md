<p align="center">
  <h1 align="center">SubForge</h1>
  <p align="center">
    Offline-first subtitle generation with intelligent segmentation — powered by Whisper, spaCy, and Demucs.
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#language-support">Language Support</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#contributing">Contributing</a> •
    <a href="#license">License</a>
  </p>
</p>

---

## Why SubForge?

Most subtitle tools stop at speech recognition — paste audio into Whisper and ship the raw output. The result is usable, but far from broadcast-ready: overly long lines, unnatural breaks mid-sentence, missing punctuation, and garbled segments from background noise.

SubForge bridges the gap between **raw ASR output** and **human-edited subtitles** with a multi-stage post-processing pipeline that runs entirely on your machine — no API keys required for the core workflow.

### How SubForge Compares

| Capability              | Typical SaaS Tools    | LLM-based Tools                     | SubForge                         |
| ----------------------- | --------------------- | ----------------------------------- | -------------------------------- |
| Speech recognition      | Whisper / proprietary | Whisper                             | Whisper (faster-whisper)         |
| Smart segmentation      | ❌ Raw ASR segments   | ✅ LLM-based (requires API)         | ✅ NLP rule-based (offline)      |
| Punctuation restoration | ❌                    | ✅ Via LLM                          | ✅ Local transformer model       |
| Vocal isolation         | ❌                    | ✅ Optional                         | ✅ Demucs built-in               |
| Translation             | Machine translation   | LLM / Bing / Google                 | NLLB (local) / Gemini (API)      |
| Fully offline           | ❌                    | ❌ (needs LLM API for segmentation) | ✅ Core pipeline is 100% offline |
| Deterministic output    | N/A                   | ❌ (LLM variance)                   | ✅ Same input → same output      |
| Cost per run            | Per-minute pricing    | API token cost                      | Free (local compute only)        |

## Features

- **Intelligent Segmentation** — NLP-driven sentence splitting using spaCy with timing-aware heuristics: breath gaps, pause detection, break-word boundaries, and soft/hard word-count limits. No LLM required.
- **Timestamp Alignment** — Word-level timestamps from faster-whisper are preserved through every pipeline stage. Segments split and merge without losing sync.
- **Vocal Isolation** — Built-in Demucs integration separates vocals from background music/noise before transcription. Automatically chunks long files to manage memory.
- **Punctuation Restoration** — A local transformer model adds missing punctuation where Whisper didn't, while preserving Whisper's own punctuation where it's more accurate.
- **Bilingual Subtitles** — Translate and output dual-language SRT files. Supports NLLB (fully offline) and Gemini (API) backends.
- **YouTube Integration** — Paste a URL to download audio (and optionally video) via yt-dlp.
- **GUI & CLI** — PySide6 desktop app with progress tracking, or a full-featured command-line interface.
- **Caching & Resumability** — Each pipeline stage checkpoints its output. Re-runs skip completed steps automatically.

## Language Support

> **Current scope:** SubForge is optimized for **English-language audio**. The NLP pipeline
> (spaCy sentence splitting, break-word rules) is English-only. Non-English transcription
> is on the [roadmap](#roadmap).

| Pipeline Stage          | Language Support         | Notes                                            |
| ----------------------- | ------------------------ | ------------------------------------------------ |
| Transcription (Whisper) | English ¹                | Whisper supports 99+ languages; others untested   |
| Punctuation Restoration | Multilingual             | fullstop-punctuation-multilang-large              |
| Sentence Splitting      | **English only**         | Hardcoded to spaCy `en_core_web_sm`               |
| Segmentation            | English only             | Break-word list is English-specific               |
| Translation (NLLB)      | → 11 target languages ² | Sentence-level — quality is limited               |
| Translation (Gemini)    | → 11 target languages ² | Block-level — better quality, may misalign output |

¹ Other languages may work but are not tested or officially supported.  
² Target languages: Traditional Chinese, Simplified Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Vietnamese, Thai, English.

## How It Works

```
YouTube URL
    │
    ▼
┌──────────────┐
│  yt-dlp      │ Download audio (MP3)
└──────┬───────┘
       ▼
┌──────────────┐
│  Demucs      │ Vocal isolation (optional, chunked for long files)
└──────┬───────┘
       ▼
┌──────────────┐
│  ffmpeg      │ Convert to 16kHz mono WAV
└──────┬───────┘
       ▼
┌──────────────┐
│faster-whisper│ Word-level transcription with VAD
└──────┬───────┘
       ▼
┌──────────────┐
│  Punctuation │ Local transformer model restores missing punctuation
│  Restoration │
└──────┬───────┘
       ▼
┌──────────────┐
│  spaCy NLP   │ Sentence splitting with contraction & compound merging
└──────┬───────┘
       ▼
┌──────────────┐
│  Alignment   │ Map NLP tokens back to word-level timestamps
└──────┬───────┘
       ▼
┌──────────────┐
│  Refinement  │ Split by timing gaps, merge short segments
└──────┬───────┘
       ▼
┌──────────────┐
│ Segmentation │ Enforce min/soft/hard word limits with cut-strength logic
└──────┬───────┘
       ▼
┌──────────────┐
│  Translation │ NLLB (offline) or Gemini (API) — optional
└──────┬───────┘
       ▼
   output.srt
```

## Installation

**Requirements:**

- **Python 3.11+** — make sure to check "Add Python to PATH" during installation ([download](https://www.python.org/downloads/))
- **NVIDIA GPU** with CUDA support (recommended) — CPU-only mode works but is significantly slower.

> **Note:** ffmpeg is downloaded automatically during setup. If you already have ffmpeg on PATH, the bundled download is skipped.

### Windows — Double-Click Launch

```bash
git clone https://github.com/tomchang25/subforge.git
```

Double-click **`SubForge.bat`** in the project root. On the first run it will automatically:

1. Create a Python virtual environment
2. Install PyTorch with CUDA 12.4 support
3. Install all project dependencies
4. Download the spaCy English model
5. Download ffmpeg (if not already on PATH)
6. Launch the GUI

Subsequent launches skip setup and open the app directly. If setup is interrupted, the next launch will re-run setup from where it left off.

### macOS / Linux

```bash
git clone https://github.com/tomchang25/subforge.git
cd subforge
./scripts/setup.sh
```

The setup script creates a virtual environment, installs the correct PyTorch build for your platform (CUDA 12.4 on Linux, MPS on macOS), installs all dependencies, and downloads the spaCy English model.

### Manual Setup

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install PyTorch for your platform first:
# Linux / Windows with NVIDIA GPU:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# macOS:
# pip install torch torchaudio

pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

### Troubleshooting

If something goes wrong during setup, delete the `.setup_done` file in the project root and double-click `SubForge.bat` again to re-run setup.

## Usage

### GUI

Double-click **`SubForge.bat`** (Windows) or run:

```bash
subforge
```

### CLI

```bash
# Basic usage
subforge-cli --url "https://www.youtube.com/watch?v=..."

# With options
subforge-cli \
  --url "https://www.youtube.com/watch?v=..." \
  --model large-v3 \
  --translate gemini \
  --download-video \
  --video-quality 720p \
  --force
```

**CLI Options:**

| Flag               | Description                           | Default          |
| ------------------ | ------------------------------------- | ---------------- |
| `--url`            | YouTube URL (prompted if omitted)     | —                |
| `--model`          | Whisper model name                    | `large-v3-turbo` |
| `--translate`      | Translation backend: `nllb`, `gemini` | disabled         |
| `--download-video` | Also download the MP4 video           | `false`          |
| `--video-quality`  | Video quality: 480p–2160p             | `1080p`          |
| `--no-punctuation` | Disable punctuation restoration       | `false`          |
| `--force`          | Ignore cache, re-run all steps        | `false`          |
| `--debug`          | Enable DEBUG logging                  | `false`          |

Output is saved to `~/Documents/SubForge/<video title>/output.srt`.

## Configuration

Core parameters are in `src/subforge/config.py`:

**Segmentation**

| Parameter             | Default | Description                                     |
| --------------------- | ------- | ----------------------------------------------- |
| `SEG_MIN_WORDS`       | 4       | Never create a segment shorter than this        |
| `SEG_SOFT_WORDS`      | 8       | After this count, cut at next punctuation/pause |
| `SEG_HARD_WORDS`      | 15      | Hard cut regardless                             |
| `SEG_PAUSE_THRESHOLD` | 0.25s   | Timing gap treated as a cut opportunity         |

**Merge**

| Parameter            | Default | Description                                      |
| -------------------- | ------- | ------------------------------------------------ |
| `MERGE_MAX_WORDS`    | 12      | Don't merge if combined word count exceeds this  |
| `MERGE_MAX_DURATION` | 4.0s    | Don't merge if combined duration exceeds this    |
| `MERGE_MAX_GAP`      | 1.0s    | Don't merge if gap between segments exceeds this |

**Refinement**

| Parameter                    | Default | Description                                     |
| ---------------------------- | ------- | ----------------------------------------------- |
| `BREATH_GAP`                 | 0.3s    | Breathing pause threshold for splitting         |
| `MIN_WORDS_FOR_BREATH_SPLIT` | 8       | Only split at breath gaps if chunk is this long |
| `MIN_DURATION`               | 1.5s    | Minimum segment duration before merging         |

## Project Structure

```
src/subforge/
├── audio/              # Demucs wrapper, ffmpeg preprocessing
├── downloader/         # yt-dlp integration
├── llm/                # Gemini client with model fallback
├── nlp/                # Sentence splitting, alignment, segmentation
│   ├── text_semantically.py   # spaCy-based sentence chunking
│   ├── alignment.py           # Timestamp alignment & timing refinement
│   ├── segmentation.py        # Split/merge by word count & cut strength
│   └── punctuation.py         # Local punctuation restoration model
├── pipeline/           # Main processing pipeline with caching
├── subtitle/           # SRT formatting & writing
├── transcription/      # faster-whisper transcriber
├── translation/        # NLLB (offline) & Gemini backends
│   ├── aligner.py             # DP-based translation realignment
│   └── factory.py             # Backend registry
└── ui/                 # PySide6 GUI
```

## Roadmap

- [x] Auto-translate and generate dual-language subtitles
- [ ] Improve translation quality (reduce segment misalignment)
- [ ] Japanese speech recognition model support
- [ ] Speaker diarization
- [ ] Language-specific segmentation configs (break words, punctuation sets)
- [ ] Web UI alternative
- [ ] SRT style/formatting options (ASS/VTT export)

## Running Tests

```bash
pytest tests/ -v
```

Some integration tests require network access and are marked `slow`:

```bash
pytest tests/ -v -m slow    # include network tests
```

## Contributing

Contributions are welcome! Areas where help is especially appreciated:

- **Non-English segmentation rules** — break-word lists, punctuation sets, and spaCy model configs for other languages
- **ASR model backends** — additional transcription engines beyond faster-whisper
- **Translation quality** — improving the Gemini prompt or adding new translation backends
- **Bug reports** — especially around edge cases in long-form spoken content

Please open an issue before starting major work so we can discuss the approach.

## Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-accelerated Whisper inference
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube audio/video downloading
- [spaCy](https://spacy.io) — NLP tokenization and sentence processing
- [Demucs](https://github.com/facebookresearch/demucs) — Music source separation
- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) — Offline neural machine translation
- [fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large) — Punctuation restoration model

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2026 Greysuki
