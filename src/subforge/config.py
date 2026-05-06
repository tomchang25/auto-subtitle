import logging
from pathlib import Path

# Logging
LOG_LEVEL = logging.DEBUG  # set to logging.DEBUG for full prompts/responses

# Video URL
DEFAULT_URL = "https://www.youtube.com/watch?v=ByreRudsyoc"

# ASR/NLP Models
WHISPER_MODEL = "large-v3-turbo"
SPACY_MODEL = "en_core_web_sm"

# ASR backend selection
ASR_BACKEND = "auto"          # "auto" | "whisper" | "funasr"
ASR_SOURCE_LANGUAGE = "auto"  # ISO 639-1 source language hint, or "auto"

# Subtitle formatting
MAX_GAP = 3.0
MIN_DURATION = 1.5
BREATH_GAP = 0.3
MIN_WORDS_FOR_BREATH_SPLIT = 8

# Segmentation thresholds
SEG_PAUSE_THRESHOLD = 0.25  # timing gap (seconds) treated as a cut opportunity
# NOTE: word/char count thresholds (seg_min, seg_soft, seg_hard, merge_max)
# are defined per-language in nlp/lang_profile.py

# Merge thresholds (soft merge after splitting)
MERGE_MAX_DURATION = 4.0    # don't merge if combined duration > this (seconds)
MERGE_MAX_GAP = 1.0         # don't merge if gap between segments > this (seconds)

# Punctuation restoration (LLM-based, optional)
USE_LLM_PUNCTUATION = True

# Translation
TRANSLATE_METHOD: str | None = None  # None = disabled by default
TRANSLATE_SRC_LANG = "eng_Latn"  # NLLB language code
TRANSLATE_TGT_LANG = "zho_Hant"  # FLORES language code (default)

# Supported target languages (code → display name)
# Keys use FLORES-200 / NLLB language codes
TARGET_LANGUAGES: dict[str, str] = {
    "zho_Hant": "繁體中文",
    "zho_Hans": "简体中文",
    "jpn_Jpan": "日本語",
    "kor_Hang": "한국어",
    "fra_Latn": "Français",
    "deu_Latn": "Deutsch",
    "spa_Latn": "Español",
    "por_Latn": "Português",
    "vie_Latn": "Tiếng Việt",
    "tha_Thai": "ไทย",
    "eng_Latn": "English",
}

# Short code for output filenames (e.g. output_zh-Hant.srt)
TARGET_LANG_SHORT: dict[str, str] = {
    "zho_Hant": "zh-Hant",
    "zho_Hans": "zh-Hans",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "spa_Latn": "es",
    "por_Latn": "pt",
    "vie_Latn": "vi",
    "tha_Thai": "th",
    "eng_Latn": "en",
}

# Paths
OUTPUT_DIR = Path.home() / "Documents" / "SubForge"

# Demucs output structure
DEMUCS_MODEL = "htdemucs"
DEMUCS_VOCALS_FILENAME = "vocals.mp3"
DEMUCS_CHUNK_MINUTES = 30  # split audio into chunks of this length for demucs
