import logging
from pathlib import Path

# Logging
LOG_LEVEL = logging.DEBUG  # set to logging.DEBUG for full prompts/responses

# Video URL
DEFAULT_URL = "https://www.youtube.com/watch?v=ByreRudsyoc"

# ASR/NLP Models
WHISPER_MODEL = "large-v3-turbo"
SPACY_MODEL = "en_core_web_sm"

# Subtitle formatting
MAX_GAP = 3.0
MIN_DURATION = 1.5
BREATH_GAP = 0.3
MIN_WORDS_FOR_BREATH_SPLIT = 8

# Segmentation thresholds
SEG_MIN_WORDS = 4       # never create a segment shorter than this
SEG_SOFT_WORDS = 8      # after this many words, cut at next punctuation/pause
SEG_HARD_WORDS = 15     # hard cut regardless
SEG_PAUSE_THRESHOLD = 0.25  # timing gap (seconds) treated as a cut opportunity

# Merge thresholds (soft merge after splitting)
MERGE_MAX_WORDS = 12        # don't merge if combined > this
MERGE_MAX_DURATION = 4.0    # don't merge if combined duration > this (seconds)
MERGE_MAX_GAP = 1.0         # don't merge if gap between segments > this (seconds)

# Translation
TRANSLATE_METHOD: str | None = None  # None = disabled by default
TRANSLATE_SRC_LANG = "eng_Latn"  # NLLB language code
TRANSLATE_TGT_LANG = "zho_Hant"  # FLORES language code

# Paths
OUTPUT_DIR = Path.home() / "Documents" / "SubForge"

# Demucs output structure
DEMUCS_MODEL = "htdemucs"
DEMUCS_VOCALS_FILENAME = "vocals.mp3"
DEMUCS_CHUNK_MINUTES = 30  # split audio into chunks of this length for demucs
