from pathlib import Path

# Video URL
DEFAULT_URL = "https://www.youtube.com/watch?v=392JUMCBSQY"

# ASR/NLP Models
WHISPER_MODEL = "large-v3-turbo"
SPACY_MODEL = "en_core_web_sm"

# Subtitle formatting
MAX_WORDS = 20
SOFT_LIMIT = 5
MAX_GAP = 3.0
MIN_DURATION = 1.5

# Paths
OUTPUT_DIR = Path.home() / "Documents" / "AutoSubtitle"
