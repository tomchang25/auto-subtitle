"""Language-specific configuration profiles for the NLP pipeline.

Each LanguageProfile carries the parameters that differ between languages:
word joining, sentence splitting strategy, break words, punctuation sets,
and segmentation thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass


def _is_cjk_char(ch: str) -> bool:
    """Return True if *ch* is a CJK unified ideograph or common CJK symbol."""
    cp = ord(ch)
    # CJK Unified Ideographs and extensions
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
        # Katakana / Hiragana
        or (0x3040 <= cp <= 0x309F)
        or (0x30A0 <= cp <= 0x30FF)
        # Hangul Syllables
        or (0xAC00 <= cp <= 0xD7AF)
    )


def _has_cjk(text: str) -> bool:
    """Return True if *text* contains any CJK character."""
    return any(_is_cjk_char(ch) for ch in text)


@dataclass(frozen=True)
class LanguageProfile:
    """Holds all language-specific knobs consumed by the NLP pipeline."""

    code: str  # ISO 639-1 code, e.g. "en", "zh", "ja"

    # --- tokenisation / joining ---
    join_token: str = " "
    use_spacy: bool = True
    spacy_model: str = "en_core_web_sm"

    # --- segmentation ---
    break_words: frozenset[str] = frozenset()
    punctuation: frozenset[str] = frozenset()
    sentence_end: frozenset[str] = frozenset()

    # Use character count instead of word count for thresholds
    use_char_count: bool = False
    seg_min: int = 4
    seg_soft: int = 8
    seg_hard: int = 15
    merge_max: int = 12

    # --- punctuation restoration ---
    skip_punctuation_model: bool = False


# ---------------------------------------------------------------------------
# Pre-built profiles
# ---------------------------------------------------------------------------

_ASCII_PUNCT = frozenset(",.!?;:")
_ASCII_SENTENCE_END = frozenset(".!?")

ENGLISH = LanguageProfile(
    code="en",
    join_token=" ",
    use_spacy=True,
    spacy_model="en_core_web_sm",
    break_words=frozenset({
        "but", "and", "so", "because", "like", "then",
        "right", "now", "anyway", "although", "however",
    }),
    punctuation=_ASCII_PUNCT,
    sentence_end=_ASCII_SENTENCE_END,
    use_char_count=False,
    seg_min=4,
    seg_soft=8,
    seg_hard=15,
    merge_max=12,
    skip_punctuation_model=False,
)

_CJK_PUNCT = frozenset("，。！？、；：,.!?;:")
_CJK_SENTENCE_END = frozenset("。！？.!?")

CHINESE = LanguageProfile(
    code="zh",
    join_token="",
    use_spacy=False,
    spacy_model="",
    break_words=frozenset({
        "但是", "而且", "所以", "因為", "因为", "然後", "然后",
        "不過", "不过", "雖然", "虽然", "然而", "那麼", "那么",
        "可是", "就是", "於是", "于是",
    }),
    punctuation=_CJK_PUNCT,
    sentence_end=_CJK_SENTENCE_END,
    use_char_count=True,
    seg_min=6,       # ~6 characters minimum
    seg_soft=15,     # start looking for breaks after 15 chars
    seg_hard=30,     # hard cut at 30 chars
    merge_max=25,    # max chars after merge
    skip_punctuation_model=True,  # Whisper already punctuates Chinese well
)

JAPANESE = LanguageProfile(
    code="ja",
    join_token="",
    use_spacy=False,
    spacy_model="",
    break_words=frozenset({
        "しかし", "そして", "だから", "けど", "それで",
        "でも", "ただ", "なので", "それから",
    }),
    punctuation=frozenset("、。！？，,.!?;:"),
    sentence_end=frozenset("。！？.!?"),
    use_char_count=True,
    seg_min=6,
    seg_soft=15,
    seg_hard=30,
    merge_max=25,
    skip_punctuation_model=True,
)

KOREAN = LanguageProfile(
    code="ko",
    join_token=" ",  # Korean uses spaces between words
    use_spacy=False,
    spacy_model="",
    break_words=frozenset({
        "그런데", "하지만", "그래서", "그리고", "그러면",
        "그러나", "왜냐하면", "그러므로",
    }),
    punctuation=frozenset(",.!?;:"),
    sentence_end=frozenset(".!?"),
    use_char_count=False,
    seg_min=4,
    seg_soft=8,
    seg_hard=15,
    merge_max=12,
    skip_punctuation_model=True,
)

# Fallback: use English defaults
DEFAULT = ENGLISH

# Lookup by Whisper language code (ISO 639-1)
_PROFILES: dict[str, LanguageProfile] = {
    "en": ENGLISH,
    "zh": CHINESE,
    "ja": JAPANESE,
    "ko": KOREAN,
}


def get_profile(lang_code: str) -> LanguageProfile:
    """Return the LanguageProfile for a Whisper language code.

    Falls back to ENGLISH for unknown languages.
    """
    return _PROFILES.get(lang_code, DEFAULT)


def detect_profile_from_text(text: str) -> LanguageProfile:
    """Heuristic fallback: guess profile from text content."""
    if _has_cjk(text):
        # Could be Chinese or Japanese — default to Chinese
        return CHINESE
    return ENGLISH
