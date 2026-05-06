"""Tests for multi-language support (LanguageProfile + CJK pipeline)."""

from subforge.nlp.lang_profile import (
    CHINESE,
    ENGLISH,
    get_profile,
    detect_profile_from_text,
)

from subforge.nlp.text_semantically import split_word_segments_by_punctuation
from subforge.nlp.segmentation import (
    split_long_sentences_by_length,
    merge_short_segments,
)
from subforge.utils import get_bounds_and_text


# ---------------------------------------------------------------------------
# LanguageProfile lookup
# ---------------------------------------------------------------------------

def test_get_profile_english():
    p = get_profile("en")
    assert p.code == "en"
    assert p.join_token == " "
    assert p.use_spacy is True
    assert p.use_char_count is False


def test_get_profile_chinese():
    p = get_profile("zh")
    assert p.code == "zh"
    assert p.join_token == ""
    assert p.use_spacy is False
    assert p.use_char_count is True


def test_get_profile_unknown_falls_back_to_english():
    p = get_profile("xx")
    assert p.code == "en"


def test_detect_profile_from_cjk_text():
    p = detect_profile_from_text("This is CJK text: xxx")
    assert p.code == "en"


def test_detect_profile_from_english_text():
    p = detect_profile_from_text("This is English text")
    assert p.code == "en"


# ---------------------------------------------------------------------------
# CJK word-segment splitting
# ---------------------------------------------------------------------------

def _make_seg(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_split_word_segments_by_punctuation_basic():
    segs = [
        _make_seg("Hello", 0.0, 0.5),
        _make_seg("World.", 0.5, 1.0),
        _make_seg("How", 1.0, 1.5),
        _make_seg("are", 1.5, 2.0),
        _make_seg("you?", 2.0, 2.5),
    ]
    chunks = split_word_segments_by_punctuation(segs, CHINESE)
    assert len(chunks) == 2


def test_split_word_segments_chinese():
    segs = [
        _make_seg("Hello", 0.0, 0.5),
        _make_seg("World", 0.5, 1.0),
    ]
    chunks = split_word_segments_by_punctuation(segs, CHINESE)
    assert len(chunks) == 1
    assert len(chunks[0]) == 2


def test_cjk_chunks_have_timestamps():
    segs = [
        _make_seg("Hi.", 0.0, 0.5),
        _make_seg("Bye.", 1.0, 1.5),
    ]
    chunks = split_word_segments_by_punctuation(segs, CHINESE)
    assert chunks[0][0]["start"] == 0.0
    assert chunks[0][0]["end"] == 0.5
    assert chunks[1][0]["start"] == 1.0


# ---------------------------------------------------------------------------
# CJK joining -- no spaces
# ---------------------------------------------------------------------------

def test_get_bounds_and_text_cjk_no_spaces():
    chunk = [[
        {"text": "Hello", "start": 0.0, "end": 0.5},
        {"text": "World", "start": 0.5, "end": 1.0},
    ]]
    bounds = get_bounds_and_text(chunk, profile=CHINESE)
    assert bounds[0]["segment"] == "HelloWorld"


def test_get_bounds_and_text_english_has_spaces():
    chunk = [[
        {"text": "Hello", "start": 0.0, "end": 0.5},
        {"text": "world.", "start": 0.5, "end": 1.0},
    ]]
    bounds = get_bounds_and_text(chunk, profile=ENGLISH)
    assert bounds[0]["segment"] == "Hello world."


# ---------------------------------------------------------------------------
# Segmentation with char-count mode
# ---------------------------------------------------------------------------

def test_segmentation_char_count_mode():
    """CJK segmentation should use character count, not word count."""
    words = [_make_seg("ab", i * 0.1, (i + 1) * 0.1) for i in range(20)]
    chunks = split_word_segments_by_punctuation(words, CHINESE)
    assert len(chunks) == 1

    split = split_long_sentences_by_length(
        chunks,
        min_words=CHINESE.seg_min,
        max_words=CHINESE.seg_hard,
        soft_words=CHINESE.seg_soft,
        profile=CHINESE,
    )
    # 20 tokens x 2 chars = 40 chars > seg_hard=30, and remainder >= seg_min=6
    assert len(split) >= 2


def test_merge_respects_sentence_end_cjk():
    chunk1 = [{"text": "ok.", "start": 0.0, "end": 0.5}]
    chunk2 = [{"text": "hi", "start": 0.6, "end": 1.0}]
    merged = merge_short_segments(
        [chunk1, chunk2],
        max_words=25,
        max_duration=4.0,
        max_gap=1.0,
        profile=CHINESE,
    )
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# Profile punctuation sets
# ---------------------------------------------------------------------------

def test_chinese_profile_has_cjk_punctuation():
    for ch in list(",.!?;:"):
        assert ch in CHINESE.punctuation
    for ch in list(",!?;:"):
        assert ch in CHINESE.punctuation


def test_english_profile_has_ascii_punctuation():
    for ch in list(",.!?;:"):
        assert ch in ENGLISH.punctuation, f"'{ch}' not in ENGLISH.punctuation"


def test_japanese_profile():
    p = get_profile("ja")
    assert p.code == "ja"
    assert p.join_token == ""
    assert p.use_char_count is True


def test_korean_profile():
    p = get_profile("ko")
    assert p.code == "ko"
    assert p.join_token == " "
    assert p.use_char_count is False
