"""Tests for hard_cut_chinese_segments (Chinese benchmark mode)."""
from subforge.nlp.chinese_benchmark import hard_cut_chinese_segments


def _segs(*words_and_times):
    """Build a list of word_segment dicts from (word, start, end) triples."""
    return [{"word": w, "start": s, "end": e} for w, s, e in words_and_times]


def _text(chunk):
    return "".join(tok["text"] for tok in chunk)


# ---------------------------------------------------------------------------
# Basic cuts
# ---------------------------------------------------------------------------


def test_sentence_end_punctuation_cuts():
    segs = _segs(
        ("今天", 0.0, 0.5),
        ("天氣", 0.5, 1.0),
        ("很好。", 1.0, 1.5),
        ("我們", 1.5, 2.0),
        ("出去", 2.0, 2.5),
        ("玩吧。", 2.5, 3.0),
    )
    chunks = hard_cut_chinese_segments(segs, hard_chars=100, gap_seconds=10.0)
    assert len(chunks) == 2
    assert _text(chunks[0]) == "今天天氣很好。"
    assert _text(chunks[1]) == "我們出去玩吧。"


def test_hard_char_cut():
    # 5 segments of 6 chars each → hard_chars=15 should cut after the 3rd
    segs = _segs(
        ("abcdef", 0.0, 0.2),
        ("ghijkl", 0.2, 0.4),
        ("mnopqr", 0.4, 0.6),
        ("stuvwx", 0.6, 0.8),
        ("yzabcd", 0.8, 1.0),
    )
    chunks = hard_cut_chinese_segments(segs, hard_chars=15, gap_seconds=10.0)
    # First cut at 18 chars (>= 15 after 3rd token)
    assert len(chunks) >= 2
    # All text must be preserved
    all_text = "".join(_text(c) for c in chunks)
    assert all_text == "abcdefghijklmnopqrstuvwxyzabcd"


def test_gap_based_cut():
    segs = _segs(
        ("你好", 0.0, 0.5),
        ("世界", 0.5, 1.0),
        # 3-second gap
        ("再見", 4.0, 4.5),
        ("朋友", 4.5, 5.0),
    )
    chunks = hard_cut_chinese_segments(segs, hard_chars=100, gap_seconds=1.5)
    assert len(chunks) == 2
    assert _text(chunks[0]) == "你好世界"
    assert _text(chunks[1]) == "再見朋友"


def test_no_cut_below_gap_threshold():
    segs = _segs(
        ("你好", 0.0, 0.5),
        ("世界", 1.0, 1.5),  # 0.5s gap — below threshold of 1.5s
    )
    chunks = hard_cut_chinese_segments(segs, hard_chars=100, gap_seconds=1.5)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Timestamp safety
# ---------------------------------------------------------------------------


def test_skips_invalid_interval_end_equals_start():
    segs = _segs(
        ("有效", 0.0, 0.5),
        ("無效", 1.0, 1.0),  # end == start → skip
        ("也有效", 1.5, 2.0),
    )
    chunks = hard_cut_chinese_segments(segs, hard_chars=100, gap_seconds=10.0)
    all_text = "".join(_text(c) for c in chunks)
    assert "無效" not in all_text
    assert "有效" in all_text
    assert "也有效" in all_text


def test_skips_invalid_interval_end_before_start():
    segs = _segs(
        ("有效", 0.0, 0.5),
        ("倒轉", 2.0, 1.0),  # end < start → skip
        ("也有效", 2.5, 3.0),
    )
    chunks = hard_cut_chinese_segments(segs, hard_chars=100, gap_seconds=10.0)
    all_text = "".join(_text(c) for c in chunks)
    assert "倒轉" not in all_text


def test_warns_on_non_monotonic_timestamps(caplog):
    import logging

    segs = _segs(
        ("第一", 0.0, 1.0),
        ("第二", 0.5, 1.5),  # start (0.5) < prev_end (1.0)
    )
    with caplog.at_level(logging.WARNING, logger="subforge.nlp.chinese_benchmark"):
        chunks = hard_cut_chinese_segments(segs, hard_chars=100, gap_seconds=10.0)
    # Text is preserved despite non-monotonic timestamps
    all_text = "".join(_text(c) for c in chunks)
    assert "第二" in all_text
    assert any("Non-monotonic" in r.message for r in caplog.records)


def test_each_chunk_has_valid_interval():
    segs = _segs(
        ("我", 0.0, 0.3),
        ("愛", 0.3, 0.6),
        ("你。", 0.6, 1.0),
        ("你", 1.0, 1.3),
        ("好嗎？", 1.3, 1.8),
    )
    chunks = hard_cut_chinese_segments(segs, hard_chars=100, gap_seconds=10.0)
    for chunk in chunks:
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        assert end > start, f"Chunk has invalid interval: [{start}, {end}]"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input():
    assert hard_cut_chinese_segments([]) == []


def test_single_segment():
    segs = _segs(("你好", 0.0, 0.5))
    chunks = hard_cut_chinese_segments(segs)
    assert len(chunks) == 1
    assert _text(chunks[0]) == "你好"


def test_all_invalid_segments_returns_empty():
    segs = _segs(
        ("壞的", 1.0, 0.5),
        ("也壞", 2.0, 2.0),
    )
    assert hard_cut_chinese_segments(segs) == []


def test_token_fields():
    segs = _segs(("好。", 0.0, 0.5))
    chunks = hard_cut_chinese_segments(segs)
    tok = chunks[0][0]
    assert tok["text"] == "好。"
    assert tok["whitespace"] == ""
    assert tok["start"] == 0.0
    assert tok["end"] == 0.5
    assert isinstance(tok["is_punct"], bool)


def test_text_fully_preserved_across_cuts():
    segs = _segs(
        ("今天", 0.0, 0.5),
        ("是個", 0.5, 1.0),
        ("好日子。", 1.0, 1.5),
        ("讓我們", 1.5, 2.0),
        ("好好利用", 2.0, 2.5),
        ("這段時間吧。", 2.5, 3.0),
    )
    chunks = hard_cut_chinese_segments(segs, hard_chars=100, gap_seconds=10.0)
    all_text = "".join(_text(c) for c in chunks)
    assert all_text == "今天是個好日子。讓我們好好利用這段時間吧。"
