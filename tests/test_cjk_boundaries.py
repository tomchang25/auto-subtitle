"""Tests for the rule-based CJK boundary restorer (Pre-Plan 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subforge.nlp.cjk_boundaries import (
    BOUNDARY_REASONS,
    BoundaryConfig,
    restore_cjk_boundaries,
)
from subforge.nlp.lang_profile import CHINESE, ENGLISH, JAPANESE, KOREAN
from subforge.pipeline.strategies import CjkPipelineStrategy, StrategyContext
from subforge.pipeline.strategies.cjk_models import CjkTimingAnchor, CjkTimingAnchors


def _segs(*words_and_times):
    return [{"word": w, "start": s, "end": e} for w, s, e in words_and_times]


def _ctx(tmp_path: Path, profile=CHINESE, **overrides):
    kwargs = dict(
        profile=profile,
        project_dir=tmp_path,
        force=False,
        emit=lambda step, detail="": None,
        check_cancel=lambda: None,
    )
    kwargs.update(overrides)
    return StrategyContext(**kwargs)


# ---------------------------------------------------------------------------
# Pure restorer behaviour
# ---------------------------------------------------------------------------


def test_existing_punctuation_preserved_and_marked():
    text = "你好。我是學生。"
    out, marks, diag = restore_cjk_boundaries(text, CHINESE)

    assert out == text
    assert diag["inserted_count"] == 0
    reasons = [m.reason for m in marks]
    assert reasons == ["existing", "existing"]
    assert diag["reason_counts"]["existing"] == 2


def test_fallback_appends_terminal_when_missing():
    out, marks, diag = restore_cjk_boundaries("你好我是學生", CHINESE)

    assert out == "你好我是學生。"
    assert any(m.reason == "fallback" for m in marks)
    assert diag["reason_counts"].get("fallback", 0) == 1


def test_length_break_inserts_phrase_then_sentence():
    # 32 unpunctuated chars — soft phrase break fires at 15, sentence cap
    # at 30 (counted from the previous *sentence* break, not phrase).
    text = "一二三四五六七八九十甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
    assert len(text) == 32

    out, marks, _diag = restore_cjk_boundaries(text, CHINESE)

    inserted = [m for m in marks if m.inserted]
    # At least: one phrase break, one sentence break (length cap), and the
    # fallback terminator at the very end.
    reasons = [m.reason for m in inserted]
    assert "length" in reasons
    assert "fallback" in reasons
    # The first inserted character should be the phrase punct (，).
    first_length = next(m for m in inserted if m.reason == "length")
    assert first_length.char in {"，", ","}
    # Output length grew by exactly the number of inserted chars.
    assert len(out) == len(text) + sum(1 for m in inserted)


def test_pause_triggers_phrase_break_when_short_span():
    text = "你好我們走"
    # Long silence between the 2nd and 3rd char.
    gaps: list[float | None] = [0.0, 1.5, 0.0, 0.0, None]
    cfg = BoundaryConfig(pause_threshold=0.6, min_chars_between_breaks=2)

    out, marks, _diag = restore_cjk_boundaries(text, CHINESE, gap_after=gaps, cfg=cfg)

    pause_marks = [m for m in marks if m.reason == "pause"]
    assert len(pause_marks) == 1
    # Span before the pause was short → phrase punct, not sentence-end.
    assert pause_marks[0].char == "，"
    assert "，" in out


def test_pause_promotes_to_sentence_when_span_long():
    text = "今天天氣特別好我們一起去公園走走"  # 16 chars
    gaps: list[float | None] = [0.0] * len(text)
    gaps[14] = 1.2  # long pause after char 15
    cfg = BoundaryConfig(
        pause_threshold=0.6,
        soft_phrase_chars=15,
        min_chars_between_breaks=2,
    )

    out, marks, _diag = restore_cjk_boundaries(text, CHINESE, gap_after=gaps, cfg=cfg)

    pause_marks = [m for m in marks if m.reason == "pause"]
    assert len(pause_marks) == 1
    # Long span → promoted to sentence-end.
    assert pause_marks[0].char == "。"


def test_min_chars_between_breaks_suppresses_close_breaks():
    text = "你好我們"
    # Pause that would otherwise insert a phrase break between the very
    # first chars; min_chars_between_breaks=4 should suppress it.
    gaps: list[float | None] = [1.0, 0.0, 0.0, None]
    cfg = BoundaryConfig(
        pause_threshold=0.5,
        min_chars_between_breaks=4,
    )
    out, marks, _diag = restore_cjk_boundaries(text, CHINESE, gap_after=gaps, cfg=cfg)
    # Only the fallback terminator should have been inserted.
    inserted_reasons = [m.reason for m in marks if m.inserted]
    assert inserted_reasons == ["fallback"]
    assert out == "你好我們。"


def test_mode_none_returns_input_unchanged():
    text = "你好我是學生"
    cfg = BoundaryConfig(mode="none")
    out, marks, diag = restore_cjk_boundaries(text, CHINESE, cfg=cfg)

    assert out == text
    assert marks == []
    assert diag["applied"] is False
    assert diag["mode"] == "none"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        restore_cjk_boundaries("你好", CHINESE, cfg=BoundaryConfig(mode="bogus"))


def test_empty_input_is_safe():
    out, marks, diag = restore_cjk_boundaries("", CHINESE)
    assert out == ""
    assert marks == []
    assert diag["applied"] is False


def test_existing_phrase_punct_does_not_count_as_sentence_break():
    # 18 chars + 1 comma; sentence budget should keep accumulating past
    # the comma until it reaches max_sentence_chars (default 30).
    text = "今天天氣很好，我們去公園走走玩遊戲"  # 17 chars before fallback
    out, marks, diag = restore_cjk_boundaries(text, CHINESE)

    # The comma is recorded as existing.
    assert any(m.reason == "existing" and m.char == "，" for m in marks)
    # No length-based sentence break fired (we stayed below 30 chars).
    inserted_reasons = [m.reason for m in marks if m.inserted]
    assert "length" not in inserted_reasons
    # Trailing fallback terminator appended.
    assert out.endswith("。")


def test_korean_uses_latin_punctuation_by_default():
    # Korean profile only knows ASCII punct; the restorer must not emit
    # full-width 「。」「，」 there.
    text = "안녕하세요반갑습니다오늘은좋은하루입니다정말로감사합니다"
    out, _marks, diag = restore_cjk_boundaries(text, KOREAN)
    assert "。" not in out
    assert "，" not in out
    # Some ASCII sentence-end appears at minimum from the fallback rule.
    assert out.endswith(".")
    assert diag["applied"] is True


def test_japanese_uses_full_width_punctuation():
    text = "あいうえおかきくけこさしすせそたちつてとなにぬねの"  # 25 hiragana
    out, _marks, _diag = restore_cjk_boundaries(text, JAPANESE)
    assert "，" in out or "、" in out or "。" in out  # at least one CJK punct


def test_diagnostics_reasons_in_documented_vocabulary():
    text = "今天天氣特別好我們一起去公園走走"
    gaps = [0.0] * len(text)
    gaps[7] = 1.0
    out, _marks, diag = restore_cjk_boundaries(
        text,
        CHINESE,
        gap_after=gaps,
        cfg=BoundaryConfig(min_chars_between_breaks=4),
    )
    for reason in diag["reason_counts"]:
        assert reason in BOUNDARY_REASONS
    # Every recorded mark's position is a valid index in the output text
    # and matches the recorded character.
    for mark in diag["marks"]:
        assert 0 <= mark["position"] < len(out)
        assert out[mark["position"]] == mark["char"]


def test_speaker_text_is_never_altered():
    """Inserted chars are punctuation only; speaker characters survive."""
    text = "今天天氣很好我們去公園散步吃飯"
    out, _marks, _diag = restore_cjk_boundaries(text, CHINESE)
    out_no_punct = "".join(
        ch for ch in out if ch not in CHINESE.punctuation
    )
    original_no_punct = "".join(
        ch for ch in text if ch not in CHINESE.punctuation
    )
    assert out_no_punct == original_no_punct


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _unpunctuated_segs(text: str, step: float = 0.2) -> list[dict]:
    return [
        {"word": ch, "start": i * step, "end": (i + 1) * step}
        for i, ch in enumerate(text)
    ]


def test_strategy_writes_boundaries_artifact(tmp_path):
    # Long unpunctuated transcript: the strategy should restore at least
    # one terminator and persist boundaries.json.
    text = "今天天氣特別好我們一起去公園散步並且野餐吧"
    segs = _unpunctuated_segs(text)
    strat = CjkPipelineStrategy()
    chunks = strat.run(segs, _ctx(tmp_path))

    boundaries = json.loads(
        (tmp_path / "cjk" / "boundaries.json").read_text(encoding="utf-8")
    )["data"]
    assert boundaries["text"].endswith("。")  # fallback terminator added
    assert boundaries["source"] == "boundary_restored"
    diag = boundaries["diagnostics"]
    assert diag["applied"] is True
    assert diag["mode"] == "rule"
    assert diag["inserted_count"] >= 1
    for reason in diag["reason_counts"]:
        assert reason in BOUNDARY_REASONS

    # final_cues.json should expose the same diagnostics for inspection.
    final = json.loads(
        (tmp_path / "cjk" / "final_cues.json").read_text(encoding="utf-8")
    )
    assert "boundary_restoration" in final["meta"]
    assert final["meta"]["boundary_restoration"]["applied"] is True

    # We should now produce *multiple* sentence-shaped cues, not a single
    # mega-cue.
    assert chunks
    joined = "".join(tok["text"] for chunk in chunks for tok in chunk)
    # Speaker text survives in the final output.
    for ch in text:
        assert ch in joined


def test_strategy_pause_triggers_boundary(tmp_path):
    # Two short fragments separated by a 1.5-second silence — the
    # restorer should split them into two sentences via a pause break.
    chars_a = "今天天氣很好"
    chars_b = "我們去公園走走"
    segs_a = _unpunctuated_segs(chars_a)
    last_end = segs_a[-1]["end"]
    pause_start = last_end + 1.5
    segs_b = [
        {"word": ch, "start": pause_start + i * 0.2, "end": pause_start + (i + 1) * 0.2}
        for i, ch in enumerate(chars_b)
    ]
    segs = segs_a + segs_b

    cfg = BoundaryConfig(
        mode="rule",
        max_sentence_chars=30,
        soft_phrase_chars=4,    # span before pause is 6 → promoted to sentence
        pause_threshold=0.6,
        min_chars_between_breaks=2,
    )
    strat = CjkPipelineStrategy(boundary_config=cfg)
    strat.run(segs, _ctx(tmp_path))

    boundaries = json.loads(
        (tmp_path / "cjk" / "boundaries.json").read_text(encoding="utf-8")
    )["data"]
    diag = boundaries["diagnostics"]
    assert diag["reason_counts"].get("pause", 0) >= 1
    sentences = json.loads(
        (tmp_path / "cjk" / "sentences.json").read_text(encoding="utf-8")
    )["data"]
    # At least two sentences after restoration.
    assert len(sentences) >= 2


def test_strategy_mode_none_disables_restoration(tmp_path):
    text = "今天天氣特別好我們一起去公園散步並且野餐吧"
    segs = _unpunctuated_segs(text)
    cfg = BoundaryConfig(mode="none")
    strat = CjkPipelineStrategy(boundary_config=cfg)
    strat.run(segs, _ctx(tmp_path))

    boundaries = json.loads(
        (tmp_path / "cjk" / "boundaries.json").read_text(encoding="utf-8")
    )["data"]
    assert boundaries["diagnostics"]["applied"] is False
    assert boundaries["diagnostics"]["mode"] == "none"
    assert boundaries["text"] == text  # unchanged
    final = json.loads(
        (tmp_path / "cjk" / "final_cues.json").read_text(encoding="utf-8")
    )
    assert final["meta"]["boundary_restoration"]["applied"] is False


def test_strategy_caches_boundaries_artifact(tmp_path):
    segs = _unpunctuated_segs("你好我是學生今天去公園走走")
    strat = CjkPipelineStrategy()
    strat.run(segs, _ctx(tmp_path))

    path = tmp_path / "cjk" / "boundaries.json"
    mtime = path.stat().st_mtime_ns
    # Same input — artifact must not be rewritten.
    strat.run(segs, _ctx(tmp_path))
    assert path.stat().st_mtime_ns == mtime


def test_strategy_force_regenerates_boundaries(tmp_path):
    cjk_dir = tmp_path / "cjk"
    cjk_dir.mkdir()
    (cjk_dir / "boundaries.json").write_text("{}", encoding="utf-8")

    segs = _unpunctuated_segs("你好今天")
    strat = CjkPipelineStrategy()
    strat.run(segs, _ctx(tmp_path, force=True))

    payload = json.loads(
        (cjk_dir / "boundaries.json").read_text(encoding="utf-8")
    )
    # Was rewritten with our format, not the empty stub.
    assert "input_hash" in payload
    assert "data" in payload


def test_strategy_chinese_benchmark_skips_boundary_restoration(tmp_path):
    segs = _segs(
        ("今天", 0.0, 0.5),
        ("天氣", 0.5, 1.0),
        ("很好。", 1.0, 1.5),
    )
    strat = CjkPipelineStrategy()
    ctx = _ctx(tmp_path)
    ctx.chinese_benchmark = True
    strat.run(segs, ctx)

    final = json.loads(
        (tmp_path / "cjk" / "final_cues.json").read_text(encoding="utf-8")
    )
    # Benchmark mode short-circuits everything below stage 1, including
    # boundary restoration.
    assert final["meta"]["mode"] == "chinese_benchmark"
    assert "boundary_restoration" not in final["meta"]
    assert not (tmp_path / "cjk" / "boundaries.json").exists()


def test_english_strategy_does_not_invoke_boundary_restorer(tmp_path, monkeypatch):
    """The English path must not call into the CJK boundary restorer."""
    from subforge.nlp import cjk_boundaries as cjk_b
    from subforge.pipeline.strategies import EnglishPipelineStrategy

    def boom(*args, **kwargs):
        raise AssertionError("English path must not invoke CJK boundary restorer")

    monkeypatch.setattr(cjk_b, "restore_cjk_boundaries", boom)
    segs = _segs(
        ("Hello", 0.0, 0.3),
        ("world", 0.3, 0.7),
        (".", 0.7, 0.8),
    )
    try:
        EnglishPipelineStrategy().run(segs, _ctx(tmp_path, profile=ENGLISH))
    except (OSError, ImportError, ModuleNotFoundError):
        # spaCy / model missing — what matters is the sentinel was not
        # triggered.
        pass


def test_build_gap_after_handles_split_backend_alignment():
    # Strategy helper: when timing.text differs from corrected.text, the
    # gap_after list should still be produced for matching characters.
    timing = CjkTimingAnchors(
        anchors=[
            CjkTimingAnchor(0.0, 0.2, "word_segments"),
            CjkTimingAnchor(0.2, 0.4, "word_segments"),
            # Long silence between idx 1 and idx 2.
            CjkTimingAnchor(2.0, 2.2, "word_segments"),
        ],
        source="word_segments",
        status="word_timing",
        text="你好嗎",  # timing-side text
    )
    gaps = CjkPipelineStrategy._build_gap_after("你好嗎", timing)
    assert gaps is not None
    # Gap between char 1 ("好") and char 2 ("嗎") = 2.0 - 0.4 = 1.6
    assert gaps[1] == pytest.approx(1.6, abs=1e-6)


def test_build_gap_after_returns_none_for_empty_timing():
    timing = CjkTimingAnchors(
        anchors=[],
        source="missing",
        status="missing",
        text="",
    )
    assert CjkPipelineStrategy._build_gap_after("你好", timing) is None
