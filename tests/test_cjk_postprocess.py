"""Tests for the CJK subtitle postprocess (Pre-Plan 3)."""

from __future__ import annotations

import json
from pathlib import Path

from subforge.nlp.cjk_postprocess import (
    POSTPROCESS_ACTIONS,
    PostprocessConfig,
    PostprocessedCue,
    display_width,
    postprocess_cjk_cues,
    postprocess_cues_to_writer_chunks,
)
from subforge.nlp.lang_profile import CHINESE, JAPANESE, KOREAN
from subforge.pipeline.strategies import CjkPipelineStrategy, StrategyContext
from subforge.pipeline.strategies.cjk_models import CjkAlignedCue


def _cue(
    text: str,
    start: float,
    end: float,
    *,
    fallback_reason: str | None = None,
    text_source: str = "corrected",
) -> CjkAlignedCue:
    return CjkAlignedCue(
        raw_text=text,
        corrected_text=text,
        display_text=text,
        start=start,
        end=end,
        confidence=1.0,
        fallback_reason=fallback_reason,
        text_source=text_source,
        timing_source="word_segments",
        timing_status="word_timing",
    )


def _segs(*words_and_times):
    return [{"word": w, "start": s, "end": e} for w, s, e in words_and_times]


def _ctx(tmp_path: Path, profile=CHINESE):
    return StrategyContext(
        profile=profile,
        project_dir=tmp_path,
        force=False,
        emit=lambda step, detail="": None,
        check_cancel=lambda: None,
    )


# ---------------------------------------------------------------------------
# Display-width helper
# ---------------------------------------------------------------------------


def test_display_width_counts_cjk_as_two():
    assert display_width("a") == 1
    assert display_width("漢") == 2
    assert display_width("a漢b") == 4
    # Combining marks contribute zero.
    assert display_width("é") == 1


def test_display_width_handles_punct_and_spaces():
    assert display_width("。") == 2  # full-width punct
    assert display_width(",") == 1


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_long_cue_splits_at_sentence_punctuation():
    cue = _cue(
        "今天天氣很好。我們一起去公園散步吧。",
        start=0.0,
        end=4.0,
    )
    cfg = PostprocessConfig(max_display_width=20)
    out, diag = postprocess_cjk_cues([cue], CHINESE, cfg)

    # The sentence boundary at "。" should be selected as the split point.
    assert len(out) == 2
    assert out[0].text == "今天天氣很好。"
    assert out[1].text == "我們一起去公園散步吧。"
    for cue_out in out:
        assert "split" in cue_out.actions
    assert diag["action_counts"].get("split", 0) >= 2


def test_long_cue_splits_at_secondary_punctuation_when_no_sentence_end():
    text = "今天非常累，但我還是要繼續努力工作"
    cue = _cue(text, 0.0, 4.0)
    cfg = PostprocessConfig(max_display_width=20)
    out, _ = postprocess_cjk_cues([cue], CHINESE, cfg)

    assert len(out) >= 2
    # First cue ends with a comma — picked before any width fallback.
    assert out[0].text.endswith("，")
    assert all("split" in c.actions for c in out)


def test_long_cue_falls_back_to_width_when_no_punctuation():
    text = "あいうえおかきくけこさしすせそたちつてと"  # 20 hiragana, no punct
    cue = _cue(text, 0.0, 4.0)
    cfg = PostprocessConfig(max_display_width=16)
    out, diag = postprocess_cjk_cues([cue], JAPANESE, cfg)

    assert len(out) >= 2
    for c in out:
        assert display_width(c.text) <= cfg.max_display_width
    # At least one cue should be tagged as a fallback split.
    assert any("fallback" in c.actions for c in out)
    assert diag["action_counts"].get("fallback", 0) >= 1


def test_split_preserves_total_text_and_monotonic_timing():
    cue = _cue("今天天氣很好。我們一起出門吧。", 0.0, 3.0)
    cfg = PostprocessConfig(max_display_width=18)
    out, _ = postprocess_cjk_cues([cue], CHINESE, cfg)

    joined = "".join(c.text for c in out)
    assert joined == cue.display_text
    # Monotonic & non-overlapping.
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start
    # Cover the source interval boundaries.
    assert out[0].start == cue.start
    assert out[-1].end <= cue.end + 1e-6


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def test_short_cues_merge_when_within_budget():
    cues = [
        _cue("你好", 0.0, 0.3),
        _cue("我是", 0.4, 0.7),
    ]
    cfg = PostprocessConfig(short_cue_width=6, merge_max_gap=0.5)
    out, diag = postprocess_cjk_cues(cues, CHINESE, cfg)

    assert len(out) == 1
    merged = out[0]
    assert merged.text == "你好我是"
    assert merged.start == 0.0
    assert merged.end >= 0.7
    assert "merged" in merged.actions
    assert diag["action_counts"].get("merged", 0) == 1


def test_short_cues_do_not_merge_across_sentence_boundary():
    cues = [
        _cue("你好。", 0.0, 0.4),
        _cue("再見", 0.5, 0.8),
    ]
    cfg = PostprocessConfig(short_cue_width=6, merge_max_gap=1.0)
    out, _ = postprocess_cjk_cues(cues, CHINESE, cfg)
    assert len(out) == 2
    assert out[0].text == "你好。"
    assert out[1].text == "再見"


def test_short_cues_do_not_merge_when_gap_exceeds_threshold():
    cues = [
        _cue("你好", 0.0, 0.3),
        _cue("世界", 5.0, 5.4),  # huge gap
    ]
    cfg = PostprocessConfig(short_cue_width=6, merge_max_gap=0.5)
    out, _ = postprocess_cjk_cues(cues, CHINESE, cfg)
    assert len(out) == 2


def test_merge_respects_combined_width_budget():
    # Each cue is 6 chars (width 12), combined width 24 — below default
    # merge_max_width (24). Push merge_max_width down to forbid merge.
    cues = [
        _cue("一二三四五六", 0.0, 1.0),
        _cue("七八九十甲乙", 1.1, 2.0),
    ]
    cfg = PostprocessConfig(
        short_cue_width=20,
        merge_max_width=20,
        merge_max_gap=0.5,
    )
    out, _ = postprocess_cjk_cues(cues, CHINESE, cfg)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Duration constraints
# ---------------------------------------------------------------------------


def test_short_cue_is_extended_to_min_duration():
    cues = [
        _cue("你好", 0.0, 0.3),
        _cue("再見", 5.0, 5.4),  # far enough that extension is safe
    ]
    cfg = PostprocessConfig(min_duration=0.8, merge_max_gap=0.0)
    out, _ = postprocess_cjk_cues(cues, CHINESE, cfg)
    assert out[0].end - out[0].start >= 0.8
    assert "expanded" in out[0].actions


def test_short_cue_extension_does_not_overlap_next():
    cues = [
        _cue("你好", 0.0, 0.3),
        _cue("再見", 0.4, 1.5),
    ]
    cfg = PostprocessConfig(min_duration=2.0, merge_max_gap=0.0, short_cue_width=0)
    out, _ = postprocess_cjk_cues(cues, CHINESE, cfg)
    # Even though min_duration is 2s, cue 0 should not eat into cue 1.
    assert out[0].end <= out[1].start
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start


def test_long_cue_is_capped_to_max_duration():
    cue = _cue("好", 0.0, 30.0)
    cfg = PostprocessConfig(max_duration=4.0, max_display_width=4)
    out, _ = postprocess_cjk_cues([cue], CHINESE, cfg)
    assert len(out) == 1
    assert out[0].end - out[0].start <= 4.0
    assert "shortened" in out[0].actions


# ---------------------------------------------------------------------------
# Timing safety
# ---------------------------------------------------------------------------


def test_overlapping_cues_get_clipped_to_be_monotonic():
    cues = [
        _cue("你好", 0.0, 1.0),
        _cue("再見", 0.5, 2.0),  # starts before previous ends
    ]
    cfg = PostprocessConfig(merge_max_gap=0.0, short_cue_width=0)
    out, _ = postprocess_cjk_cues(cues, CHINESE, cfg)
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start
    assert any("clipped" in c.actions for c in out)


def test_invalid_timing_marked_fallback_and_kept_safe():
    cue = _cue(
        "你好",
        start=1.0,
        end=0.5,  # end < start
        fallback_reason="no_timing_anchor",
    )
    cfg = PostprocessConfig(min_duration=0.5)
    out, _ = postprocess_cjk_cues([cue], CHINESE, cfg)
    assert len(out) == 1
    assert out[0].end >= out[0].start
    assert "fallback" in out[0].actions


def test_empty_cue_is_dropped_with_diagnostic():
    cue = _cue("", 0.0, 1.0)
    out, diag = postprocess_cjk_cues([cue], CHINESE)
    assert out == []
    assert diag["dropped"] == [{"source_index": 0, "reason": "empty_text"}]
    assert diag["total_in"] == 1
    assert diag["total_out"] == 0


# ---------------------------------------------------------------------------
# Action vocabulary / diagnostics
# ---------------------------------------------------------------------------


def test_preserved_action_when_cue_passes_through():
    cue = _cue("你好。", 0.0, 1.5)
    out, diag = postprocess_cjk_cues([cue], CHINESE, PostprocessConfig())
    assert len(out) == 1
    assert out[0].text == "你好。"
    assert "preserved" in out[0].actions
    assert diag["action_counts"]["preserved"] == 1


def test_diagnostics_records_per_cue_actions_and_widths():
    cue = _cue("今天天氣很好。我們走吧。", 0.0, 3.0)
    cfg = PostprocessConfig(max_display_width=14)
    out, diag = postprocess_cjk_cues([cue], CHINESE, cfg)

    assert "cues" in diag
    assert len(diag["cues"]) == len(out)
    for entry, cue_out in zip(diag["cues"], out):
        assert entry["text"] == cue_out.text
        assert entry["start"] == cue_out.start
        assert entry["end"] == cue_out.end
        assert entry["display_width"] == display_width(cue_out.text)
        assert entry["actions"] == cue_out.actions
        assert entry["source_index"] == 0
    # Vocabulary check: every recorded action is from POSTPROCESS_ACTIONS.
    for action in diag["action_counts"]:
        assert action in POSTPROCESS_ACTIONS


def test_diagnostics_includes_config_snapshot():
    cfg = PostprocessConfig(max_display_width=12)
    _, diag = postprocess_cjk_cues([_cue("你好", 0.0, 0.5)], CHINESE, cfg)
    assert diag["config"]["max_display_width"] == 12


# ---------------------------------------------------------------------------
# Writer chunks bridge
# ---------------------------------------------------------------------------


def test_postprocess_chunks_use_display_text_and_timing():
    cue = PostprocessedCue(
        text="你好",
        start=0.0,
        end=0.6,
        actions=["preserved"],
    )
    chunks = postprocess_cues_to_writer_chunks([cue], CHINESE)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert [t["text"] for t in chunk] == ["你", "好"]
    assert chunk[0]["start"] == 0.0
    assert chunk[-1]["end"] == 0.6
    for tok in chunk:
        assert "is_punct" in tok and "whitespace" in tok


# ---------------------------------------------------------------------------
# Integration with the CJK strategy
# ---------------------------------------------------------------------------


def test_cjk_strategy_records_postprocess_diagnostics(tmp_path):
    segs = _segs(
        ("今", 0.0, 0.2),
        ("天", 0.2, 0.4),
        ("天", 0.4, 0.6),
        ("氣", 0.6, 0.8),
        ("很", 0.8, 1.0),
        ("好", 1.0, 1.2),
        ("。", 1.2, 1.3),
        ("我", 1.5, 1.7),
        ("們", 1.7, 1.9),
        ("走", 1.9, 2.1),
        ("吧", 2.1, 2.3),
        ("。", 2.3, 2.4),
    )
    strat = CjkPipelineStrategy()
    strat.run(segs, _ctx(tmp_path))

    final = json.loads((tmp_path / "cjk" / "final_cues.json").read_text("utf-8"))
    assert "postprocess" in final["meta"]
    pp = final["meta"]["postprocess"]
    assert "config" in pp and "action_counts" in pp and "cues" in pp
    # Every cue in the writer chunks shows up in the postprocess diagnostics.
    assert pp["total_out"] == len(final["chunks"])
    # All actions belong to the documented vocabulary.
    for action in pp["action_counts"]:
        assert action in POSTPROCESS_ACTIONS


def test_cjk_strategy_long_paragraph_splits_into_readable_cues(tmp_path):
    # A single long span without punctuation between chars: postprocess
    # should split it down to multiple cues that respect the width budget.
    chars = "今天天氣特別好我們一起去公園散步並且野餐吧"
    segs = [
        {"word": ch, "start": i * 0.2, "end": (i + 1) * 0.2}
        for i, ch in enumerate(chars)
    ]
    # Add a sentence end so the alignment stage has a sentence boundary.
    segs.append({"word": "。", "start": len(chars) * 0.2, "end": len(chars) * 0.2 + 0.1})

    strat = CjkPipelineStrategy()
    strat.run(segs, _ctx(tmp_path))

    final = json.loads((tmp_path / "cjk" / "final_cues.json").read_text("utf-8"))
    pp = final["meta"]["postprocess"]
    max_width = pp["config"]["max_display_width"]
    for cue in pp["cues"]:
        assert cue["display_width"] <= max_width


def test_cjk_strategy_korean_uses_postprocess(tmp_path):
    # Korean profile uses spaces but is still routed through CJK strategy.
    segs = _segs(
        ("안녕", 0.0, 0.4),
        ("하세요", 0.4, 0.9),
        (".", 0.9, 1.0),
    )
    strat = CjkPipelineStrategy()
    ctx = _ctx(tmp_path, profile=KOREAN)
    chunks = strat.run(segs, ctx)
    assert chunks
    final = json.loads((tmp_path / "cjk" / "final_cues.json").read_text("utf-8"))
    assert "postprocess" in final["meta"]


def test_cjk_strategy_final_cues_have_monotonic_timing(tmp_path):
    segs = _segs(
        ("你", 0.0, 0.2),
        ("好", 0.2, 0.4),
        ("！", 0.4, 0.5),
        ("我", 0.6, 0.8),
        ("是", 0.8, 1.0),
        ("學", 1.0, 1.2),
        ("生", 1.2, 1.4),
        ("。", 1.4, 1.5),
    )
    strat = CjkPipelineStrategy()
    chunks = strat.run(segs, _ctx(tmp_path))

    bounds = []
    for chunk in chunks:
        bounds.append((chunk[0]["start"], chunk[-1]["end"]))
    for (s, e) in bounds:
        assert s <= e
    for (s1, e1), (s2, _e2) in zip(bounds, bounds[1:]):
        assert e1 <= s2 + 1e-6


# ---------------------------------------------------------------------------
# Regression safety: English path is untouched
# ---------------------------------------------------------------------------


def test_english_strategy_does_not_use_cjk_postprocess(tmp_path, monkeypatch):
    """The English strategy must not call into the CJK postprocess.

    Sentinel: monkeypatch ``postprocess_cjk_cues`` to fail. The English
    pipeline should still return a result.
    """
    from subforge.nlp import cjk_postprocess as cjk_pp
    from subforge.nlp.lang_profile import ENGLISH
    from subforge.pipeline.strategies import EnglishPipelineStrategy

    def boom(*args, **kwargs):
        raise AssertionError("English strategy must not invoke CJK postprocess")

    monkeypatch.setattr(cjk_pp, "postprocess_cjk_cues", boom)

    segs = _segs(
        ("Hello", 0.0, 0.3),
        ("world", 0.3, 0.7),
        (".", 0.7, 0.8),
    )
    strat = EnglishPipelineStrategy()
    # English path requires spaCy; if unavailable in this environment we
    # still want to confirm the import path doesn't hit cjk_postprocess.
    try:
        strat.run(segs, _ctx(tmp_path, profile=ENGLISH))
    except OSError:
        # spaCy model missing — that's fine; what matters is that the
        # CJK postprocess sentinel was not triggered.
        pass


def test_chinese_benchmark_mode_bypasses_postprocess(tmp_path):
    segs = _segs(
        ("今天", 0.0, 0.5),
        ("天氣", 0.5, 1.0),
        ("很好。", 1.0, 1.5),
    )
    strat = CjkPipelineStrategy()
    ctx = _ctx(tmp_path)
    ctx.chinese_benchmark = True
    strat.run(segs, ctx)

    final = json.loads((tmp_path / "cjk" / "final_cues.json").read_text("utf-8"))
    # Benchmark mode short-circuits before postprocess; the postprocess
    # diagnostics block should not be present in benchmark output.
    assert final["meta"]["mode"] == "chinese_benchmark"
    assert "postprocess" not in final["meta"]
