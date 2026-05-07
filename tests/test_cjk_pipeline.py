"""Tests for the transcript-first CJK subtitle pipeline strategy."""

from __future__ import annotations

import json
from pathlib import Path

from subforge.nlp.cjk_corrector import NoOpCorrector
from subforge.nlp.lang_profile import CHINESE, ENGLISH, JAPANESE, KOREAN
from subforge.pipeline.strategies import (
    CjkPipelineStrategy,
    EnglishPipelineStrategy,
    StrategyContext,
    get_strategy,
)
from subforge.pipeline.strategies.cjk import _map_corrected_to_raw
from subforge.pipeline.strategies.cjk_models import (
    CjkAlignedCue,
    CjkSentence,
    CjkTimingAnchors,
    CjkTranscript,
    cjk_cues_to_writer_chunks,
    word_segments_to_cjk_inputs,
)


def _segs(*words_and_times):
    return [{"word": w, "start": s, "end": e} for w, s, e in words_and_times]


def _ctx(tmp_path: Path, profile=CHINESE, force=False, chinese_benchmark=False):
    return StrategyContext(
        profile=profile,
        project_dir=tmp_path,
        force=force,
        emit=lambda step, detail="": None,
        check_cancel=lambda: None,
        chinese_benchmark=chinese_benchmark,
    )


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def test_get_strategy_dispatches_by_profile():
    assert isinstance(get_strategy(ENGLISH), EnglishPipelineStrategy)
    assert isinstance(get_strategy(CHINESE), CjkPipelineStrategy)
    assert isinstance(get_strategy(JAPANESE), CjkPipelineStrategy)
    assert isinstance(get_strategy(KOREAN), CjkPipelineStrategy)


# ---------------------------------------------------------------------------
# Stage artifacts
# ---------------------------------------------------------------------------


def test_cjk_pipeline_writes_stage_artifacts(tmp_path):
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
    ctx = _ctx(tmp_path)
    strat.run(segs, ctx)

    cjk_dir = tmp_path / "cjk"
    for name in (
        "raw_transcript.json",
        "timing_anchors.json",
        "corrected_transcript.json",
        "sentences.json",
        "alignment.json",
        "final_cues.json",
    ):
        assert (cjk_dir / name).exists(), f"Missing artifact: {name}"

    raw = json.loads((cjk_dir / "raw_transcript.json").read_text(encoding="utf-8"))
    assert raw["data"]["text"] == "今天天氣很好。我們走吧。"
    assert raw["data"]["source"] == "asr_raw"

    timing = json.loads((cjk_dir / "timing_anchors.json").read_text(encoding="utf-8"))
    assert timing["data"]["source"] == "word_segments"
    assert timing["data"]["status"] == "word_timing"
    assert len(timing["data"]["anchors"]) == len(raw["data"]["text"])

    sentences = json.loads((cjk_dir / "sentences.json").read_text(encoding="utf-8"))
    sent_data = sentences["data"]
    assert [s["text"] for s in sent_data] == ["今天天氣很好。", "我們走吧。"]
    # Offsets should index back into the corrected transcript.
    assert sent_data[0]["char_start"] == 0
    assert sent_data[0]["char_end"] == 7
    assert sent_data[1]["char_start"] == 7

    alignment = json.loads((cjk_dir / "alignment.json").read_text(encoding="utf-8"))
    cues = alignment["data"]
    assert len(cues) == 2
    for cue in cues:
        assert cue["text_source"] in {"corrected", "raw"}
        assert cue["timing_source"] == "word_segments"
        assert cue["timing_status"] in {"word_timing", "missing"}
        assert cue["fallback_reason"] is None
        assert cue["display_text"] == cue["corrected_text"]

    final = json.loads((cjk_dir / "final_cues.json").read_text(encoding="utf-8"))
    assert final["meta"]["text_source"] in {"corrected", "raw", "mixed"}
    assert final["meta"]["timing_source"] == "word_segments"
    assert final["meta"]["timing_status"] == "word_timing"
    assert final["meta"]["fallback_used"] is False
    assert final["meta"]["mode"] == "transcript_first"


def test_cjk_pipeline_reuses_cached_stages(tmp_path):
    segs = _segs(
        ("今", 0.0, 0.1),
        ("天", 0.1, 0.2),
        ("。", 0.2, 0.3),
    )
    strat = CjkPipelineStrategy()
    ctx = _ctx(tmp_path)
    strat.run(segs, ctx)

    raw_path = tmp_path / "cjk" / "raw_transcript.json"
    timing_path = tmp_path / "cjk" / "timing_anchors.json"
    raw_mtime = raw_path.stat().st_mtime_ns
    timing_mtime = timing_path.stat().st_mtime_ns

    # Re-run with same input — artifacts should not be rewritten.
    strat.run(segs, ctx)
    assert raw_path.stat().st_mtime_ns == raw_mtime
    assert timing_path.stat().st_mtime_ns == timing_mtime


def test_cjk_pipeline_force_clears_artifacts(tmp_path):
    cjk_dir = tmp_path / "cjk"
    cjk_dir.mkdir()
    (cjk_dir / "raw_transcript.json").write_text("{}", encoding="utf-8")
    (cjk_dir / "timing_anchors.json").write_text("{}", encoding="utf-8")
    (cjk_dir / "alignment.json").write_text("{}", encoding="utf-8")
    (cjk_dir / "final_cues.json").write_text("{}", encoding="utf-8")

    segs = _segs(("今", 0.0, 0.1), ("。", 0.1, 0.2))
    strat = CjkPipelineStrategy()
    strat.run(segs, _ctx(tmp_path, force=True))

    raw = json.loads((cjk_dir / "raw_transcript.json").read_text(encoding="utf-8"))
    # Was rewritten with our format, not the empty stub.
    assert "input_hash" in raw
    timing = json.loads((cjk_dir / "timing_anchors.json").read_text(encoding="utf-8"))
    assert "input_hash" in timing


# ---------------------------------------------------------------------------
# Output equivalence with no-op corrector
# ---------------------------------------------------------------------------


def test_cjk_pipeline_with_noop_corrector_produces_chunks(tmp_path):
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
    strat = CjkPipelineStrategy(corrector=NoOpCorrector())
    chunks = strat.run(segs, _ctx(tmp_path))

    assert chunks, "Expected at least one chunk"
    joined = "".join(tok["text"] for chunk in chunks for tok in chunk)
    assert "你好！" in joined
    assert "我是學生。" in joined

    for chunk in chunks:
        for tok in chunk:
            assert "start" in tok and "end" in tok
            assert tok["start"] <= tok["end"]


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class _BrokenCorrector:
    def correct(self, text, lang):
        raise RuntimeError("corrector exploded")


def test_corrector_failure_falls_back_to_raw(tmp_path):
    segs = _segs(("今", 0.0, 0.1), ("天", 0.1, 0.2), ("。", 0.2, 0.3))
    strat = CjkPipelineStrategy(corrector=_BrokenCorrector())
    chunks = strat.run(segs, _ctx(tmp_path))

    assert chunks  # Pipeline didn't blow up.
    corrected = json.loads(
        (tmp_path / "cjk" / "corrected_transcript.json").read_text(encoding="utf-8")
    )["data"]
    assert corrected["text"] == "今天。"
    assert corrected["applied"] is False
    assert corrected["source"] == "asr_raw"

    # Final cues should display the raw text since correction was rejected.
    joined = "".join(tok["text"] for chunk in chunks for tok in chunk)
    assert joined == "今天。"

    alignment = json.loads(
        (tmp_path / "cjk" / "alignment.json").read_text(encoding="utf-8")
    )["data"]
    assert all(cue["text_source"] == "raw" for cue in alignment)


class _DropCharCorrector:
    """Corrector that deletes every '今' to test alignment under edits."""

    def correct(self, text, lang):
        return text.replace("今", "")


def test_alignment_uses_corrected_text_as_display(tmp_path):
    segs = _segs(
        ("今", 0.0, 0.2),
        ("天", 0.2, 0.4),
        ("天", 0.4, 0.6),
        ("氣", 0.6, 0.8),
        ("好", 0.8, 1.0),
        ("。", 1.0, 1.1),
    )
    strat = CjkPipelineStrategy(corrector=_DropCharCorrector())
    chunks = strat.run(segs, _ctx(tmp_path))

    assert chunks
    # Display text should now follow the corrected transcript, not the raw
    # ASR output that still contains '今'.
    joined = "".join(tok["text"] for chunk in chunks for tok in chunk)
    assert joined == "天天氣好。"

    alignment = json.loads(
        (tmp_path / "cjk" / "alignment.json").read_text(encoding="utf-8")
    )["data"]
    cue = alignment[0]
    assert cue["corrected_text"] == "天天氣好。"
    assert cue["raw_text"] == "天天氣好。"
    assert cue["display_text"] == "天天氣好。"
    assert cue["text_source"] == "corrected"


def test_empty_word_segments_does_not_crash(tmp_path):
    strat = CjkPipelineStrategy()
    chunks = strat.run([], _ctx(tmp_path))
    assert chunks == []


# ---------------------------------------------------------------------------
# Benchmark mode preserved
# ---------------------------------------------------------------------------


def test_chinese_benchmark_mode_uses_hard_cut(tmp_path):
    segs = _segs(
        ("今天", 0.0, 0.5),
        ("天氣", 0.5, 1.0),
        ("很好。", 1.0, 1.5),
        ("我們", 1.5, 2.0),
        ("走吧。", 2.0, 2.5),
    )
    strat = CjkPipelineStrategy()
    chunks = strat.run(segs, _ctx(tmp_path, chinese_benchmark=True))

    assert len(chunks) == 2
    assert "".join(t["text"] for t in chunks[0]) == "今天天氣很好。"
    assert "".join(t["text"] for t in chunks[1]) == "我們走吧。"

    # Benchmark mode short-circuits the transcript-first stages — no per-stage
    # artifacts — but still records final_cues.json with bypass metadata so a
    # later benchmark report can tell apart "ran the full pipeline" from
    # "ran the hard-cut path".
    cjk_dir = tmp_path / "cjk"
    assert not (cjk_dir / "raw_transcript.json").exists()
    assert not (cjk_dir / "timing_anchors.json").exists()
    final = json.loads((cjk_dir / "final_cues.json").read_text(encoding="utf-8"))
    assert final["meta"]["mode"] == "chinese_benchmark"
    assert "correction" in final["meta"]["bypassed_stages"]


# ---------------------------------------------------------------------------
# Char-level alignment helper
# ---------------------------------------------------------------------------


def test_map_corrected_to_raw_identity():
    mapping = _map_corrected_to_raw("abc", "abc")
    assert mapping == [0, 1, 2]


def test_map_corrected_to_raw_with_deletion():
    # corrected drops "X" → b is shorter
    mapping = _map_corrected_to_raw("abc", "ac")
    assert mapping[0] == 0          # 'a' aligns
    assert mapping[2] == 1          # 'c' aligns
    # mapping[1] ('b' in corrected, missing from raw) is None
    assert mapping[1] is None


def test_map_corrected_to_raw_empty():
    assert _map_corrected_to_raw("", "abc") == []
    assert _map_corrected_to_raw("abc", "") == [None, None, None]


# ---------------------------------------------------------------------------
# CJK internal data contract
# ---------------------------------------------------------------------------


def test_word_segments_adapter_separates_text_and_timing():
    segs = _segs(("你", 0.0, 0.2), ("好", 0.2, 0.4))
    transcript, timing = word_segments_to_cjk_inputs(segs, join_token="")

    assert isinstance(transcript, CjkTranscript)
    assert transcript.text == "你好"
    assert transcript.source == "asr_raw"

    assert isinstance(timing, CjkTimingAnchors)
    assert timing.source == "word_segments"
    assert timing.status == "word_timing"
    assert len(timing.anchors) == 2
    assert timing.anchors[0].source == "word_segments"
    assert timing.char_to_word == [0, 1]


def test_word_segments_adapter_handles_join_token():
    segs = _segs(("hi", 0.0, 0.2), ("there", 0.3, 0.6))
    transcript, timing = word_segments_to_cjk_inputs(segs, join_token=" ")

    # 'hi' (2) + ' ' (1 join) + 'there' (5) = 8 chars
    assert transcript.text == "hi there"
    assert len(timing.anchors) == len(transcript.text)
    # The join-token char should be tagged distinctly.
    join_anchor = timing.anchors[2]
    assert join_anchor.source == "join_token"
    assert join_anchor.start == join_anchor.end


def test_word_segments_adapter_empty_input():
    transcript, timing = word_segments_to_cjk_inputs([], join_token="")
    assert transcript.text == ""
    assert timing.anchors == []
    assert timing.status == "missing"


def test_cjk_cues_to_writer_chunks_uses_display_text():
    cues = [
        CjkAlignedCue(
            raw_text="abc",
            corrected_text="abc",
            display_text="abc",
            start=0.0,
            end=0.6,
            confidence=1.0,
            fallback_reason=None,
            text_source="corrected",
            timing_source="word_segments",
            timing_status="word_timing",
        ),
    ]
    chunks = cjk_cues_to_writer_chunks(cues, profile=CHINESE)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert [t["text"] for t in chunk] == ["a", "b", "c"]
    assert chunk[0]["start"] == 0.0
    assert chunk[-1]["end"] == 0.6
    # Each token has a non-decreasing interval.
    for tok in chunk:
        assert tok["start"] <= tok["end"]


def test_cjk_cues_to_writer_chunks_skips_empty_display_text():
    cues = [
        CjkAlignedCue(
            raw_text="",
            corrected_text="",
            display_text="",
            start=0.0,
            end=1.0,
            confidence=0.0,
            fallback_reason="no_timing_anchor",
            text_source="raw",
            timing_source="word_segments",
            timing_status="missing",
        ),
    ]
    assert cjk_cues_to_writer_chunks(cues, profile=CHINESE) == []


def test_cjk_sentence_roundtrip():
    s = CjkSentence(text="你好。", char_start=0, char_end=3)
    restored = CjkSentence.from_dict(s.to_dict())
    assert restored == s
