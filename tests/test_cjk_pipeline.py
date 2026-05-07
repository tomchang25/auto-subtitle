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
        "corrected_transcript.json",
        "sentences.json",
        "alignment.json",
    ):
        assert (cjk_dir / name).exists(), f"Missing artifact: {name}"

    raw = json.loads((cjk_dir / "raw_transcript.json").read_text(encoding="utf-8"))
    assert raw["data"]["text"] == "今天天氣很好。我們走吧。"

    sentences = json.loads((cjk_dir / "sentences.json").read_text(encoding="utf-8"))
    assert sentences["data"] == ["今天天氣很好。", "我們走吧。"]


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
    raw_mtime = raw_path.stat().st_mtime_ns

    # Re-run with same input — artifact should not be rewritten.
    strat.run(segs, ctx)
    assert raw_path.stat().st_mtime_ns == raw_mtime


def test_cjk_pipeline_force_clears_artifacts(tmp_path):
    cjk_dir = tmp_path / "cjk"
    cjk_dir.mkdir()
    (cjk_dir / "raw_transcript.json").write_text("{}", encoding="utf-8")
    (cjk_dir / "alignment.json").write_text("{}", encoding="utf-8")

    segs = _segs(("今", 0.0, 0.1), ("。", 0.1, 0.2))
    strat = CjkPipelineStrategy()
    strat.run(segs, _ctx(tmp_path, force=True))

    raw = json.loads((cjk_dir / "raw_transcript.json").read_text(encoding="utf-8"))
    assert "input_hash" in raw  # Was rewritten with our format, not the empty stub.


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


class _DropCharCorrector:
    """Corrector that deletes every '今' to test alignment under edits."""

    def correct(self, text, lang):
        return text.replace("今", "")


def test_alignment_survives_corrector_edits(tmp_path):
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
    # All original word_segments should still appear in the output despite
    # the corrector dropping a char from the corrected text.
    joined = "".join(tok["text"] for chunk in chunks for tok in chunk)
    assert joined == "今天天氣好。"


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

    # Benchmark mode short-circuits the transcript-first stages — no artifacts.
    assert not (tmp_path / "cjk" / "raw_transcript.json").exists()


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
