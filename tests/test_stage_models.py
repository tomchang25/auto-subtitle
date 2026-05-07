"""Tests for the language-agnostic stage models.

These exercise the shared types directly and lock in two compatibility
guarantees needed by PR1:

1. The legacy CJK names continue to import from
   :mod:`subforge.pipeline.strategies.cjk_models` and resolve to the same
   class objects as the shared names in :mod:`subforge.pipeline.stages`.
2. The shared word-segment adapters reproduce the existing CJK adapter
   behavior on the same fixtures, so the rename is purely structural.
"""

from __future__ import annotations

from subforge.pipeline.stages import (
    AlignedCue,
    PipelineResult,
    Sentence,
    TimingAnchor,
    TimingAnchors,
    TokenInterval,
    Transcript,
    build_split_inputs,
    word_segments_to_inputs,
)


def _segs(*words_and_times):
    return [{"word": w, "start": s, "end": e} for w, s, e in words_and_times]


def _cue(**overrides) -> AlignedCue:
    base = dict(
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
    )
    base.update(overrides)
    return AlignedCue(**base)


# ---------------------------------------------------------------------------
# TokenInterval
# ---------------------------------------------------------------------------


def test_token_interval_roundtrip_defaults():
    tok = TokenInterval(text="hi", start=0.0, end=0.2)
    d = tok.to_dict()
    assert d == {
        "text": "hi",
        "start": 0.0,
        "end": 0.2,
        "is_punct": False,
        "whitespace": "",
        "source": "",
    }
    assert TokenInterval.from_dict(d) == tok


def test_token_interval_roundtrip_full():
    tok = TokenInterval(
        text=".",
        start=1.0,
        end=1.05,
        is_punct=True,
        whitespace=" ",
        source="asr_word",
    )
    restored = TokenInterval.from_dict(tok.to_dict())
    assert restored == tok


# ---------------------------------------------------------------------------
# AlignedCue serialization
# ---------------------------------------------------------------------------


def test_aligned_cue_roundtrip_without_tokens_omits_key():
    cue = _cue()
    d = cue.to_dict()
    # Existing CJK JSON shape is preserved when no tokens are populated.
    assert "tokens" not in d
    restored = AlignedCue.from_dict(d)
    assert restored == cue
    assert restored.tokens is None


def test_aligned_cue_from_dict_accepts_legacy_payload_without_tokens_key():
    legacy_payload = {
        "raw_text": "abc",
        "corrected_text": "abc",
        "display_text": "abc",
        "start": 0.0,
        "end": 0.6,
        "confidence": 1.0,
        "fallback_reason": None,
        "text_source": "corrected",
        "timing_source": "word_segments",
        "timing_status": "word_timing",
    }
    restored = AlignedCue.from_dict(legacy_payload)
    assert restored.tokens is None
    assert restored.to_dict() == legacy_payload


def test_aligned_cue_roundtrip_with_tokens():
    cue = _cue(
        tokens=[
            TokenInterval(text="ab", start=0.0, end=0.4, source="asr_word"),
            TokenInterval(
                text=".",
                start=0.4,
                end=0.6,
                is_punct=True,
                source="char_split",
            ),
        ],
    )
    d = cue.to_dict()
    assert "tokens" in d
    assert d["tokens"][0]["source"] == "asr_word"
    assert d["tokens"][1]["is_punct"] is True
    restored = AlignedCue.from_dict(d)
    assert restored == cue
    assert isinstance(restored.tokens[0], TokenInterval)


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------


def test_cjk_compat_aliases_resolve_to_shared_classes():
    from subforge.pipeline.strategies import cjk_models as compat

    assert compat.CjkTranscript is Transcript
    assert compat.CjkTimingAnchor is TimingAnchor
    assert compat.CjkTimingAnchors is TimingAnchors
    assert compat.CjkSentence is Sentence
    assert compat.CjkAlignedCue is AlignedCue
    assert compat.CjkPipelineResult is PipelineResult
    assert compat.TokenInterval is TokenInterval
    assert compat.word_segments_to_cjk_inputs is word_segments_to_inputs
    assert compat.build_split_cjk_inputs is build_split_inputs


def test_cjk_compat_aligned_cue_isinstance():
    from subforge.pipeline.strategies.cjk_models import CjkAlignedCue

    cue = _cue()
    assert isinstance(cue, CjkAlignedCue)


# ---------------------------------------------------------------------------
# Adapters — exercise the same fixtures as test_cjk_pipeline.py to prove
# the rename is purely structural.
# ---------------------------------------------------------------------------


def test_word_segments_to_inputs_separates_text_and_timing():
    segs = _segs(("你", 0.0, 0.2), ("好", 0.2, 0.4))
    transcript, timing = word_segments_to_inputs(segs, join_token="")

    assert isinstance(transcript, Transcript)
    assert transcript.text == "你好"
    assert transcript.source == "asr_raw"

    assert isinstance(timing, TimingAnchors)
    assert timing.source == "word_segments"
    assert timing.status == "word_timing"
    assert timing.text == "你好"
    assert len(timing.anchors) == 2
    assert timing.anchors[0].source == "word_segments"
    assert timing.char_to_word == [0, 1]


def test_word_segments_to_inputs_handles_join_token():
    segs = _segs(("hi", 0.0, 0.2), ("there", 0.3, 0.6))
    transcript, timing = word_segments_to_inputs(segs, join_token=" ")

    assert transcript.text == "hi there"
    assert len(timing.anchors) == len(transcript.text)
    join_anchor = timing.anchors[2]
    assert join_anchor.source == "join_token"
    assert join_anchor.start == join_anchor.end


def test_word_segments_to_inputs_empty_input():
    transcript, timing = word_segments_to_inputs([], join_token="")
    assert transcript.text == ""
    assert timing.anchors == []
    assert timing.status == "missing"


def test_build_split_inputs_separates_sources():
    segs = _segs(("今", 0.0, 0.2), ("天", 0.2, 0.4), ("。", 0.4, 0.5))
    transcript, timing = build_split_inputs(
        segs,
        transcript_text="今天。",
        transcript_source="sensevoice",
        join_token="",
    )
    assert transcript.text == "今天。"
    assert transcript.source == "sensevoice"
    assert timing.text == "今天。"
    assert timing.source == "word_segments"
    assert len(timing.anchors) == len(timing.text)


# ---------------------------------------------------------------------------
# PipelineResult round-trip — ensures cues serialize through the container
# without surprises (e.g. tokens key leaking when unset).
# ---------------------------------------------------------------------------


def test_pipeline_result_roundtrip_omits_token_key_for_unset_cues():
    result = PipelineResult(
        cues=[_cue()],
        text_source="corrected",
        timing_source="word_segments",
        timing_status="word_timing",
        fallback_used=False,
        fallback_reason=None,
    )
    d = result.to_dict()
    assert "tokens" not in d["cues"][0]
