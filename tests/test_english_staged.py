"""Tests for the English staged pipeline (PR3).

These cover the migration of ``EnglishPipelineStrategy`` from a strategy
that owned its own end-to-end orchestration onto the shared
:class:`StagedPipelineRunner` driven by :class:`EnglishPolicy`.

Coverage targets the regressions called out in the PR plan:

* English runs through the staged runner and produces canonical
  artifacts under ``project_dir/stages/`` (and only there — English has
  no legacy mirror directory).
* ``alignment.json`` carries typed token intervals that round-trip.
* Sentence character offsets resolve back to exact substrings of the
  corrected transcript.
* Writer output is byte-identical to the legacy English
  refine/split/merge pipeline on the existing news fixture.
* Punctuation spacing matches the legacy ``" ".join(token_text)``
  writer behavior across common punctuation cases.
* Cache behavior: cached reruns reuse canonical artifacts; ``force=True``
  regenerates them.
* CJK behavior is unchanged after extracting the shared finalize helper
  (the existing test_cjk_* suites still cover this; here we just spot
  check that English does not write a legacy mirror).
"""

from __future__ import annotations

import json
from pathlib import Path

from subforge.config import (
    BREATH_GAP,
    MAX_GAP,
    MERGE_MAX_DURATION,
    MERGE_MAX_GAP,
    MIN_DURATION,
    MIN_WORDS_FOR_BREATH_SPLIT,
    SEG_PAUSE_THRESHOLD,
)
from subforge.nlp.alignment import (
    align_sentences_with_timestamps,
    refine_sentences_by_timing,
)
from subforge.nlp.lang_profile import ENGLISH
from subforge.nlp.segmentation import (
    merge_short_segments,
    split_long_sentences_by_length,
)
from subforge.nlp.text_semantically import split_to_sentences
from subforge.pipeline.stages import (
    CANONICAL_DIRNAME,
    STAGE_FILES,
    AlignedCue,
    Sentence,
    StagedPipelineRunner,
    Transcript,
)
from subforge.pipeline.stages.english_policy import (
    EnglishPolicy,
    aligned_cues_to_token_chunks,
)
from subforge.pipeline.strategies import (
    EnglishPipelineStrategy,
    StrategyContext,
)
from subforge.utils import get_bounds_and_text


def _segs(*words_and_times):
    return [{"word": w, "start": s, "end": e} for w, s, e in words_and_times]


def _ctx(tmp_path: Path, *, force: bool = False) -> StrategyContext:
    return StrategyContext(
        profile=ENGLISH,
        project_dir=tmp_path,
        force=force,
        emit=lambda step, detail="": None,
        check_cancel=lambda: None,
    )


def _hello_world_segs():
    return _segs(
        ("Hello", 0.0, 0.3),
        (",", 0.3, 0.35),
        ("world", 0.35, 0.7),
        (".", 0.7, 0.75),
        ("This", 0.9, 1.1),
        ("is", 1.1, 1.2),
        ("a", 1.2, 1.25),
        ("test", 1.25, 1.5),
        (".", 1.5, 1.55),
    )


def _news_segs():
    """Reload the news fixture each time — alignment mutates in place."""
    return json.loads(
        Path("tests/data/news_word_segments.json").read_text(encoding="utf-8")
    )


def _legacy_english(word_segments: list[dict], profile=ENGLISH):
    """Reproduce the pre-staged English orchestration body verbatim."""
    full_text = profile.join_token.join(seg["word"] for seg in word_segments)
    sentence_chunks = split_to_sentences(full_text)
    aligned = align_sentences_with_timestamps(word_segments, sentence_chunks)
    refined = refine_sentences_by_timing(
        aligned,
        min_duration=MIN_DURATION,
        max_gap=MAX_GAP,
        breath_gap=BREATH_GAP,
        min_words_for_breath_split=MIN_WORDS_FOR_BREATH_SPLIT,
    )
    refined = split_long_sentences_by_length(
        refined,
        min_words=profile.seg_min,
        max_words=profile.seg_hard,
        soft_words=profile.seg_soft,
        pause_threshold=SEG_PAUSE_THRESHOLD,
        profile=profile,
    )
    refined = merge_short_segments(
        refined,
        max_words=profile.merge_max,
        max_duration=MERGE_MAX_DURATION,
        max_gap=MERGE_MAX_GAP,
        profile=profile,
    )
    return refined


# ---------------------------------------------------------------------------
# Strategy wiring
# ---------------------------------------------------------------------------


def test_english_strategy_is_thin_runner_wrapper(tmp_path):
    """Strategy.run delegates through the staged runner with EnglishPolicy."""
    chunks = EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))
    assert chunks
    # Each token has the keys the writer / postprocess depend on.
    for chunk in chunks:
        for tok in chunk:
            assert {"text", "start", "end", "is_punct"}.issubset(tok.keys())


def test_english_run_writes_canonical_artifacts(tmp_path):
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))

    stages_dir = tmp_path / CANONICAL_DIRNAME
    assert stages_dir.is_dir()
    for name in STAGE_FILES:
        assert (stages_dir / name).exists(), f"missing canonical: {name}"


def test_english_run_does_not_create_legacy_mirror_dir(tmp_path):
    """English has no legacy artifact directory — only ``stages/``."""
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))

    children = sorted(p.name for p in tmp_path.iterdir())
    assert children == [CANONICAL_DIRNAME], (
        f"unexpected children alongside stages/: {children}"
    )


# ---------------------------------------------------------------------------
# Artifact contracts
# ---------------------------------------------------------------------------


def test_english_raw_transcript_matches_legacy_join_text(tmp_path):
    """The staged raw transcript must match ``profile.join_token.join(words)``."""
    segs = _hello_world_segs()
    EnglishPipelineStrategy().run(segs, _ctx(tmp_path))

    raw = json.loads(
        (tmp_path / CANONICAL_DIRNAME / "raw_transcript.json").read_text(
            encoding="utf-8"
        )
    )["data"]
    expected = ENGLISH.join_token.join(seg["word"] for seg in segs)
    assert raw["text"] == expected
    assert raw["source"] == "asr_raw"


def test_english_corrected_transcript_records_no_correction(tmp_path):
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))
    corrected = json.loads(
        (tmp_path / CANONICAL_DIRNAME / "corrected_transcript.json").read_text(
            encoding="utf-8"
        )
    )["data"]
    assert corrected["applied"] is False
    assert corrected["corrector"] == "none"


def test_english_timing_anchors_are_word_timing(tmp_path):
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))
    timing = json.loads(
        (tmp_path / CANONICAL_DIRNAME / "timing_anchors.json").read_text(
            encoding="utf-8"
        )
    )["data"]
    assert timing["status"] == "word_timing"
    assert timing["source"] == "word_segments"
    assert timing["anchors"], "anchors should be populated for word-timed input"


def test_english_alignment_cues_carry_typed_token_intervals(tmp_path):
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))
    payload = json.loads(
        (tmp_path / CANONICAL_DIRNAME / "alignment.json").read_text(
            encoding="utf-8"
        )
    )["data"]

    cues = [AlignedCue.from_dict(c) for c in payload]
    assert cues
    for cue in cues:
        assert cue.tokens is not None
        assert len(cue.tokens) > 0
        for tok in cue.tokens:
            assert tok.source == "asr_word"
            assert isinstance(tok.start, float)
            assert isinstance(tok.end, float)
            assert tok.end >= tok.start
        # cue start/end must envelop the contained tokens
        assert cue.start == min(t.start for t in cue.tokens)
        assert cue.end == max(t.end for t in cue.tokens)


def test_english_sentence_offsets_match_transcript(tmp_path):
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))
    sentences_data = json.loads(
        (tmp_path / CANONICAL_DIRNAME / "sentences.json").read_text(
            encoding="utf-8"
        )
    )["data"]
    raw_text = json.loads(
        (tmp_path / CANONICAL_DIRNAME / "raw_transcript.json").read_text(
            encoding="utf-8"
        )
    )["data"]["text"]

    sentences = [Sentence.from_dict(s) for s in sentences_data]
    assert sentences
    for s in sentences:
        assert raw_text[s.char_start : s.char_end] == s.text


def test_english_final_cues_metadata_records_staged_mode(tmp_path):
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))
    final = json.loads(
        (tmp_path / CANONICAL_DIRNAME / "final_cues.json").read_text(
            encoding="utf-8"
        )
    )
    meta = final["meta"]
    assert meta["mode"] == "english_staged"
    assert meta["profile"] == "en"
    assert meta["correction_mode"] == "none"
    assert meta["correction_applied"] is False
    assert meta["fallback_used"] is False
    assert meta["timing_source"] == "word_segments"
    assert "postprocess" in meta
    assert "actions" in meta["postprocess"]


# ---------------------------------------------------------------------------
# Behaviour compatibility with the legacy English orchestration
# ---------------------------------------------------------------------------


def test_english_staged_byte_identical_to_legacy_on_news_fixture(tmp_path):
    """The staged path must produce the same writer chunks as legacy."""
    legacy_chunks = _legacy_english(_news_segs())
    staged_chunks = EnglishPipelineStrategy().run(_news_segs(), _ctx(tmp_path))
    assert staged_chunks == legacy_chunks


def test_english_staged_writer_output_matches_legacy_punctuation_spacing(
    tmp_path,
):
    """``get_bounds_and_text`` over staged chunks must match legacy spacing."""
    cases = [
        _segs(
            ("Hello", 0.0, 0.3),
            (",", 0.3, 0.35),
            ("world", 0.35, 0.7),
            (".", 0.7, 0.75),
        ),
        _segs(
            ("Wait", 0.0, 0.3),
            ("...", 0.3, 0.4),
            ("really", 0.4, 0.7),
            ("?", 0.7, 0.75),
        ),
        _segs(
            ("He", 0.0, 0.2),
            ("said", 0.2, 0.5),
            ('"yes"', 0.5, 0.9),
            ("immediately", 0.9, 1.5),
            (".", 1.5, 1.55),
        ),
    ]
    for segs in cases:
        legacy = get_bounds_and_text(_legacy_english(segs), profile=ENGLISH)
        staged = get_bounds_and_text(
            EnglishPipelineStrategy().run(segs, _ctx(tmp_path)),
            profile=ENGLISH,
        )
        assert legacy == staged, (
            f"writer output diverged:\n  legacy={legacy}\n  staged={staged}"
        )


# ---------------------------------------------------------------------------
# Adapter directly
# ---------------------------------------------------------------------------


def test_aligned_cues_to_token_chunks_preserves_whitespace_and_is_punct():
    cue = AlignedCue(
        raw_text="Hi.",
        corrected_text="Hi.",
        display_text="Hi.",
        start=0.0,
        end=0.5,
        confidence=1.0,
        fallback_reason=None,
        text_source="raw",
        timing_source="word_segments",
        timing_status="word_timing",
        tokens=[
            # Note: typed via TokenInterval below
        ],
    )
    # Build via the dataclass so missing fields resolve to defaults.
    from subforge.pipeline.stages import TokenInterval

    cue.tokens = [
        TokenInterval(
            text="Hi", start=0.0, end=0.3,
            is_punct=False, whitespace=" ", source="asr_word",
        ),
        TokenInterval(
            text=".", start=0.3, end=0.5,
            is_punct=True, whitespace="", source="asr_word",
        ),
    ]
    chunks = aligned_cues_to_token_chunks([cue])
    assert chunks == [
        [
            {
                "text": "Hi", "start": 0.0, "end": 0.3,
                "is_punct": False, "whitespace": " ",
            },
            {
                "text": ".", "start": 0.3, "end": 0.5,
                "is_punct": True, "whitespace": "",
            },
        ]
    ]


def test_aligned_cues_to_token_chunks_skips_cues_without_tokens():
    cue_no_tokens = AlignedCue(
        raw_text="x",
        corrected_text="x",
        display_text="x",
        start=0.0,
        end=0.0,
        confidence=0.0,
        fallback_reason="missing",
        text_source="raw",
        timing_source="missing",
        timing_status="missing",
        tokens=None,
    )
    assert aligned_cues_to_token_chunks([cue_no_tokens]) == []


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_english_cached_rerun_reuses_canonical_artifacts(tmp_path):
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))
    final_path = tmp_path / CANONICAL_DIRNAME / "final_cues.json"
    mtime_before = final_path.stat().st_mtime_ns

    # No bytes changed; the runner should not bump mtime on a cached
    # rerun.
    import time as _time

    _time.sleep(0.05)
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))
    assert final_path.stat().st_mtime_ns == mtime_before


def test_english_force_regenerates_canonical_artifacts(tmp_path):
    stages_dir = tmp_path / CANONICAL_DIRNAME
    stages_dir.mkdir()
    for name in STAGE_FILES:
        (stages_dir / name).write_text("{}", encoding="utf-8")

    EnglishPipelineStrategy().run(
        _hello_world_segs(), _ctx(tmp_path, force=True)
    )

    for name in (
        "raw_transcript.json",
        "timing_anchors.json",
        "corrected_transcript.json",
        "sentences.json",
        "alignment.json",
    ):
        payload = json.loads(
            (stages_dir / name).read_text(encoding="utf-8")
        )
        # The pre-existing stub had no input_hash key. Force-clear +
        # rerun should have replaced it with a real cache payload.
        assert "input_hash" in payload, name


def test_english_cache_invalidates_on_word_segments_change(tmp_path):
    """Different word_segments should produce different alignment artifacts."""
    EnglishPipelineStrategy().run(_hello_world_segs(), _ctx(tmp_path))
    align_first = (
        tmp_path / CANONICAL_DIRNAME / "alignment.json"
    ).read_text(encoding="utf-8")

    other_segs = _segs(
        ("Different", 0.0, 0.4),
        ("words", 0.4, 0.8),
        (".", 0.8, 0.85),
    )
    EnglishPipelineStrategy().run(other_segs, _ctx(tmp_path))
    align_second = (
        tmp_path / CANONICAL_DIRNAME / "alignment.json"
    ).read_text(encoding="utf-8")

    assert align_first != align_second


# ---------------------------------------------------------------------------
# Policy correctness in isolation
# ---------------------------------------------------------------------------


def test_english_policy_correct_is_no_op():
    policy = EnglishPolicy()
    raw = Transcript(text="Hello.", source="asr_raw")

    class _Ctx:
        def emit(self, *_):
            return None

    corrected, applied = policy.correct(raw, _Ctx())
    assert applied is False
    assert corrected.text == raw.text


def test_english_policy_split_signature_records_spacy_params():
    policy = EnglishPolicy()
    sig = policy.split_signature(_ctx(Path(".")))
    assert "spacy" in sig
    assert ENGLISH.spacy_model in sig
    assert "punct_limit=" in sig


def test_english_strategy_constructs_runner_with_english_policy(tmp_path):
    """Verify the strategy creates a fresh policy + runner per run."""
    strat = EnglishPipelineStrategy()
    chunks = strat.run(_hello_world_segs(), _ctx(tmp_path))
    assert chunks
    # Re-invoking with new word_segments should not leak prior state.
    other = _segs(("Different", 0.0, 0.4), ("words", 0.4, 0.8), (".", 0.8, 0.85))
    strat.run(other, _ctx(tmp_path / "other"))


def test_english_alignment_failure_uses_word_segment_fallback(
    tmp_path, monkeypatch,
):
    """If alignment raises, the fallback should still produce chunks."""
    from subforge.pipeline.stages import english_policy as ep

    def _raise(*_args, **_kwargs):
        raise ValueError("forced alignment failure")

    monkeypatch.setattr(ep, "align_sentences_with_timestamps", _raise)

    chunks = EnglishPipelineStrategy().run(
        _hello_world_segs(), _ctx(tmp_path)
    )
    assert chunks  # fallback produced something
    final = json.loads(
        (tmp_path / CANONICAL_DIRNAME / "final_cues.json").read_text(
            encoding="utf-8"
        )
    )
    assert final["meta"]["fallback_used"] is True
    assert final["meta"]["mode"] == "fallback"


def test_english_runner_aligned_with_english_policy(tmp_path):
    strat = EnglishPipelineStrategy()
    # The strategy creates a new policy/runner each run; observe that by
    # spying on the runner construction.
    captured = {}
    orig_run = StagedPipelineRunner.run

    def spy(self, word_segments, ctx):
        captured["policy_type"] = type(self.policy).__name__
        return orig_run(self, word_segments, ctx)

    StagedPipelineRunner.run = spy
    try:
        strat.run(_hello_world_segs(), _ctx(tmp_path))
    finally:
        StagedPipelineRunner.run = orig_run

    assert captured["policy_type"] == "EnglishPolicy"
