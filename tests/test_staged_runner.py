"""Tests for the shared staged runner that drives the CJK pipeline.

These cover behaviors PR2 introduces on top of the existing CJK tests:

* Canonical staged artifacts under ``project_dir/stages/`` exist alongside
  the legacy ``project_dir/cjk/`` mirrors.
* The runner reads only canonical artifacts as cache input — legacy CJK
  artifacts are write-only.
* Stale pre-refactor artifacts are ignored after the schema bump.
* Force reruns refresh both canonical and legacy mirrors.
* Repeated cached runs do not bump the legacy mirror's mtime when the
  contents haven't changed.

Existing CJK test coverage (alignment, fallback, benchmark, split-source
inputs, corrector seam) still lives in ``test_cjk_pipeline.py`` and exercises
the same code paths through the shared runner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from subforge.nlp.lang_profile import CHINESE
from subforge.pipeline.stages import (
    CANONICAL_DIRNAME,
    STAGE_FILES,
    STAGE_SCHEMA_VERSION,
    StagedPipelineRunner,
)
from subforge.pipeline.strategies import (
    CjkPipelineStrategy,
    StrategyContext,
)
from subforge.pipeline.strategies.cjk import CjkPolicy


def _segs(*words_and_times):
    return [{"word": w, "start": s, "end": e} for w, s, e in words_and_times]


def _ctx(
    tmp_path: Path,
    *,
    profile=CHINESE,
    force=False,
    chinese_benchmark=False,
    transcript_text=None,
    transcript_source=None,
    transcript_backend=None,
    transcript_model=None,
    timing_backend=None,
    timing_model=None,
    transcript_fallback=None,
):
    return StrategyContext(
        profile=profile,
        project_dir=tmp_path,
        force=force,
        emit=lambda step, detail="": None,
        check_cancel=lambda: None,
        chinese_benchmark=chinese_benchmark,
        transcript_text=transcript_text,
        transcript_source=transcript_source,
        transcript_backend=transcript_backend,
        transcript_model=transcript_model,
        timing_backend=timing_backend,
        timing_model=timing_model,
        transcript_fallback=transcript_fallback,
    )


def _basic_segs():
    return _segs(
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


# ---------------------------------------------------------------------------
# Canonical artifacts
# ---------------------------------------------------------------------------


def test_runner_writes_canonical_artifacts_under_stages_dir(tmp_path):
    CjkPipelineStrategy().run(_basic_segs(), _ctx(tmp_path))

    stages_dir = tmp_path / CANONICAL_DIRNAME
    assert stages_dir.is_dir()
    for name in STAGE_FILES:
        assert (stages_dir / name).exists(), f"missing canonical: {name}"


def test_runner_mirrors_artifacts_to_legacy_cjk_dir(tmp_path):
    CjkPipelineStrategy().run(_basic_segs(), _ctx(tmp_path))

    cjk_dir = tmp_path / "cjk"
    stages_dir = tmp_path / CANONICAL_DIRNAME
    for name in STAGE_FILES:
        canonical = (stages_dir / name).read_text(encoding="utf-8")
        legacy = (cjk_dir / name).read_text(encoding="utf-8")
        assert canonical == legacy, f"legacy mirror diverged: {name}"


# ---------------------------------------------------------------------------
# Cache reads come only from canonical
# ---------------------------------------------------------------------------


def test_runner_reads_only_canonical_not_legacy(tmp_path):
    """If the legacy CJK dir holds different content, the runner ignores it.

    We populate the legacy dir with stage files whose ``input_hash`` would
    match a fresh run, but whose ``data`` is deliberately wrong. If the
    runner read them as cache, the resulting artifacts (and chunks) would
    show that wrong data. The canonical dir is empty so the runner must
    recompute everything.
    """
    cjk_dir = tmp_path / "cjk"
    cjk_dir.mkdir()
    poisoned_payload = {
        "input_hash": "this-will-not-match",
        "data": {"text": "POISONED", "source": "fake"},
    }
    for name in STAGE_FILES:
        (cjk_dir / name).write_text(
            json.dumps(poisoned_payload), encoding="utf-8"
        )

    chunks = CjkPipelineStrategy().run(_basic_segs(), _ctx(tmp_path))

    joined = "".join(tok["text"] for chunk in chunks for tok in chunk)
    assert "POISONED" not in joined
    assert "今天天氣很好" in joined

    # Canonical artifacts now exist and were freshly computed.
    canonical_raw = json.loads(
        (tmp_path / CANONICAL_DIRNAME / "raw_transcript.json").read_text(
            encoding="utf-8"
        )
    )
    assert canonical_raw["data"]["text"] == "今天天氣很好。我們走吧。"
    assert canonical_raw["data"]["source"] == "asr_raw"


def test_runner_recomputes_when_only_legacy_has_a_matching_cache(tmp_path):
    """A first run primes the legacy mirror. Deleting the canonical version
    should force a recompute on the next run, rather than the runner falling
    back to the legacy artifact as a cache source."""
    strat = CjkPipelineStrategy()
    strat.run(_basic_segs(), _ctx(tmp_path))

    canonical_raw_path = tmp_path / CANONICAL_DIRNAME / "raw_transcript.json"
    legacy_raw_path = tmp_path / "cjk" / "raw_transcript.json"

    # Capture the pristine legacy contents, then delete the canonical copy so
    # a cache hit is only possible if the runner illegally reads from legacy.
    legacy_raw_before = legacy_raw_path.read_text(encoding="utf-8")
    canonical_raw_path.unlink()
    assert not canonical_raw_path.exists()

    strat.run(_basic_segs(), _ctx(tmp_path))

    # Canonical was rewritten by the recompute path.
    assert canonical_raw_path.exists()
    # Legacy mirror wasn't disturbed because contents are identical.
    assert legacy_raw_path.read_text(encoding="utf-8") == legacy_raw_before


# ---------------------------------------------------------------------------
# Schema version invalidates pre-refactor caches
# ---------------------------------------------------------------------------


def test_stale_pre_refactor_artifacts_are_ignored_after_schema_bump(tmp_path):
    stages_dir = tmp_path / CANONICAL_DIRNAME
    stages_dir.mkdir()
    cjk_dir = tmp_path / "cjk"
    cjk_dir.mkdir()

    # Forge a v3-era artifact: the right shape, with an old schema baked
    # into the input_hash. If the runner ignored the schema version it
    # would pick this up as a cache hit and skip recompute.
    stale_payload = {
        "input_hash": "v3-stale-hash-from-previous-layout",
        "data": {"text": "STALE", "source": "v3"},
    }
    raw_canonical = stages_dir / "raw_transcript.json"
    raw_canonical.write_text(json.dumps(stale_payload), encoding="utf-8")
    raw_legacy = cjk_dir / "raw_transcript.json"
    raw_legacy.write_text(json.dumps(stale_payload), encoding="utf-8")

    CjkPipelineStrategy().run(_basic_segs(), _ctx(tmp_path))

    # Stale canonical artifact was overwritten with v4-shaped content.
    payload = json.loads(raw_canonical.read_text(encoding="utf-8"))
    assert payload["data"]["text"] == "今天天氣很好。我們走吧。"
    assert payload["data"]["source"] == "asr_raw"
    assert payload["input_hash"] != stale_payload["input_hash"]
    # Legacy mirror also refreshed.
    legacy = json.loads(raw_legacy.read_text(encoding="utf-8"))
    assert legacy["data"]["text"] == "今天天氣很好。我們走吧。"


def test_schema_version_constant_was_bumped():
    # PR2 expected schema bump. Locking the value prevents accidental
    # regressions to a previous layout.
    assert STAGE_SCHEMA_VERSION == "v4"


# ---------------------------------------------------------------------------
# Force reruns refresh both canonical and legacy mirrors
# ---------------------------------------------------------------------------


def test_force_clears_canonical_and_legacy_stage_files(tmp_path):
    stages_dir = tmp_path / CANONICAL_DIRNAME
    cjk_dir = tmp_path / "cjk"
    stages_dir.mkdir()
    cjk_dir.mkdir()

    for name in STAGE_FILES:
        (stages_dir / name).write_text("{}", encoding="utf-8")
        (cjk_dir / name).write_text("{}", encoding="utf-8")

    CjkPipelineStrategy().run(_basic_segs(), _ctx(tmp_path, force=True))

    for name in (
        "raw_transcript.json",
        "timing_anchors.json",
        "corrected_transcript.json",
        "sentences.json",
        "alignment.json",
    ):
        canonical_payload = json.loads(
            (stages_dir / name).read_text(encoding="utf-8")
        )
        legacy_payload = json.loads(
            (cjk_dir / name).read_text(encoding="utf-8")
        )
        # Both directories now hold real cache payloads, not the pre-existing
        # stub.
        assert "input_hash" in canonical_payload, name
        assert "input_hash" in legacy_payload, name
        assert canonical_payload == legacy_payload, name


# ---------------------------------------------------------------------------
# Cached re-runs avoid unnecessary legacy mirror rewrites
# ---------------------------------------------------------------------------


def test_repeated_cached_runs_do_not_bump_legacy_mirror_mtime(tmp_path):
    strat = CjkPipelineStrategy()
    ctx = _ctx(tmp_path)
    strat.run(_basic_segs(), ctx)

    legacy_raw = tmp_path / "cjk" / "raw_transcript.json"
    legacy_corrected = tmp_path / "cjk" / "corrected_transcript.json"
    legacy_alignment = tmp_path / "cjk" / "alignment.json"
    legacy_final = tmp_path / "cjk" / "final_cues.json"
    before = {
        path: path.stat().st_mtime_ns
        for path in (
            legacy_raw,
            legacy_corrected,
            legacy_alignment,
            legacy_final,
        )
    }

    # Sleep enough that any rewrite would land in a new mtime tick on
    # filesystems with coarse timestamp resolution.
    time.sleep(0.05)
    strat.run(_basic_segs(), ctx)

    for path, mtime_before in before.items():
        assert (
            path.stat().st_mtime_ns == mtime_before
        ), f"legacy mirror was rewritten unnecessarily: {path.name}"


# ---------------------------------------------------------------------------
# Strategy compatibility shim
# ---------------------------------------------------------------------------


def test_cjk_pipeline_strategy_is_thin_runner_wrapper():
    strat = CjkPipelineStrategy()
    # The strategy now binds a CjkPolicy to a StagedPipelineRunner.
    assert isinstance(strat._runner, StagedPipelineRunner)
    assert isinstance(strat._runner.policy, CjkPolicy)
    # Existing public construction (with a custom corrector) still works.
    from subforge.nlp.cjk_corrector import NoOpCorrector

    custom = NoOpCorrector()
    strat2 = CjkPipelineStrategy(corrector=custom)
    assert strat2._runner.policy.corrector is custom
