"""Cross-language ``final_cues.json`` metadata completeness tests.

Both the CJK and English staged pipelines write a ``meta`` block at the
head of ``final_cues.json``. PR4b normalizes that block so a downstream
consumer can rely on a fixed set of required keys regardless of which
policy produced the file (English normal, CJK normal, English fallback,
or CJK fallback).
"""

from __future__ import annotations

import json
from pathlib import Path

from subforge.nlp.lang_profile import CHINESE, ENGLISH
from subforge.pipeline.stages import (
    CANONICAL_DIRNAME,
    REQUIRED_META_KEYS,
)
from subforge.pipeline.strategies import (
    CjkPipelineStrategy,
    EnglishPipelineStrategy,
    StrategyContext,
)


def _segs(*words_and_times):
    return [{"word": w, "start": s, "end": e} for w, s, e in words_and_times]


def _ctx(tmp_path: Path, *, profile=CHINESE) -> StrategyContext:
    return StrategyContext(
        profile=profile,
        project_dir=tmp_path,
        force=False,
        emit=lambda step, detail="": None,
        check_cancel=lambda: None,
    )


def _cjk_basic_segs():
    return _segs(
        ("今", 0.0, 0.2),
        ("天", 0.2, 0.4),
        ("天", 0.4, 0.6),
        ("氣", 0.6, 0.8),
        ("很", 0.8, 1.0),
        ("好", 1.0, 1.2),
        ("。", 1.2, 1.3),
    )


def _english_basic_segs():
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


def _read_meta(canonical_dir: Path) -> dict:
    payload = json.loads(
        (canonical_dir / "final_cues.json").read_text(encoding="utf-8")
    )
    return payload["meta"]


def test_cjk_normal_run_meta_has_required_keys(tmp_path):
    CjkPipelineStrategy().run(_cjk_basic_segs(), _ctx(tmp_path))
    meta = _read_meta(tmp_path / CANONICAL_DIRNAME)
    missing = REQUIRED_META_KEYS - set(meta.keys())
    assert not missing, f"CJK normal meta missing keys: {sorted(missing)}"


def test_english_normal_run_meta_has_required_keys(tmp_path):
    EnglishPipelineStrategy().run(
        _english_basic_segs(), _ctx(tmp_path, profile=ENGLISH)
    )
    meta = _read_meta(tmp_path / CANONICAL_DIRNAME)
    missing = REQUIRED_META_KEYS - set(meta.keys())
    assert not missing, f"English normal meta missing keys: {sorted(missing)}"


def test_cjk_fallback_run_meta_has_required_keys(tmp_path, monkeypatch):
    """When CJK alignment returns no cues, fallback meta must still be complete."""
    from subforge.pipeline.stages import cjk_policy as cp

    def _empty_align(*_args, **_kwargs):
        return [], "mapping_failed"

    monkeypatch.setattr(cp, "align_cjk", _empty_align)

    CjkPipelineStrategy().run(_cjk_basic_segs(), _ctx(tmp_path))
    meta = _read_meta(tmp_path / CANONICAL_DIRNAME)
    missing = REQUIRED_META_KEYS - set(meta.keys())
    assert not missing, f"CJK fallback meta missing keys: {sorted(missing)}"
    assert meta["fallback_used"] is True


def test_english_fallback_run_meta_has_required_keys(tmp_path, monkeypatch):
    """When English alignment raises, fallback meta must still be complete."""
    from subforge.pipeline.stages import english_policy as ep

    def _raise(*_args, **_kwargs):
        raise ValueError("forced alignment failure")

    monkeypatch.setattr(ep, "align_sentences_with_timestamps", _raise)

    EnglishPipelineStrategy().run(
        _english_basic_segs(), _ctx(tmp_path, profile=ENGLISH)
    )
    meta = _read_meta(tmp_path / CANONICAL_DIRNAME)
    missing = REQUIRED_META_KEYS - set(meta.keys())
    assert not missing, f"English fallback meta missing keys: {sorted(missing)}"
    assert meta["fallback_used"] is True
