"""Shared staged runner for the subtitle pipeline.

The runner owns the language-agnostic skeleton: stage order, per-stage
artifact filenames, the canonical artifact directory, force-clear of stale
caches, and read-from-canonical-only caching. Language-specific behavior
(input shaping, correction, sentence splitting, alignment, postprocess,
fallback) is delegated to a :class:`Policy` callback bag.

Stage order:

1. inputs                 — :class:`Transcript` + :class:`TimingAnchors`
2. correction             — :class:`Transcript` (corrected)
3. sentence splitting     — ``list[Sentence]``
4. alignment              — ``list[AlignedCue]`` (or fallback)
5. postprocess / final    — writer chunks + final-cues metadata

For backward compatibility a policy may declare a legacy artifact
directory; the runner mirrors each stage artifact to that directory after a
successful compute. The runner never reads the legacy directory back as
cache input — canonical staged artifacts are the only cache source.

The runner is currently used by the CJK strategy. English will land later
through its own policy without touching this file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from subforge.pipeline.stages.models import (
    AlignedCue,
    Sentence,
    TimingAnchors,
    Transcript,
)

if TYPE_CHECKING:
    from subforge.pipeline.strategies.base import StrategyContext

logger = logging.getLogger(__name__)


# Bumped whenever the staged-runner artifact contract changes so caches
# from earlier layouts (including the pre-runner CJK strategy) are
# ignored on the first post-bump run.
STAGE_SCHEMA_VERSION = "v4"

# Canonical artifact directory under each project. The runner reads only
# from this directory; legacy mirror directories declared by the policy
# are write-only.
CANONICAL_DIRNAME = "stages"

STAGE_FILES: tuple[str, ...] = (
    "raw_transcript.json",
    "timing_anchors.json",
    "corrected_transcript.json",
    "sentences.json",
    "alignment.json",
    "final_cues.json",
)


class Policy(Protocol):
    """Per-language callbacks driven by :class:`StagedPipelineRunner`.

    The protocol is intentionally minimal — only the methods needed for
    the CJK extraction. Speculative hooks for English or other languages
    are deferred until those policies actually land.
    """

    @property
    def corrector_id(self) -> str: ...

    @property
    def stage_label(self) -> str: ...

    def legacy_artifact_dir(
        self, ctx: "StrategyContext"
    ) -> Path | None: ...

    def short_circuit(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> tuple[list[list[dict]], dict] | None: ...

    def stage_inputs_hash(
        self,
        schema_version: str,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> str: ...

    def split_signature(self, ctx: "StrategyContext") -> str: ...

    def build_inputs(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> tuple[Transcript, TimingAnchors]: ...

    def correct(
        self,
        raw: Transcript,
        ctx: "StrategyContext",
    ) -> tuple[Transcript, bool]: ...

    def split_sentences(
        self,
        corrected: Transcript,
        ctx: "StrategyContext",
    ) -> list[Sentence]: ...

    def align(
        self,
        sentences: list[Sentence],
        raw: Transcript,
        corrected: Transcript,
        timing: TimingAnchors,
        correction_applied: bool,
        ctx: "StrategyContext",
    ) -> tuple[list[AlignedCue], str | None]: ...

    def postprocess(
        self,
        cues: list[AlignedCue],
        ctx: "StrategyContext",
    ) -> tuple[list[list[dict]], dict]: ...

    def fallback(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
        raw: Transcript,
        timing: TimingAnchors,
        fallback_reason: str | None,
    ) -> tuple[list[list[dict]], dict]: ...

    def summarise_meta(
        self,
        cues: list[AlignedCue],
        raw: Transcript,
        timing: TimingAnchors,
        ctx: "StrategyContext",
    ) -> dict: ...


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _load_stage(path: Path, expected_hash: str) -> dict | list | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Stage cache unreadable at %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    if data.get("input_hash") != expected_hash:
        return None
    return data.get("data")


def _serialize_stage(input_hash: str, data) -> str:
    return json.dumps(
        {"input_hash": input_hash, "data": data},
        ensure_ascii=False,
        indent=2,
    )


def _write_text_if_changed(path: Path, content: str) -> None:
    """Write ``content`` to *path* unless the file already has it.

    Used for the legacy mirror so repeated cached runs don't bump mtime
    when the canonical artifact didn't change.
    """
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing == content:
            return
    path.write_text(content, encoding="utf-8")


def _save_stage_with_mirror(
    canonical_path: Path,
    legacy_dir: Path | None,
    file_name: str,
    input_hash: str,
    data,
) -> None:
    serialized = _serialize_stage(input_hash, data)
    canonical_path.write_text(serialized, encoding="utf-8")
    if legacy_dir is not None:
        _write_text_if_changed(legacy_dir / file_name, serialized)


def _delete_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class StagedPipelineRunner:
    """Generic staged orchestration driven by a :class:`Policy`.

    The runner is responsible for:

    * locating the canonical artifact directory and the optional legacy
      mirror directory,
    * clearing stale stage files when ``ctx.force`` is set,
    * routing each stage through cache-load → policy-compute → cache-save,
    * mirroring each successfully-computed stage artifact to the legacy
      directory (write-only, never read back),
    * dispatching to ``policy.short_circuit`` (e.g. CJK benchmark mode)
      and to ``policy.fallback`` when alignment yields no anchored cues,
    * persisting ``final_cues.json`` with the policy-supplied metadata.
    """

    SCHEMA_VERSION = STAGE_SCHEMA_VERSION
    CANONICAL_DIRNAME = CANONICAL_DIRNAME
    STAGE_FILES = STAGE_FILES

    def __init__(self, policy: Policy):
        self.policy = policy

    def run(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> list[list[dict]]:
        canonical_dir = ctx.project_dir / self.CANONICAL_DIRNAME
        canonical_dir.mkdir(parents=True, exist_ok=True)
        legacy_dir = self.policy.legacy_artifact_dir(ctx)
        if legacy_dir is not None:
            legacy_dir.mkdir(parents=True, exist_ok=True)

        if ctx.force:
            self._clear_stage_files(canonical_dir, legacy_dir)

        short = self.policy.short_circuit(word_segments, ctx)
        if short is not None:
            chunks, meta = short
            self._write_final_cues(canonical_dir, legacy_dir, chunks, meta)
            return chunks

        ws_hash = self.policy.stage_inputs_hash(
            self.SCHEMA_VERSION, word_segments, ctx
        )

        raw, timing = self._stage_inputs(
            word_segments, ctx, canonical_dir, legacy_dir, ws_hash
        )
        ctx.check_cancel()

        corrected, correction_applied = self._stage_correct(
            raw, ctx, canonical_dir, legacy_dir, ws_hash
        )
        ctx.check_cancel()

        sentences = self._stage_split_sentences(
            corrected, ctx, canonical_dir, legacy_dir, ws_hash
        )
        ctx.check_cancel()

        cues, fallback_reason = self._stage_align(
            sentences,
            raw,
            corrected,
            timing,
            correction_applied,
            ctx,
            canonical_dir,
            legacy_dir,
            ws_hash,
        )
        ctx.check_cancel()

        if not cues:
            chunks, meta = self.policy.fallback(
                word_segments, ctx, raw, timing, fallback_reason
            )
        else:
            chunks, post_diag = self.policy.postprocess(cues, ctx)
            meta = self.policy.summarise_meta(cues, raw, timing, ctx)
            meta["postprocess"] = post_diag

        self._write_final_cues(canonical_dir, legacy_dir, chunks, meta)
        return chunks

    # ------------------------------------------------------------------
    # Stage 1 — input shaping
    # ------------------------------------------------------------------
    def _stage_inputs(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
        canonical_dir: Path,
        legacy_dir: Path | None,
        ws_hash: str,
    ) -> tuple[Transcript, TimingAnchors]:
        raw_path = canonical_dir / "raw_transcript.json"
        timing_path = canonical_dir / "timing_anchors.json"

        cached_raw = None if ctx.force else _load_stage(raw_path, ws_hash)
        cached_timing = (
            None if ctx.force else _load_stage(timing_path, ws_hash)
        )
        if cached_raw is not None and cached_timing is not None:
            ctx.emit(
                self.policy.stage_label,
                "Stage 1: raw transcript + timing anchors (cached)",
            )
            return (
                Transcript.from_dict(cached_raw),
                TimingAnchors.from_dict(cached_timing),
            )

        raw, timing = self.policy.build_inputs(word_segments, ctx)
        _save_stage_with_mirror(
            raw_path, legacy_dir, "raw_transcript.json", ws_hash, raw.to_dict()
        )
        _save_stage_with_mirror(
            timing_path,
            legacy_dir,
            "timing_anchors.json",
            ws_hash,
            timing.to_dict(),
        )
        return raw, timing

    # ------------------------------------------------------------------
    # Stage 2 — correction
    # ------------------------------------------------------------------
    def _stage_correct(
        self,
        raw: Transcript,
        ctx: "StrategyContext",
        canonical_dir: Path,
        legacy_dir: Path | None,
        ws_hash: str,
    ) -> tuple[Transcript, bool]:
        canonical_path = canonical_dir / "corrected_transcript.json"
        corrector_id = self.policy.corrector_id
        input_hash = _hash(ws_hash, raw.text, corrector_id)

        cached = None if ctx.force else _load_stage(canonical_path, input_hash)
        if cached is not None:
            ctx.emit(
                self.policy.stage_label,
                f"Stage 2: corrected transcript (cached, {corrector_id})",
            )
            return (
                Transcript(
                    text=cached["text"],
                    source=cached.get("source", "corrector"),
                ),
                bool(cached.get("applied", False)),
            )

        corrected, applied = self.policy.correct(raw, ctx)
        data = {
            "text": corrected.text,
            "source": corrected.source,
            "corrector": corrector_id,
            "applied": applied,
        }
        _save_stage_with_mirror(
            canonical_path,
            legacy_dir,
            "corrected_transcript.json",
            input_hash,
            data,
        )
        return corrected, applied

    # ------------------------------------------------------------------
    # Stage 3 — sentence split
    # ------------------------------------------------------------------
    def _stage_split_sentences(
        self,
        corrected: Transcript,
        ctx: "StrategyContext",
        canonical_dir: Path,
        legacy_dir: Path | None,
        ws_hash: str,
    ) -> list[Sentence]:
        canonical_path = canonical_dir / "sentences.json"
        input_hash = _hash(
            ws_hash, corrected.text, self.policy.split_signature(ctx)
        )

        cached = None if ctx.force else _load_stage(canonical_path, input_hash)
        if cached is not None:
            ctx.emit(
                self.policy.stage_label,
                f"Stage 3: sentences (cached, {len(cached)})",
            )
            return [Sentence.from_dict(s) for s in cached]

        sentences = self.policy.split_sentences(corrected, ctx)
        _save_stage_with_mirror(
            canonical_path,
            legacy_dir,
            "sentences.json",
            input_hash,
            [s.to_dict() for s in sentences],
        )
        return sentences

    # ------------------------------------------------------------------
    # Stage 4 — alignment
    # ------------------------------------------------------------------
    def _stage_align(
        self,
        sentences: list[Sentence],
        raw: Transcript,
        corrected: Transcript,
        timing: TimingAnchors,
        correction_applied: bool,
        ctx: "StrategyContext",
        canonical_dir: Path,
        legacy_dir: Path | None,
        ws_hash: str,
    ) -> tuple[list[AlignedCue], str | None]:
        canonical_path = canonical_dir / "alignment.json"
        input_hash = _hash(
            ws_hash,
            raw.text,
            corrected.text,
            json.dumps([s.to_dict() for s in sentences], ensure_ascii=False),
            timing.source,
            timing.status,
        )

        cached = None if ctx.force else _load_stage(canonical_path, input_hash)
        if cached is not None:
            ctx.emit(
                self.policy.stage_label,
                f"Stage 4: alignment (cached, {len(cached)} cues)",
            )
            return [AlignedCue.from_dict(c) for c in cached], None

        cues, fallback_reason = self.policy.align(
            sentences,
            raw,
            corrected,
            timing,
            correction_applied,
            ctx,
        )
        if cues:
            _save_stage_with_mirror(
                canonical_path,
                legacy_dir,
                "alignment.json",
                input_hash,
                [c.to_dict() for c in cues],
            )
        return cues, fallback_reason

    # ------------------------------------------------------------------
    # Final cues + cleanup
    # ------------------------------------------------------------------
    def _write_final_cues(
        self,
        canonical_dir: Path,
        legacy_dir: Path | None,
        chunks: list[list[dict]],
        meta: dict,
    ) -> None:
        path = canonical_dir / "final_cues.json"
        payload = {"meta": meta, "chunks": chunks}
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        path.write_text(serialized, encoding="utf-8")
        if legacy_dir is not None:
            _write_text_if_changed(legacy_dir / "final_cues.json", serialized)

    def _clear_stage_files(
        self,
        canonical_dir: Path,
        legacy_dir: Path | None,
    ) -> None:
        for name in self.STAGE_FILES:
            _delete_if_exists(canonical_dir / name)
            if legacy_dir is not None:
                _delete_if_exists(legacy_dir / name)


__all__ = [
    "CANONICAL_DIRNAME",
    "Policy",
    "STAGE_FILES",
    "STAGE_SCHEMA_VERSION",
    "StagedPipelineRunner",
]
