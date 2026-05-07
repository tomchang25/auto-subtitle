"""Generic staged orchestration for the subtitle pipeline.

The runner owns the language-agnostic skeleton: stage order, force-clear
of stale caches, and read-from-canonical-only caching with legacy mirror
backfill. Cache helpers, schema constants, and the policy protocol live
in sibling modules so this file can focus on orchestration alone.

Stage order:

1. inputs                 — :class:`Transcript` + :class:`TimingAnchors`
2. correction             — :class:`Transcript` (corrected)
3. sentence splitting     — ``list[Sentence]``
4. alignment              — ``list[AlignedCue]`` (or fallback)
5. postprocess / final    — writer chunks + final-cues metadata

The runner is currently used by the CJK strategy. English will land
later through its own policy without touching this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from subforge.pipeline.stages.cache import (
    CANONICAL_DIRNAME,
    STAGE_FILES,
    STAGE_SCHEMA_VERSION,
    cached_stage_with_mirror,
    clear_stage_files,
    hash_inputs,
    save_stage_with_mirror,
    write_artifact_with_mirror,
)
from subforge.pipeline.stages.models import (
    AlignedCue,
    Sentence,
    TimingAnchors,
    Transcript,
)
from subforge.pipeline.stages.policy import Policy

if TYPE_CHECKING:
    from subforge.pipeline.strategies.base import StrategyContext


class StagedPipelineRunner:
    """Generic staged orchestration driven by a :class:`Policy`."""

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
            clear_stage_files(canonical_dir, legacy_dir)

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
        raw_canonical = canonical_dir / "raw_transcript.json"
        timing_canonical = canonical_dir / "timing_anchors.json"
        raw_legacy = (
            legacy_dir / "raw_transcript.json" if legacy_dir is not None else None
        )
        timing_legacy = (
            legacy_dir / "timing_anchors.json" if legacy_dir is not None else None
        )

        cached_raw = (
            None
            if ctx.force
            else cached_stage_with_mirror(raw_canonical, raw_legacy, ws_hash)
        )
        cached_timing = (
            None
            if ctx.force
            else cached_stage_with_mirror(
                timing_canonical, timing_legacy, ws_hash
            )
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
        save_stage_with_mirror(
            raw_canonical, raw_legacy, ws_hash, raw.to_dict()
        )
        save_stage_with_mirror(
            timing_canonical, timing_legacy, ws_hash, timing.to_dict()
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
        legacy_path = (
            legacy_dir / "corrected_transcript.json"
            if legacy_dir is not None
            else None
        )
        corrector_id = self.policy.corrector_id
        input_hash = hash_inputs(ws_hash, raw.text, corrector_id)

        cached = (
            None
            if ctx.force
            else cached_stage_with_mirror(
                canonical_path, legacy_path, input_hash
            )
        )
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
        save_stage_with_mirror(canonical_path, legacy_path, input_hash, data)
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
        legacy_path = (
            legacy_dir / "sentences.json" if legacy_dir is not None else None
        )
        input_hash = hash_inputs(
            ws_hash, corrected.text, self.policy.split_signature(ctx)
        )

        cached = (
            None
            if ctx.force
            else cached_stage_with_mirror(
                canonical_path, legacy_path, input_hash
            )
        )
        if cached is not None:
            ctx.emit(
                self.policy.stage_label,
                f"Stage 3: sentences (cached, {len(cached)})",
            )
            return [Sentence.from_dict(s) for s in cached]

        sentences = self.policy.split_sentences(corrected, ctx)
        save_stage_with_mirror(
            canonical_path,
            legacy_path,
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
        legacy_path = (
            legacy_dir / "alignment.json" if legacy_dir is not None else None
        )
        input_hash = hash_inputs(
            ws_hash,
            raw.text,
            corrected.text,
            json.dumps([s.to_dict() for s in sentences], ensure_ascii=False),
            timing.source,
            timing.status,
        )

        cached = (
            None
            if ctx.force
            else cached_stage_with_mirror(
                canonical_path, legacy_path, input_hash
            )
        )
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
            save_stage_with_mirror(
                canonical_path,
                legacy_path,
                input_hash,
                [c.to_dict() for c in cues],
            )
        return cues, fallback_reason

    # ------------------------------------------------------------------
    # Final cues
    # ------------------------------------------------------------------
    def _write_final_cues(
        self,
        canonical_dir: Path,
        legacy_dir: Path | None,
        chunks: list[list[dict]],
        meta: dict,
    ) -> None:
        canonical_path = canonical_dir / "final_cues.json"
        legacy_path = (
            legacy_dir / "final_cues.json" if legacy_dir is not None else None
        )
        payload = {"meta": meta, "chunks": chunks}
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        write_artifact_with_mirror(canonical_path, legacy_path, serialized)


__all__ = [
    "StagedPipelineRunner",
]
