"""Policy protocol for the staged subtitle pipeline.

A policy supplies the language-specific behavior the generic
:class:`StagedPipelineRunner` cannot provide: how to shape inputs, run
correction, split sentences, align text to timing, postprocess into
writer chunks, and choose the fallback path. The runner owns the
staged orchestration and artifact lifecycle.

The protocol is intentionally minimal — only the methods the CJK
extraction needs. Speculative hooks for other languages are deferred
until those policies actually land.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from subforge.pipeline.stages.models import (
        AlignedCue,
        Sentence,
        TimingAnchors,
        Transcript,
    )
    from subforge.pipeline.strategies.base import StrategyContext


class Policy(Protocol):
    """Per-language callbacks driven by :class:`StagedPipelineRunner`."""

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
    ) -> tuple["Transcript", "TimingAnchors"]: ...

    def correct(
        self,
        raw: "Transcript",
        ctx: "StrategyContext",
    ) -> tuple["Transcript", bool]: ...

    def split_sentences(
        self,
        corrected: "Transcript",
        ctx: "StrategyContext",
    ) -> list["Sentence"]: ...

    def align(
        self,
        sentences: list["Sentence"],
        raw: "Transcript",
        corrected: "Transcript",
        timing: "TimingAnchors",
        correction_applied: bool,
        ctx: "StrategyContext",
    ) -> tuple[list["AlignedCue"], str | None]: ...

    def postprocess(
        self,
        cues: list["AlignedCue"],
        ctx: "StrategyContext",
    ) -> tuple[list[list[dict]], dict]: ...

    def fallback(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
        raw: "Transcript",
        timing: "TimingAnchors",
        fallback_reason: str | None,
    ) -> tuple[list[list[dict]], dict]: ...

    def summarise_meta(
        self,
        cues: list["AlignedCue"],
        raw: "Transcript",
        timing: "TimingAnchors",
        ctx: "StrategyContext",
    ) -> dict: ...


__all__ = ["Policy"]
