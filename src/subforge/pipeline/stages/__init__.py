"""Shared, language-agnostic stage models for the subtitle pipeline.

These types describe the transcript-first data contract that flows between
pipeline stages: transcript text, per-character timing anchors, sentence
units, optional per-token timing, aligned cues, and the aggregated
pipeline result. They are language-agnostic so the same vocabulary fits
both the CJK and English (and future) paths.

The CJK strategy currently imports the legacy ``Cjk*`` names from
:mod:`subforge.pipeline.strategies.cjk_models`; that module now re-exports
the symbols defined here under the legacy names so existing call sites
keep working unchanged.
"""

from __future__ import annotations

from subforge.pipeline.stages.models import (
    TIMING_STATUSES,
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
from subforge.pipeline.stages.runner import (
    CANONICAL_DIRNAME,
    STAGE_FILES,
    STAGE_SCHEMA_VERSION,
    Policy,
    StagedPipelineRunner,
)

__all__ = [
    "CANONICAL_DIRNAME",
    "Policy",
    "STAGE_FILES",
    "STAGE_SCHEMA_VERSION",
    "StagedPipelineRunner",
    "TIMING_STATUSES",
    "AlignedCue",
    "PipelineResult",
    "Sentence",
    "TimingAnchor",
    "TimingAnchors",
    "TokenInterval",
    "Transcript",
    "build_split_inputs",
    "word_segments_to_inputs",
]
