"""Shared, language-agnostic stage models and runner for the subtitle pipeline.

Modules in this package:

* :mod:`models` — dataclass contract that flows between stages
  (transcript text, per-character timing anchors, sentence units,
  optional per-token timing, aligned cues, and the aggregated pipeline
  result).
* :mod:`cache` — schema constants, hash helpers, and the canonical /
  legacy mirror cache write/load helpers.
* :mod:`policy` — :class:`Policy` protocol that the runner calls into
  for language-specific behavior.
* :mod:`runner` — :class:`StagedPipelineRunner`, the language-agnostic
  staged orchestration.
* :mod:`postprocess_helpers` — shared refine/split/merge passes used by
  CJK fallback and English postprocess.
* :mod:`cjk_policy` — concrete :class:`CjkPolicy`.
* :mod:`english_policy` — concrete :class:`EnglishPolicy`.

The legacy CJK names (``CjkTranscript``, ``CjkAlignedCue``, …) live in
:mod:`subforge.pipeline.strategies.cjk_models` as compatibility aliases.
"""

from __future__ import annotations

from subforge.pipeline.stages.cache import (
    CANONICAL_DIRNAME,
    STAGE_FILES,
    STAGE_SCHEMA_VERSION,
)
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
from subforge.pipeline.stages.policy import Policy
from subforge.pipeline.stages.runner import StagedPipelineRunner

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
