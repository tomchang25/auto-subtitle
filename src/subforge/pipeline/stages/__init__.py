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
* :mod:`cjk_policy` — :class:`CjkPolicy` wiring layer.
* :mod:`english_policy` — :class:`EnglishPolicy` wiring layer.
* :mod:`sentences` — concrete sentence-splitter implementations
  (spaCy, punctuation).
* :mod:`align` — concrete alignment implementations (char-level for
  CJK, word-level for English).
* :mod:`postprocess` — concrete postprocess implementations
  (token-aware word_count, display-width-aware) and the shared
  ``finalize_token_chunks`` helper.
* :mod:`fallback` — per-language fallback chunk assembly used when the
  transcript-first alignment path produces no cues.

The legacy CJK names (``CjkTranscript``, ``CjkAlignedCue``, …) live in
:mod:`subforge.pipeline.strategies.cjk_models` as compatibility aliases.
"""

from __future__ import annotations

from subforge.pipeline.stages.cache import (
    CANONICAL_DIRNAME,
    STAGE_FILES,
    STAGE_SCHEMA_VERSION,
)
from subforge.pipeline.stages.metadata import (
    REQUIRED_META_KEYS,
    validate_meta,
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
    "REQUIRED_META_KEYS",
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
    "validate_meta",
    "word_segments_to_inputs",
]
