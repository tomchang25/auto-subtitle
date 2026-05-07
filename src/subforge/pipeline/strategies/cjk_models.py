"""Compatibility re-exports for the CJK pipeline.

The transcript / timing / sentence / aligned-cue / pipeline-result types
now live in :mod:`subforge.pipeline.stages.models` under language-agnostic
names. This module preserves the historical CJK import surface so existing
callers (and tests) can keep importing ``CjkTranscript``, ``CjkAlignedCue``,
``word_segments_to_cjk_inputs``, and friends from this exact path.

Each legacy ``Cjk*`` symbol is an alias for the corresponding shared class
or helper, so ``isinstance``, dataclass equality, and ``from_dict`` behave
identically whether code uses the old or the new name.

The writer-format helper :func:`cjk_cues_to_writer_chunks` stays defined
here because its return type — raw per-character dicts shaped for the
subtitle writer — is intentionally not part of the shared stage-model
vocabulary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from subforge.pipeline.stages.models import (
    TIMING_STATUSES,
    AlignedCue as CjkAlignedCue,
    PipelineResult as CjkPipelineResult,
    Sentence as CjkSentence,
    TimingAnchor as CjkTimingAnchor,
    TimingAnchors as CjkTimingAnchors,
    TokenInterval,
    Transcript as CjkTranscript,
    build_split_inputs as build_split_cjk_inputs,
    word_segments_to_inputs as word_segments_to_cjk_inputs,
)

if TYPE_CHECKING:
    from subforge.nlp.lang_profile import LanguageProfile


def cjk_cues_to_writer_chunks(
    cues: list[CjkAlignedCue],
    profile: "LanguageProfile",
) -> list[list[dict]]:
    """Convert aligned CJK cues into the writer's per-character token format.

    The shared length-split / short-merge passes operate on
    ``list[list[token]]`` where every token has ``text``, ``start``, ``end``
    and ``is_punct``. Rather than reusing English-style word tokens, each
    cue's ``display_text`` is exploded into per-character tokens with timing
    distributed uniformly across ``[cue.start, cue.end]``. Cues with no
    display text are skipped.
    """
    punct = profile.punctuation
    chunks: list[list[dict]] = []
    for cue in cues:
        text = cue.display_text
        if not text:
            continue
        n = len(text)
        duration = max(cue.end - cue.start, 0.0)
        step = duration / n if n > 0 else 0.0
        tokens: list[dict] = []
        for i, ch in enumerate(text):
            ts = cue.start + i * step
            te = cue.end if i + 1 == n else cue.start + (i + 1) * step
            tokens.append({
                "text": ch,
                "whitespace": "",
                "is_punct": ch in punct,
                "start": ts,
                "end": te,
            })
        if tokens:
            chunks.append(tokens)
    return chunks


__all__ = [
    "TIMING_STATUSES",
    "CjkAlignedCue",
    "CjkPipelineResult",
    "CjkSentence",
    "CjkTimingAnchor",
    "CjkTimingAnchors",
    "CjkTranscript",
    "TokenInterval",
    "build_split_cjk_inputs",
    "cjk_cues_to_writer_chunks",
    "word_segments_to_cjk_inputs",
]
