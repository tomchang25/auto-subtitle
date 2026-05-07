"""Token-aware postprocess for the English staged pipeline.

This module owns three pieces:

* :func:`finalize_token_chunks` — the shared refine → split-long →
  merge-short sequence over ``list[list[dict]]`` token chunks. Used
  both by the English postprocess stage and the CJK fallback path.
  Calls ``ctx.emit`` and ``ctx.check_cancel`` because the per-step
  progress reporting is part of its contract.
* :func:`aligned_cues_to_token_chunks` — adapter that converts
  :class:`AlignedCue` tokens to the legacy token-chunk dict shape
  expected by the postprocess primitives and the writer.
* :func:`postprocess_english` — the English policy's postprocess body.
  Drives the adapter and the shared finalize sequence and returns the
  diagnostic metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from subforge.config import (
    BREATH_GAP,
    MAX_GAP,
    MERGE_MAX_DURATION,
    MERGE_MAX_GAP,
    MIN_DURATION,
    MIN_WORDS_FOR_BREATH_SPLIT,
    SEG_PAUSE_THRESHOLD,
)
from subforge.nlp.alignment import refine_sentences_by_timing
from subforge.nlp.segmentation import (
    merge_short_segments,
    split_long_sentences_by_length,
)
from subforge.pipeline.stages.models import AlignedCue

if TYPE_CHECKING:
    from subforge.pipeline.strategies.base import StrategyContext


def finalize_token_chunks(
    chunks: list[list[dict]],
    ctx: "StrategyContext",
) -> list[list[dict]]:
    """Run the shared refine / split / merge passes over token chunks."""
    profile = ctx.profile

    ctx.emit("Refine", "Refining segment timing")
    refined = refine_sentences_by_timing(
        chunks,
        min_duration=MIN_DURATION,
        max_gap=MAX_GAP,
        breath_gap=BREATH_GAP,
        min_words_for_breath_split=MIN_WORDS_FOR_BREATH_SPLIT,
    )
    ctx.check_cancel()

    ctx.emit("Split", "Splitting long segments")
    refined = split_long_sentences_by_length(
        refined,
        min_words=profile.seg_min,
        max_words=profile.seg_hard,
        soft_words=profile.seg_soft,
        pause_threshold=SEG_PAUSE_THRESHOLD,
        profile=profile,
    )

    ctx.emit("Merge", "Merging short segments")
    refined = merge_short_segments(
        refined,
        max_words=profile.merge_max,
        max_duration=MERGE_MAX_DURATION,
        max_gap=MERGE_MAX_GAP,
        profile=profile,
    )
    ctx.check_cancel()
    return refined


def aligned_cues_to_token_chunks(
    cues: list[AlignedCue],
) -> list[list[dict]]:
    """Convert :class:`AlignedCue` tokens to the legacy token-chunk shape.

    The English postprocess primitives (``refine_sentences_by_timing``,
    ``split_long_sentences_by_length``, ``merge_short_segments``) all
    operate on ``list[list[dict]]`` where each token has ``text``,
    ``start``, ``end``, ``is_punct`` and (optionally) ``whitespace``.
    Preserving ``whitespace`` and ``is_punct`` is critical: the writer's
    naive ``" ".join(token["text"])`` relies on the same token shape the
    legacy English path produced, so dropping these fields would shift
    punctuation spacing.
    """
    chunks: list[list[dict]] = []
    for cue in cues:
        if not cue.tokens:
            continue
        chunk = [
            {
                "text": tok.text,
                "start": tok.start,
                "end": tok.end,
                "is_punct": tok.is_punct,
                "whitespace": tok.whitespace,
            }
            for tok in cue.tokens
        ]
        chunks.append(chunk)
    return chunks


def postprocess_english(
    cues: list[AlignedCue],
    ctx: "StrategyContext",
) -> tuple[list[list[dict]], dict]:
    """Token-aware postprocess for English aligned cues.

    Converts the cue tokens to the legacy token-chunk shape, runs the
    shared finalize sequence, and returns the chunks alongside a
    diagnostic metadata dict.
    """
    chunks = aligned_cues_to_token_chunks(cues)
    chunks = finalize_token_chunks(chunks, ctx)
    diag = {
        "input_cue_count": len(cues),
        "output_chunk_count": len(chunks),
        "token_intervals_used": all(
            c.tokens is not None and len(c.tokens) > 0 for c in cues
        ),
        "actions": ["refine_timing", "split_long", "merge_short"],
    }
    return chunks, diag


__all__ = [
    "finalize_token_chunks",
    "aligned_cues_to_token_chunks",
    "postprocess_english",
]
