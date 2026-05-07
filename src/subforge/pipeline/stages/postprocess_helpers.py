"""Shared postprocess helpers used by language policies.

Both the CJK fallback path and the English token-aware postprocess
share the same refine -> split-long -> merge-short sequence over
``list[list[dict]]`` token chunks. Centralising it here avoids drift
between the two policies.
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


__all__ = ["finalize_token_chunks"]
