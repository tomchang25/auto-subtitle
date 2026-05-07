"""English (and other spaCy-based) subtitle strategy.

Wraps the existing alphabetic-language flow: spaCy sentence split → token
alignment → timing refinement → length-based split → short-segment merge.
The behavior is intentionally identical to the inline branch that lived in
``SubtitlePipeline.run`` before the strategy split.
"""

from __future__ import annotations

from subforge.config import (
    BREATH_GAP,
    MAX_GAP,
    MERGE_MAX_DURATION,
    MERGE_MAX_GAP,
    MIN_DURATION,
    MIN_WORDS_FOR_BREATH_SPLIT,
    SEG_PAUSE_THRESHOLD,
)
from subforge.nlp.alignment import (
    align_sentences_with_timestamps,
    refine_sentences_by_timing,
)
from subforge.nlp.segmentation import (
    merge_short_segments,
    split_long_sentences_by_length,
)
from subforge.nlp.text_semantically import split_to_sentences
from subforge.pipeline.strategies.base import (
    LanguagePipelineStrategy,
    StrategyContext,
)


class EnglishPipelineStrategy(LanguagePipelineStrategy):
    def run(
        self,
        word_segments: list[dict],
        ctx: StrategyContext,
    ) -> list[list[dict]]:
        profile = ctx.profile

        ctx.emit("NLP", "Splitting into sentences (spaCy)")
        full_text = profile.join_token.join(seg["word"] for seg in word_segments)
        sentence_chunks = split_to_sentences(full_text)
        ctx.check_cancel()

        ctx.emit("Align", "Aligning sentences with timestamps")
        aligned = align_sentences_with_timestamps(word_segments, sentence_chunks)
        ctx.check_cancel()

        ctx.emit("Refine", "Refining segment timing")
        refined = refine_sentences_by_timing(
            aligned,
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
