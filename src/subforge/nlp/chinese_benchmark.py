"""Hard-cut segmentation for Chinese ASR benchmark mode.

Bypasses punctuation restoration, semantic splitting, timing refinement,
soft segmentation, and short-segment merging to preserve raw ASR text
for clean benchmark comparison against the full NLP pipeline.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SENTENCE_END = frozenset("。！？.!?")
_PUNCT = frozenset("，。！？、；：,.!?;:")

DEFAULT_HARD_CHARS = 30
DEFAULT_GAP_SECONDS = 1.5


def hard_cut_chinese_segments(
    word_segments: list[dict],
    hard_chars: int = DEFAULT_HARD_CHARS,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> list[list[dict]]:
    """Convert raw ASR word segments into subtitle chunks via deterministic hard cuts.

    Cuts after sentence-end punctuation (。！？.!?), when accumulated character
    count reaches hard_chars, or when there is a timing gap > gap_seconds.

    Segments with missing or zero timestamps are still included so that ASR
    text is preserved as faithfully as possible; timing gaps are only evaluated
    when valid timestamps are available.
    """
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    prev_end = -1.0

    for seg in word_segments:
        word = seg.get("word", "")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        has_valid_time = end > start

        tok = {
            "text": word,
            "whitespace": "",
            "is_punct": bool(word and word[-1] in _PUNCT),
            "start": start,
            "end": end,
        }

        # Flush current chunk on a large timing gap (only when timestamps are valid)
        if current and has_valid_time and prev_end >= 0 and start - prev_end > gap_seconds:
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(tok)
        current_chars += len(word)
        if has_valid_time:
            prev_end = end

        # Cut after sentence-end punctuation
        if word and word[-1] in _SENTENCE_END:
            chunks.append(current)
            current = []
            current_chars = 0
            continue

        # Hard character-count cut
        if current_chars >= hard_chars:
            chunks.append(current)
            current = []
            current_chars = 0

    if current:
        chunks.append(current)

    return chunks
