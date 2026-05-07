"""Char-level alignment for the CJK staged pipeline.

The CJK pipeline aligns corrected-transcript sentences to timing
anchors (and to the raw ASR transcript) by computing per-character
mappings via :class:`difflib.SequenceMatcher`. Each sentence's anchored
timing range is the convex hull of the timing anchors its corrected
characters resolve to; sentences with no resolvable anchor degrade to
a fallback cue so downstream stages can still emit something readable.
"""

from __future__ import annotations

import difflib
import logging

from subforge.pipeline.stages.models import (
    AlignedCue,
    Sentence,
    TimingAnchors,
    Transcript,
)

logger = logging.getLogger(__name__)


def map_corrected_to_raw(corrected: str, raw: str) -> list[int | None]:
    """Map each char index in *corrected* to the closest index in *raw*.

    Uses :class:`difflib.SequenceMatcher` so insertions/deletions/replacements
    introduced by the corrector still leave aligned regions intact.
    Positions in corrected text with no raw counterpart map to ``None``.
    """
    if not corrected:
        return []
    if not raw:
        return [None] * len(corrected)

    matcher = difflib.SequenceMatcher(a=corrected, b=raw, autojunk=False)
    mapping: list[int | None] = [None] * len(corrected)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        elif tag == "replace":
            n_corr = i2 - i1
            n_raw = j2 - j1
            if n_corr == 0 or n_raw == 0:
                continue
            for k in range(n_corr):
                mapping[i1 + k] = min(j1 + (k * n_raw) // n_corr, j2 - 1)
        # "delete" (chars only in corrected) and "insert" (chars only in
        # raw) leave the corrected positions unmapped.

    return mapping


def build_cue(
    sent: Sentence,
    *,
    raw: Transcript,
    corrected: Transcript,
    timing: TimingAnchors,
    corrected_to_raw: list[int | None],
    corrected_to_timing: list[int | None],
    correction_applied: bool,
) -> AlignedCue:
    """Resolve one sentence into a fully-decorated :class:`AlignedCue`."""
    timing_indices: list[int] = []
    for i in range(sent.char_start, sent.char_end):
        if 0 <= i < len(corrected_to_timing):
            ti = corrected_to_timing[i]
            if ti is not None and 0 <= ti < len(timing.anchors):
                timing_indices.append(ti)

    raw_chars: list[str] = []
    for i in range(sent.char_start, sent.char_end):
        if 0 <= i < len(corrected_to_raw):
            ri = corrected_to_raw[i]
            if ri is not None and 0 <= ri < len(raw.text):
                raw_chars.append(raw.text[ri])
    raw_text = "".join(raw_chars) if raw_chars else sent.text

    sent_len = max(sent.char_end - sent.char_start, 1)
    if timing_indices:
        anchors = [timing.anchors[i] for i in timing_indices]
        start = min(a.start for a in anchors)
        end = max(a.end for a in anchors)
        if end < start:
            end = start
        confidence = len(timing_indices) / sent_len
        fallback_reason: str | None = None
        cue_status = timing.status
    else:
        start = end = 0.0
        confidence = 0.0
        fallback_reason = "no_timing_anchor"
        cue_status = "missing"

    if correction_applied and fallback_reason is None:
        display_text = sent.text
        text_source = "corrected"
    else:
        # Corrector was rejected, or the cue lost its anchor — prefer the
        # raw ASR text so the user still gets something to read.
        display_text = raw_text if raw_text else sent.text
        text_source = "raw"

    return AlignedCue(
        raw_text=raw_text,
        corrected_text=sent.text,
        display_text=display_text,
        start=start,
        end=end,
        confidence=confidence,
        fallback_reason=fallback_reason,
        text_source=text_source,
        timing_source=timing.source,
        timing_status=cue_status,
    )


def align_cjk(
    sentences: list[Sentence],
    raw: Transcript,
    corrected: Transcript,
    timing: TimingAnchors,
    correction_applied: bool,
) -> tuple[list[AlignedCue], str | None]:
    """Char-level alignment of corrected sentences to timing anchors.

    Returns ``(cues, None)`` on success. On failure to compute the
    corrected→raw or corrected→timing mappings, returns
    ``([], "mapping_failed")``. When every sentence loses its timing
    anchor returns ``([], "no_timing_anchor")``.
    """
    try:
        corrected_to_raw = map_corrected_to_raw(corrected.text, raw.text)
        corrected_to_timing = map_corrected_to_raw(corrected.text, timing.text)
    except Exception as exc:  # noqa: BLE001 — alignment boundary
        logger.warning("Char-level alignment failed: %s", exc)
        return [], "mapping_failed"

    cues: list[AlignedCue] = []
    any_anchored = False
    for sent in sentences:
        cue = build_cue(
            sent,
            raw=raw,
            corrected=corrected,
            timing=timing,
            corrected_to_raw=corrected_to_raw,
            corrected_to_timing=corrected_to_timing,
            correction_applied=correction_applied,
        )
        if cue.fallback_reason is None:
            any_anchored = True
        cues.append(cue)

    if not any_anchored:
        return [], "no_timing_anchor"
    return cues, None


__all__ = [
    "map_corrected_to_raw",
    "build_cue",
    "align_cjk",
]
