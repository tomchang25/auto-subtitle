"""Per-language fallback chunk assembly.

When the transcript-first alignment path produces no cues (alignment
mapping failed, no anchors survived, or the english alignment raised),
the runner falls through to a per-language fallback that builds chunks
directly from the raw ASR ``word_segments``. The English path emits one
token per word and lets the shared finalize sequence merge them; the
CJK path uses the language's punctuation splitter.

Both functions return the chunks and a fallback metadata dict so the
runner can write ``final_cues.json`` with the correct provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from subforge.nlp.text_semantically import split_word_segments_by_punctuation
from subforge.pipeline.stages.models import TimingAnchors, Transcript
from subforge.pipeline.stages.postprocess.word_count import (
    finalize_token_chunks,
)

if TYPE_CHECKING:
    from subforge.pipeline.strategies.base import StrategyContext


def fallback_cjk(
    word_segments: list[dict],
    ctx: "StrategyContext",
    raw: Transcript,
    timing: TimingAnchors,
    fallback_reason: str | None,
    *,
    corrector_id: str = "none",
) -> tuple[list[list[dict]], dict]:
    """Build CJK fallback chunks via punctuation-driven word splitting."""
    chunks = split_word_segments_by_punctuation(word_segments, ctx.profile)
    chunks = finalize_token_chunks(chunks, ctx)
    meta = {
        "mode": "fallback",
        "profile": ctx.profile.code,
        "text_source": "raw_transcript",
        "timing_source": timing.source,
        "timing_status": "fallback",
        "transcript_backend": ctx.transcript_backend
        or ("sensevoice" if ctx.transcript_text is not None else "whisper"),
        "timing_backend": ctx.timing_backend or "whisper",
        "correction_mode": corrector_id,
        "correction_applied": False,
        "fallback_used": True,
        "fallback_reason": fallback_reason or "alignment_empty",
        "alignment_total_cues": 0,
        "alignment_anchored_cues": 0,
        "transcript_length": len(raw.text),
        "transcript_model": ctx.transcript_model,
        "transcript_provenance": raw.source,
        "timing_model": ctx.timing_model,
        "transcript_fallback": ctx.transcript_fallback,
    }
    return chunks, meta


def fallback_english(
    word_segments: list[dict],
    ctx: "StrategyContext",
    raw: Transcript,
    timing: TimingAnchors,
    fallback_reason: str | None,
    *,
    corrector_id: str = "none",
) -> tuple[list[list[dict]], dict]:
    """Build English fallback chunks one token per word, then finalize."""
    chunks = [
        [
            {
                "text": seg["word"],
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
                "is_punct": False,
                "whitespace": " ",
            }
        ]
        for seg in word_segments
        if seg.get("word")
    ]
    chunks = finalize_token_chunks(chunks, ctx)
    meta = {
        "mode": "fallback",
        "profile": ctx.profile.code,
        "text_source": "raw_transcript",
        "timing_source": timing.source,
        "timing_status": "fallback",
        "transcript_backend": "whisper",
        "timing_backend": "whisper",
        "correction_mode": corrector_id,
        "correction_applied": False,
        "fallback_used": True,
        "fallback_reason": fallback_reason or "alignment_empty",
        "alignment_total_cues": 0,
        "alignment_anchored_cues": 0,
        "transcript_length": len(raw.text),
    }
    return chunks, meta


__all__ = ["fallback_cjk", "fallback_english"]
