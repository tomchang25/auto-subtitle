"""Word-level alignment for the English staged pipeline.

The English pipeline aligns spaCy sentence chunks to ASR
``word_segments`` via :func:`align_sentences_with_timestamps`. Each
aligned chunk produces an :class:`AlignedCue` whose ``tokens`` list
carries the per-word timing onward to the postprocess stage.

The alignment function is resolved via the
:mod:`subforge.pipeline.stages.english_policy` module so existing test
monkeypatches on that module's ``align_sentences_with_timestamps``
attribute continue to take effect after the move.
"""

from __future__ import annotations

import logging

from subforge.nlp.text_semantically import split_to_sentences
from subforge.pipeline.stages.models import (
    AlignedCue,
    Sentence,
    TimingAnchors,
    TokenInterval,
)

logger = logging.getLogger(__name__)


def align_english(
    sentences: list[Sentence],
    word_segments: list[dict] | None,
    sentence_chunks: list[list[dict]] | None,
    corrected_text: str,
    timing: TimingAnchors,
) -> tuple[list[AlignedCue], str | None]:
    """Word-level alignment of spaCy sentences to ASR word timings.

    Returns ``(cues, None)`` on success. Returns
    ``([], "missing_word_segments")`` if no word segments were captured
    by the policy. Returns ``([], f"alignment_failed:{ExcName}")`` if
    the alignment call raises.

    The cache-hit recovery path passes ``sentence_chunks=None`` when the
    sentence stage was served from cache; we re-derive the chunks via
    :func:`split_to_sentences` because the cached :class:`Sentence`
    payload does not carry per-token data.
    """
    if word_segments is None:
        # Should never happen — the strategy wrapper always seeds the
        # policy's ``_word_segments`` and ``stage_inputs_hash`` re-seeds it.
        logger.warning(
            "English align called without word_segments; falling back"
        )
        return [], "missing_word_segments"

    if sentence_chunks is None:
        sentence_chunks = split_to_sentences(corrected_text)

    # Resolve the alignment function via the english_policy module so
    # existing tests that monkeypatch ``english_policy.align_sentences_with_timestamps``
    # continue to take effect after this extraction. The lazy import
    # avoids a circular import at module load time.
    from subforge.pipeline.stages import english_policy as _ep

    try:
        aligned_chunks = _ep.align_sentences_with_timestamps(
            word_segments, sentence_chunks
        )
    except Exception as exc:  # noqa: BLE001 — alignment boundary
        logger.warning("English alignment failed: %s", exc)
        return [], f"alignment_failed:{type(exc).__name__}"

    cues: list[AlignedCue] = []
    for sent, chunk in zip(sentences, aligned_chunks):
        tokens = [
            TokenInterval(
                text=tok["text"],
                start=float(tok["start"]),
                end=float(tok["end"]),
                is_punct=bool(tok.get("is_punct", False)),
                whitespace=tok.get("whitespace", ""),
                source="asr_word",
            )
            for tok in chunk
        ]
        cue_start = tokens[0].start if tokens else 0.0
        cue_end = tokens[-1].end if tokens else 0.0
        cues.append(
            AlignedCue(
                raw_text=sent.text,
                corrected_text=sent.text,
                display_text=sent.text,
                start=cue_start,
                end=cue_end,
                confidence=1.0,
                fallback_reason=None,
                text_source="raw",
                timing_source=timing.source,
                timing_status=timing.status,
                tokens=tokens,
            )
        )
    return cues, None


__all__ = ["align_english"]
