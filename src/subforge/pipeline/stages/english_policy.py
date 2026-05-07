"""English staged-pipeline policy and supporting helpers.

The policy provides English-specific behavior to the shared
:class:`subforge.pipeline.stages.runner.StagedPipelineRunner`:

* ASR ``word_segments`` input shaping via ``join_token`` (spaces for English),
* a no-op correction stage,
* spaCy sentence splitting with token-chunk preservation,
* word-level alignment via ``align_sentences_with_timestamps``,
* token-aware refine / split-long / merge-short postprocess,
* a conservative word-segment fallback when alignment fails.

The strategy module (``subforge.pipeline.strategies.english``) is now a
thin compatibility wrapper that constructs this policy and binds it to
the shared runner. This module contains the entire English staged
behavior so it can be reasoned about in one place.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from subforge.nlp.alignment import align_sentences_with_timestamps
from subforge.nlp.text_semantically import split_to_sentences
from subforge.pipeline.stages.cache import hash_inputs
from subforge.pipeline.stages.models import (
    AlignedCue,
    Sentence,
    TimingAnchors,
    TokenInterval,
    Transcript,
    word_segments_to_inputs,
)
from subforge.pipeline.stages.postprocess_helpers import finalize_token_chunks

if TYPE_CHECKING:
    from subforge.pipeline.strategies.base import StrategyContext

logger = logging.getLogger(__name__)


# ``split_to_sentences`` hardcodes its punctuation chunk limit. Mirror it
# in ``split_signature`` so the cache key invalidates if the constant
# ever moves into a profile/config.
_SPACY_PUNCT_LIMIT = 5


class EnglishPolicy:
    """English-specific stage behavior driven by the staged runner.

    The policy keeps two pieces of per-run instance state to bridge the
    existing English NLP primitives onto the staged ``Policy`` protocol
    without changing the protocol itself (see DD-1, DD-2 in the PR plan):

    * ``_word_segments`` — the original ASR word segments. Required by
      :func:`align_sentences_with_timestamps` but not part of the
      ``Policy.align`` signature, so we capture them up front.
    * ``_sentence_chunks`` — the spaCy token chunks produced during
      sentence splitting. Required by alignment to walk word-level
      timing onto each token; not derivable from the serialised
      :class:`Sentence` payload.

    Both are assigned per-run only and are never shared across threads
    because the strategy wrapper instantiates a fresh policy per call.
    """

    def __init__(self, word_segments: list[dict] | None = None):
        # Captured on construction so the value is always available even
        # when stage 1 is served from cache and ``build_inputs`` is
        # skipped (DD-1 spirit, adjusted for cache-hit safety).
        self._word_segments: list[dict] | None = word_segments
        self._sentence_chunks: list[list[dict]] | None = None

    # ------------------------------------------------------------------
    # Static descriptors
    # ------------------------------------------------------------------
    @property
    def corrector_id(self) -> str:
        # English currently has no transcript corrector. The staged
        # corrected_transcript artifact records ``"none"`` so future
        # English correction work can branch on a stable id.
        return "none"

    @property
    def stage_label(self) -> str:
        return "English"

    def legacy_artifact_dir(self, ctx: "StrategyContext") -> Path | None:
        # English has never written a language-specific artifact mirror
        # directory; canonical staged artifacts under ``stages/`` are the
        # only on-disk output.
        return None

    # ------------------------------------------------------------------
    # Short-circuit and per-run hashes
    # ------------------------------------------------------------------
    def short_circuit(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> tuple[list[list[dict]], dict] | None:
        return None

    def stage_inputs_hash(
        self,
        schema_version: str,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> str:
        # Capture word_segments here too so cached runs that skip
        # ``build_inputs`` can still reach them in ``align``.
        self._word_segments = word_segments
        return hash_inputs(
            schema_version,
            json.dumps(word_segments, ensure_ascii=False, sort_keys=True),
        )

    def split_signature(self, ctx: "StrategyContext") -> str:
        # spaCy splitting is parameterised by model name + punct chunk
        # limit; CJK's sentence-end character set is meaningless here.
        return f"spacy:{ctx.profile.spacy_model}:punct_limit={_SPACY_PUNCT_LIMIT}"

    # ------------------------------------------------------------------
    # Stage 1 — input shaping
    # ------------------------------------------------------------------
    def build_inputs(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> tuple[Transcript, TimingAnchors]:
        self._word_segments = word_segments
        ctx.emit(
            self.stage_label,
            "Stage 1: building raw transcript and timing anchors",
        )
        return word_segments_to_inputs(word_segments, ctx.profile.join_token)

    # ------------------------------------------------------------------
    # Stage 2 — correction (no-op for English)
    # ------------------------------------------------------------------
    def correct(
        self,
        raw: Transcript,
        ctx: "StrategyContext",
    ) -> tuple[Transcript, bool]:
        ctx.emit(
            self.stage_label,
            f"Stage 2: corrected transcript ({self.corrector_id})",
        )
        return Transcript(text=raw.text, source=raw.source), False

    # ------------------------------------------------------------------
    # Stage 3 — sentence split (spaCy)
    # ------------------------------------------------------------------
    def split_sentences(
        self,
        corrected: Transcript,
        ctx: "StrategyContext",
    ) -> list[Sentence]:
        ctx.emit(
            self.stage_label,
            "Stage 3: splitting into sentences (spaCy)",
        )
        token_chunks = split_to_sentences(corrected.text)
        self._sentence_chunks = token_chunks
        return _sentences_from_token_chunks(token_chunks, corrected.text)

    # ------------------------------------------------------------------
    # Stage 4 — alignment (word-level timing)
    # ------------------------------------------------------------------
    def align(
        self,
        sentences: list[Sentence],
        raw: Transcript,
        corrected: Transcript,
        timing: TimingAnchors,
        correction_applied: bool,
        ctx: "StrategyContext",
    ) -> tuple[list[AlignedCue], str | None]:
        ctx.emit(
            self.stage_label,
            "Stage 4: aligning sentences with timestamps",
        )

        word_segments = self._word_segments
        if word_segments is None:
            # Should never happen — the strategy wrapper always seeds
            # ``_word_segments`` and ``stage_inputs_hash`` re-seeds it.
            logger.warning(
                "English align called without word_segments; falling back"
            )
            return [], "missing_word_segments"

        sentence_chunks = self._sentence_chunks
        if sentence_chunks is None:
            # Sentence stage was served from cache: re-run the
            # deterministic spaCy split to recover the original token
            # chunks alignment needs. The cached Sentence objects do not
            # carry per-token data.
            sentence_chunks = split_to_sentences(corrected.text)

        try:
            aligned_chunks = align_sentences_with_timestamps(
                word_segments, sentence_chunks
            )
        except Exception as exc:  # noqa: BLE001 — alignment boundary
            logger.warning("English alignment failed: %s", exc)
            ctx.emit(
                "Align",
                f"English alignment failed ({type(exc).__name__})",
            )
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

    # ------------------------------------------------------------------
    # Stage 5 — postprocess (token-aware, shared with CJK fallback)
    # ------------------------------------------------------------------
    def postprocess(
        self,
        cues: list[AlignedCue],
        ctx: "StrategyContext",
    ) -> tuple[list[list[dict]], dict]:
        ctx.emit("Postprocess", "English token-aware postprocess")
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

    def fallback(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
        raw: Transcript,
        timing: TimingAnchors,
        fallback_reason: str | None,
    ) -> tuple[list[list[dict]], dict]:
        ctx.emit(
            "Align",
            "English alignment produced no cues — falling back to "
            "word-segment timing",
        )
        # One token per word; let the shared finalize pass merge them
        # into reasonable display chunks.
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
            "correction_mode": self.corrector_id,
            "correction_applied": False,
            "fallback_used": True,
            "fallback_reason": fallback_reason or "alignment_empty",
            "transcript_length": len(raw.text),
        }
        return chunks, meta

    # ------------------------------------------------------------------
    # Result metadata
    # ------------------------------------------------------------------
    def summarise_meta(
        self,
        cues: list[AlignedCue],
        raw: Transcript,
        timing: TimingAnchors,
        ctx: "StrategyContext",
    ) -> dict:
        fallback_cues = [c for c in cues if c.fallback_reason is not None]
        anchored = [c for c in cues if c.fallback_reason is None]
        avg_conf = (
            sum(c.confidence for c in anchored) / len(anchored)
            if anchored
            else 0.0
        )
        return {
            "mode": "english_staged",
            "profile": ctx.profile.code,
            "text_source": "raw",
            "timing_source": timing.source,
            "timing_status": timing.status,
            "correction_mode": self.corrector_id,
            "correction_applied": False,
            "fallback_used": bool(fallback_cues),
            "fallback_reason": (
                fallback_cues[0].fallback_reason if fallback_cues else None
            ),
            "alignment_total_cues": len(cues),
            "alignment_anchored_cues": len(anchored),
            "alignment_avg_confidence": avg_conf,
            "transcript_length": len(raw.text),
        }


# ---------------------------------------------------------------------------
# Adapters between AlignedCue tokens and the legacy token-chunk dict shape
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sentence reconstruction
# ---------------------------------------------------------------------------


def _sentences_from_token_chunks(
    token_chunks: list[list[dict]],
    transcript_text: str,
) -> list[Sentence]:
    """Materialise :class:`Sentence` objects from spaCy token chunks.

    The spaCy split returns ``list[list[token_dict]]`` with ``text`` and
    ``whitespace`` per token. We reconstruct each sentence text by
    concatenating those, then locate the substring in the transcript by
    forward scanning so duplicate sentences resolve to distinct offsets.
    Failure to find the reconstructed sentence indicates a bug in the
    splitter's invariants and is raised loudly rather than silently
    producing wrong offsets.
    """
    sentences: list[Sentence] = []
    offset = 0
    for chunk in token_chunks:
        sent_text = "".join(t["text"] + t["whitespace"] for t in chunk)
        idx = transcript_text.find(sent_text, offset)
        if idx < 0:
            raise ValueError(
                "Reconstructed sentence not found in transcript at "
                f"offset {offset}: {sent_text!r:.80}"
            )
        sentences.append(
            Sentence(
                text=sent_text,
                char_start=idx,
                char_end=idx + len(sent_text),
            )
        )
        offset = idx + len(sent_text)
    return sentences


__all__ = [
    "EnglishPolicy",
    "aligned_cues_to_token_chunks",
]
