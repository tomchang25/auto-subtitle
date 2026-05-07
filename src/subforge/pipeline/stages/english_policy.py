"""English staged-pipeline policy — wiring layer.

Selects and composes the concrete English stage implementations
(``sentences.spacy_splitter``, ``align.word_level``,
``postprocess.word_count``, ``fallback``). The policy itself owns
the no-op corrector seam, per-stage hash inputs, and the
result-metadata summary, and bridges per-run state
(``_word_segments``, ``_sentence_chunks``) across cache hits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

# Re-imported so existing tests that monkeypatch
# ``english_policy.align_sentences_with_timestamps`` continue to take
# effect. ``align.word_level`` resolves the alignment function via this
# module's attribute at call time.
from subforge.nlp.alignment import align_sentences_with_timestamps
from subforge.pipeline.stages.align.word_level import align_english
from subforge.pipeline.stages.cache import hash_inputs
from subforge.pipeline.stages.fallback import fallback_english
from subforge.pipeline.stages.models import (
    AlignedCue,
    Sentence,
    TimingAnchors,
    Transcript,
    word_segments_to_inputs,
)
from subforge.pipeline.stages.postprocess.word_count import (
    aligned_cues_to_token_chunks,
    postprocess_english,
)
from subforge.pipeline.stages.sentences.spacy_splitter import (
    _SPACY_PUNCT_LIMIT,
    split_spacy,
)

if TYPE_CHECKING:
    from subforge.pipeline.strategies.base import StrategyContext

logger = logging.getLogger(__name__)


class EnglishPolicy:
    """English-specific stage wiring driven by :class:`StagedPipelineRunner`.

    Keeps two pieces of per-run instance state — ``_word_segments``
    (the original ASR words, needed by alignment) and
    ``_sentence_chunks`` (the spaCy token chunks, needed by alignment
    to walk timings onto tokens) — so that the cached cache-hit path
    can still feed alignment when ``build_inputs`` /
    ``split_sentences`` are skipped. The strategy wrapper instantiates
    a fresh policy per call so this state never leaks across runs.
    """

    def __init__(self, word_segments: list[dict] | None = None):
        # Captured on construction so the value is always available even
        # when stage 1 is served from cache and ``build_inputs`` is skipped.
        self._word_segments: list[dict] | None = word_segments
        self._sentence_chunks: list[list[dict]] | None = None

    @property
    def corrector_id(self) -> str:
        # No English corrector yet; the artifact records "none" so a
        # future corrector can branch on a stable id.
        return "none"

    @property
    def stage_label(self) -> str:
        return "English"

    def legacy_artifact_dir(self, ctx: "StrategyContext") -> Path | None:
        return None

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
        # spaCy splitting is parameterised by model name + punct chunk limit.
        return f"spacy:{ctx.profile.spacy_model}:punct_limit={_SPACY_PUNCT_LIMIT}"

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

    def split_sentences(
        self,
        corrected: Transcript,
        ctx: "StrategyContext",
    ) -> list[Sentence]:
        ctx.emit(
            self.stage_label,
            "Stage 3: splitting into sentences (spaCy)",
        )
        sentences, token_chunks = split_spacy(corrected.text)
        self._sentence_chunks = token_chunks
        return sentences

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
        return align_english(
            sentences,
            self._word_segments,
            self._sentence_chunks,
            corrected.text,
            timing,
        )

    def postprocess(
        self,
        cues: list[AlignedCue],
        ctx: "StrategyContext",
    ) -> tuple[list[list[dict]], dict]:
        ctx.emit("Postprocess", "English token-aware postprocess")
        return postprocess_english(cues, ctx)

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
        return fallback_english(
            word_segments,
            ctx,
            raw,
            timing,
            fallback_reason,
            corrector_id=self.corrector_id,
        )

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


__all__ = [
    "EnglishPolicy",
    "aligned_cues_to_token_chunks",
    "align_sentences_with_timestamps",
]
