"""CJK staged-pipeline policy — wiring layer.

Selects and composes the concrete CJK stage implementations
(``sentences.punctuation``, ``align.char_level``,
``postprocess.display_width``, ``fallback``). The policy itself owns
the corrector seam, the benchmark short-circuit, per-stage hash
inputs, the legacy ``project_dir/cjk/`` mirror directory, and the
result-metadata summary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from subforge.config import (
    CHINESE_BENCHMARK_GAP_SECONDS,
    CHINESE_BENCHMARK_HARD_CHARS,
)
from subforge.nlp.cjk_corrector import Corrector
from subforge.pipeline.stages.align.char_level import (
    align_cjk,
    map_corrected_to_raw,
)
from subforge.pipeline.stages.cache import hash_inputs
from subforge.pipeline.stages.fallback import fallback_cjk
from subforge.pipeline.stages.models import (
    AlignedCue,
    Sentence,
    TimingAnchors,
    Transcript,
    build_split_inputs,
    word_segments_to_inputs,
)
from subforge.pipeline.stages.postprocess.display_width import (
    postprocess_cjk,
)
from subforge.pipeline.stages.sentences.punctuation import (
    split_by_punctuation,
)

if TYPE_CHECKING:
    from subforge.pipeline.strategies.base import StrategyContext

logger = logging.getLogger(__name__)


# Legacy mirror directory; runner copies stage artifacts here for
# backward compatibility. Canonical artifacts live under ``stages/``.
_LEGACY_CJK_DIRNAME = "cjk"

# Backward-compat re-export consumed via ``strategies.cjk`` and tests.
_map_corrected_to_raw = map_corrected_to_raw


@dataclass
class CjkPolicy:
    """CJK-specific stage wiring driven by :class:`StagedPipelineRunner`."""

    corrector: Corrector

    @property
    def corrector_id(self) -> str:
        return type(self.corrector).__name__

    @property
    def stage_label(self) -> str:
        return "CJK"

    def legacy_artifact_dir(self, ctx: "StrategyContext") -> Path | None:
        return ctx.project_dir / _LEGACY_CJK_DIRNAME

    def short_circuit(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> tuple[list[list[dict]], dict] | None:
        # Benchmark mode short-circuits the transcript-first flow but
        # still records final_cues.json with bypass metadata so a later
        # benchmark report can tell apart "ran the full pipeline" from
        # "ran the hard-cut path".
        if ctx.profile.code != "zh" or not ctx.chinese_benchmark:
            return None
        from subforge.nlp.chinese_benchmark import hard_cut_chinese_segments

        ctx.emit(
            "NLP",
            f"Chinese benchmark mode: hard-cut segmentation "
            f"(hard_chars={CHINESE_BENCHMARK_HARD_CHARS}, "
            f"gap={CHINESE_BENCHMARK_GAP_SECONDS}s)",
        )
        chunks = hard_cut_chinese_segments(
            word_segments,
            hard_chars=CHINESE_BENCHMARK_HARD_CHARS,
            gap_seconds=CHINESE_BENCHMARK_GAP_SECONDS,
        )
        meta = {
            "mode": "chinese_benchmark",
            "bypassed_stages": [
                "correction",
                "sentence_alignment",
                "cue_polishing",
            ],
            "fallback_used": False,
            "fallback_reason": None,
            "text_source": "raw_transcript",
            "timing_source": "word_segments",
            "timing_status": "word_timing",
        }
        return chunks, meta

    def stage_inputs_hash(
        self,
        schema_version: str,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> str:
        return hash_inputs(
            schema_version,
            json.dumps(word_segments, ensure_ascii=False, sort_keys=True),
            ctx.transcript_text or "",
            ctx.transcript_source or "",
        )

    def split_signature(self, ctx: "StrategyContext") -> str:
        return "".join(sorted(ctx.profile.sentence_end))

    def build_inputs(
        self,
        word_segments: list[dict],
        ctx: "StrategyContext",
    ) -> tuple[Transcript, TimingAnchors]:
        if ctx.transcript_text is not None:
            ctx.emit(
                self.stage_label,
                "Stage 1: split transcript "
                f"({ctx.transcript_source or 'transcript_only'} text + "
                f"{ctx.timing_backend or 'word_segments'} timing)",
            )
            return build_split_inputs(
                word_segments,
                transcript_text=ctx.transcript_text,
                transcript_source=ctx.transcript_source or "transcript_only",
                join_token=ctx.profile.join_token,
            )
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
        corrector_id = self.corrector_id
        ctx.emit(
            self.stage_label,
            f"Stage 2: correcting transcript ({corrector_id})",
        )
        applied = True
        try:
            corrected_text = self.corrector.correct(raw.text, ctx.profile.code)
            if not (isinstance(corrected_text, str) and corrected_text != ""):
                applied = False
                corrected_text = raw.text
        except Exception as exc:  # noqa: BLE001 — corrector boundary
            logger.warning(
                "Corrector %s raised: %s — using raw transcript",
                corrector_id,
                exc,
            )
            ctx.emit(
                self.stage_label,
                f"Stage 2: corrector failed ({type(exc).__name__}); "
                "using raw text",
            )
            corrected_text = raw.text
            applied = False

        source = "corrector" if applied else "asr_raw"
        return Transcript(text=corrected_text, source=source), applied

    def split_sentences(
        self,
        corrected: Transcript,
        ctx: "StrategyContext",
    ) -> list[Sentence]:
        ctx.emit(
            self.stage_label, "Stage 3: splitting transcript into sentences"
        )
        return split_by_punctuation(corrected.text, ctx.profile.sentence_end)

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
            "Stage 4: aligning sentences with timing anchors",
        )
        return align_cjk(sentences, raw, corrected, timing, correction_applied)

    def postprocess(
        self,
        cues: list[AlignedCue],
        ctx: "StrategyContext",
    ) -> tuple[list[list[dict]], dict]:
        ctx.emit("Postprocess", "CJK cue postprocess (width-aware)")
        return postprocess_cjk(cues, ctx.profile)

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
            "Transcript-first alignment produced no cues — "
            "falling back to word-segment punctuation split",
        )
        return fallback_cjk(word_segments, ctx, raw, timing, fallback_reason)

    def summarise_meta(
        self,
        cues: list[AlignedCue],
        raw: Transcript,
        timing: TimingAnchors,
        ctx: "StrategyContext",
    ) -> dict:
        fallback_cues = [c for c in cues if c.fallback_reason is not None]
        text_sources = {c.text_source for c in cues}
        text_source = (
            next(iter(text_sources)) if len(text_sources) == 1 else "mixed"
        )
        anchored = [c for c in cues if c.fallback_reason is None]
        avg_conf = (
            sum(c.confidence for c in anchored) / len(anchored)
            if anchored
            else 0.0
        )
        return {
            "mode": "transcript_first",
            "text_source": text_source,
            "timing_source": timing.source,
            "timing_status": timing.status,
            "fallback_used": bool(fallback_cues),
            "fallback_reason": (
                fallback_cues[0].fallback_reason if fallback_cues else None
            ),
            "transcript_backend": ctx.transcript_backend
            or ("sensevoice" if ctx.transcript_text is not None else "whisper"),
            "transcript_model": ctx.transcript_model,
            "transcript_provenance": raw.source,
            "transcript_length": len(raw.text),
            "timing_backend": ctx.timing_backend or "whisper",
            "timing_model": ctx.timing_model,
            "timing_text_length": len(timing.text),
            "transcript_fallback": ctx.transcript_fallback,
            "alignment_anchored_cues": len(anchored),
            "alignment_total_cues": len(cues),
            "alignment_avg_confidence": avg_conf,
        }


__all__ = [
    "CjkPolicy",
    "_map_corrected_to_raw",
]
