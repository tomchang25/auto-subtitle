"""CJK staged-pipeline policy and supporting helpers.

The policy provides CJK-specific behavior to the shared
:class:`subforge.pipeline.stages.runner.StagedPipelineRunner`:

* split SenseVoice transcript / Whisper timing input shaping,
* a corrector seam with no-op default,
* punctuation-driven sentence splitting,
* char-level alignment with a display-text fallback when correction is
  rejected or anchors are missing,
* display-width-aware postprocess,
* benchmark mode short-circuit (recorded with bypass metadata),
* legacy ``project_dir/cjk/`` artifact mirror directory.

The strategy module (``subforge.pipeline.strategies.cjk``) is now a thin
compatibility wrapper that constructs this policy and binds it to the
shared runner.
"""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from subforge.config import (
    CHINESE_BENCHMARK_GAP_SECONDS,
    CHINESE_BENCHMARK_HARD_CHARS,
    CJK_POSTPROCESS_MAX_DURATION,
    CJK_POSTPROCESS_MAX_WIDTH,
    CJK_POSTPROCESS_MERGE_MAX_DURATION,
    CJK_POSTPROCESS_MERGE_MAX_GAP,
    CJK_POSTPROCESS_MERGE_MAX_WIDTH,
    CJK_POSTPROCESS_MIN_DURATION,
    CJK_POSTPROCESS_SHORT_CUE_WIDTH,
)
from subforge.nlp.cjk_corrector import Corrector
from subforge.nlp.cjk_postprocess import (
    PostprocessConfig,
    postprocess_cjk_cues,
    postprocess_cues_to_writer_chunks,
)
from subforge.nlp.text_semantically import split_word_segments_by_punctuation
from subforge.pipeline.stages.cache import hash_inputs
from subforge.pipeline.stages.models import (
    AlignedCue,
    Sentence,
    TimingAnchors,
    Transcript,
    build_split_inputs,
    word_segments_to_inputs,
)
from subforge.pipeline.stages.postprocess_helpers import finalize_token_chunks

if TYPE_CHECKING:
    from subforge.pipeline.strategies.base import StrategyContext

logger = logging.getLogger(__name__)


# Legacy artifact directory name. The runner mirrors stage artifacts
# here for backward compatibility with downstream tools and tests.
# Canonical artifacts live under ``project_dir/stages/`` and are the
# runner's only cache source.
_LEGACY_CJK_DIRNAME = "cjk"


@dataclass
class CjkPolicy:
    """CJK-specific stage behavior driven by the staged runner.

    The policy holds only the CJK-shaped pieces — corrector seam,
    transcript/timing separation, no-op default correction, punctuation
    sentence splitting, char-level alignment, display-width-aware
    postprocess, fallback path and benchmark short-circuit. The runner
    handles caching, force-clear, schema versioning, and legacy
    mirroring.
    """

    corrector: Corrector

    @property
    def corrector_id(self) -> str:
        return type(self.corrector).__name__

    @property
    def stage_label(self) -> str:
        return "CJK"

    def legacy_artifact_dir(self, ctx: "StrategyContext") -> Path | None:
        return ctx.project_dir / _LEGACY_CJK_DIRNAME

    # ------------------------------------------------------------------
    # Benchmark short-circuit
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Per-stage hash inputs
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Stage 1 — input shaping
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Stage 2 — correction
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Stage 3 — sentence split
    # ------------------------------------------------------------------
    def split_sentences(
        self,
        corrected: Transcript,
        ctx: "StrategyContext",
    ) -> list[Sentence]:
        ctx.emit(
            self.stage_label, "Stage 3: splitting transcript into sentences"
        )
        sentence_end = ctx.profile.sentence_end
        sentences: list[Sentence] = []
        text = corrected.text
        start = 0
        for i, ch in enumerate(text):
            if ch in sentence_end:
                sentences.append(Sentence(text[start : i + 1], start, i + 1))
                start = i + 1
        if start < len(text):
            sentences.append(Sentence(text[start:], start, len(text)))
        return sentences

    # ------------------------------------------------------------------
    # Stage 4 — alignment
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
            "Stage 4: aligning sentences with timing anchors",
        )
        try:
            corrected_to_raw = _map_corrected_to_raw(corrected.text, raw.text)
            corrected_to_timing = _map_corrected_to_raw(
                corrected.text, timing.text
            )
        except Exception as exc:  # noqa: BLE001 — alignment boundary
            logger.warning("Char-level alignment failed: %s", exc)
            ctx.emit(
                "Align",
                f"Char-level alignment failed ({type(exc).__name__}); "
                "falling back",
            )
            return [], "mapping_failed"

        cues: list[AlignedCue] = []
        any_anchored = False
        for sent in sentences:
            cue = _build_cue(
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

    # ------------------------------------------------------------------
    # Stage 5 — postprocess (display-width-aware) and fallback
    # ------------------------------------------------------------------
    def postprocess(
        self,
        cues: list[AlignedCue],
        ctx: "StrategyContext",
    ) -> tuple[list[list[dict]], dict]:
        ctx.emit("Postprocess", "CJK cue postprocess (width-aware)")
        cfg = PostprocessConfig(
            max_display_width=CJK_POSTPROCESS_MAX_WIDTH,
            min_duration=CJK_POSTPROCESS_MIN_DURATION,
            max_duration=CJK_POSTPROCESS_MAX_DURATION,
            merge_max_width=CJK_POSTPROCESS_MERGE_MAX_WIDTH,
            merge_max_duration=CJK_POSTPROCESS_MERGE_MAX_DURATION,
            merge_max_gap=CJK_POSTPROCESS_MERGE_MAX_GAP,
            short_cue_width=CJK_POSTPROCESS_SHORT_CUE_WIDTH,
        )
        post_cues, post_diag = postprocess_cjk_cues(cues, ctx.profile, cfg)
        chunks = postprocess_cues_to_writer_chunks(post_cues, ctx.profile)
        return chunks, post_diag

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
        chunks = split_word_segments_by_punctuation(word_segments, ctx.profile)
        chunks = finalize_token_chunks(chunks, ctx)
        meta = {
            "mode": "fallback",
            "fallback_used": True,
            "fallback_reason": fallback_reason or "alignment_empty",
            "text_source": "raw_transcript",
            "timing_source": timing.source,
            "timing_status": "fallback",
            "transcript_backend": ctx.transcript_backend
            or ("sensevoice" if ctx.transcript_text is not None else "whisper"),
            "transcript_model": ctx.transcript_model,
            "transcript_length": len(raw.text),
            "transcript_provenance": raw.source,
            "timing_backend": ctx.timing_backend or "whisper",
            "timing_model": ctx.timing_model,
            "transcript_fallback": ctx.transcript_fallback,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cue(
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


def _map_corrected_to_raw(corrected: str, raw: str) -> list[int | None]:
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


__all__ = [
    "CjkPolicy",
]
