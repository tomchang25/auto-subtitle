"""Transcript-first CJK subtitle pipeline.

Unlike the English path — which relies on ASR word timestamps to drive
sentence segmentation through spaCy — the CJK strategy treats text quality
and timing as separate first-class concerns. It runs five named stages,
each of which persists its artifact under ``project_dir/cjk/`` so a rerun
can resume from the latest valid stage:

  1. ``raw_transcript.json``       — :class:`CjkTranscript` rebuilt from the
                                    ASR output (text only, with provenance).
  1b. ``timing_anchors.json``      — :class:`CjkTimingAnchors`, the per-char
                                    timing track parallel to the raw
                                    transcript. Kept separate so a future
                                    SenseVoice / Whisper split can populate
                                    them from different sources.
  2. ``corrected_transcript.json`` — output of the pluggable corrector
                                    (defaults to a no-op).
  3. ``sentences.json``            — sentence list (with offsets) split from
                                    text only, independent of timing.
  4. ``alignment.json``            — :class:`CjkAlignedCue` list: corrected
                                    text + timing interval + display text +
                                    confidence and fallback metadata.
  5. ``final_cues.json``           — writer-compatible chunks after timing
                                    refinement, length splitting and
                                    short-segment merging. Persisted before
                                    SRT writing so debugging can distinguish
                                    alignment quality from SRT formatting.

Each stage degrades gracefully: a failed corrector falls back to raw text,
a failed alignment falls back to the legacy ``split_word_segments_by_
punctuation`` path, and timing falls back to estimated intervals when char
mapping yields nothing. Fallbacks are recorded in the artifact metadata,
not just in logs, so later benchmarking can compare runs without re-reading
the source media.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
from pathlib import Path

from subforge.config import (
    BREATH_GAP,
    CHINESE_BENCHMARK_GAP_SECONDS,
    CHINESE_BENCHMARK_HARD_CHARS,
    MAX_GAP,
    MERGE_MAX_DURATION,
    MERGE_MAX_GAP,
    MIN_DURATION,
    MIN_WORDS_FOR_BREATH_SPLIT,
    SEG_PAUSE_THRESHOLD,
)
from subforge.nlp.alignment import refine_sentences_by_timing
from subforge.nlp.cjk_corrector import Corrector, NoOpCorrector
from subforge.nlp.segmentation import (
    merge_short_segments,
    split_long_sentences_by_length,
)
from subforge.nlp.text_semantically import split_word_segments_by_punctuation
from subforge.pipeline.strategies.base import (
    LanguagePipelineStrategy,
    StrategyContext,
)
from subforge.pipeline.strategies.cjk_models import (
    CjkAlignedCue,
    CjkPipelineResult,
    CjkSentence,
    CjkTimingAnchors,
    CjkTranscript,
    build_split_cjk_inputs,
    cjk_cues_to_writer_chunks,
    word_segments_to_cjk_inputs,
)

logger = logging.getLogger(__name__)


# Bumped whenever the artifact schema changes so old caches don't poison
# new pipeline runs.
_STAGE_SCHEMA_VERSION = "v3"

_STAGE_FILES = (
    "raw_transcript.json",
    "timing_anchors.json",
    "corrected_transcript.json",
    "sentences.json",
    "alignment.json",
    "final_cues.json",
)


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _load_stage(path: Path, expected_hash: str) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Stage cache unreadable at %s: %s", path, exc)
        return None
    if data.get("input_hash") != expected_hash:
        return None
    return data.get("data")


def _save_stage(path: Path, input_hash: str, data) -> None:
    payload = {"input_hash": input_hash, "data": data}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class CjkPipelineStrategy(LanguagePipelineStrategy):
    """Transcript-first subtitle strategy for CJK languages."""

    def __init__(self, corrector: Corrector | None = None):
        self.corrector = corrector or NoOpCorrector()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        word_segments: list[dict],
        ctx: StrategyContext,
    ) -> list[list[dict]]:
        cjk_dir = ctx.project_dir / "cjk"
        cjk_dir.mkdir(parents=True, exist_ok=True)

        if ctx.force:
            for name in _STAGE_FILES:
                stale = cjk_dir / name
                if stale.exists():
                    stale.unlink()

        # Benchmark mode short-circuits the transcript-first flow but still
        # lives inside the CJK strategy so the orchestrator stays language-
        # agnostic. The bypass is recorded in final_cues.json so later
        # benchmark reports can tell apart "ran the full pipeline" from
        # "ran the hard-cut path".
        if ctx.profile.code == "zh" and ctx.chinese_benchmark:
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
            self._save_final_cues(
                cjk_dir,
                chunks,
                meta={
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
                },
            )
            return chunks

        ws_hash = _hash(
            _STAGE_SCHEMA_VERSION,
            json.dumps(word_segments, ensure_ascii=False, sort_keys=True),
            (ctx.transcript_text or ""),
            (ctx.transcript_source or ""),
        )

        # Stage 1 — raw transcript + timing anchors (separate artifacts).
        raw_transcript, timing = self._stage_inputs(
            word_segments, ctx, cjk_dir, ws_hash
        )
        ctx.check_cancel()

        # Stage 2 — corrected transcript.
        corrected_transcript, correction_applied = self._stage_correct(
            raw_transcript, ctx, cjk_dir, ws_hash
        )
        ctx.check_cancel()

        # Stage 3 — sentence list (text only).
        sentences = self._stage_split_sentences(
            corrected_transcript, ctx, cjk_dir, ws_hash
        )
        ctx.check_cancel()

        # Stage 4 — re-attach timing → list[CjkAlignedCue].
        cues, fallback_reason = self._stage_align(
            sentences,
            raw_transcript,
            corrected_transcript,
            timing,
            correction_applied,
            ctx,
            cjk_dir,
            ws_hash,
        )
        ctx.check_cancel()

        if not cues:
            ctx.emit(
                "Align",
                "Transcript-first alignment produced no cues — "
                "falling back to word-segment punctuation split",
            )
            chunks = split_word_segments_by_punctuation(word_segments, ctx.profile)
            chunks = self._finalize(chunks, ctx)
            self._save_final_cues(
                cjk_dir,
                chunks,
                meta={
                    "mode": "fallback",
                    "fallback_used": True,
                    "fallback_reason": fallback_reason or "alignment_empty",
                    "text_source": "raw_transcript",
                    "timing_source": timing.source,
                    "timing_status": "fallback",
                    "transcript_backend": ctx.transcript_backend
                    or ("sensevoice" if ctx.transcript_text is not None else "whisper"),
                    "transcript_model": ctx.transcript_model,
                    "transcript_length": len(raw_transcript.text),
                    "transcript_provenance": raw_transcript.source,
                    "timing_backend": ctx.timing_backend or "whisper",
                    "timing_model": ctx.timing_model,
                    "transcript_fallback": ctx.transcript_fallback,
                },
            )
            return chunks

        # Stage 4.5 — convert cues into the writer's chunk format. The CJK
        # strategy owns this conversion so the rest of the pipeline never has
        # to manufacture English-style word tokens for CJK text.
        chunks = cjk_cues_to_writer_chunks(cues, ctx.profile)

        # Stage 5 — shared timing refinement / length split / short merge.
        chunks = self._finalize(chunks, ctx)

        result_meta = self._summarise_result(cues, timing, raw_transcript, ctx)
        self._save_final_cues(cjk_dir, chunks, meta=result_meta)
        return chunks

    # ------------------------------------------------------------------
    # Stage 1 — raw transcript + timing anchors via the adapter
    # ------------------------------------------------------------------
    def _stage_inputs(
        self,
        word_segments: list[dict],
        ctx: StrategyContext,
        cjk_dir: Path,
        ws_hash: str,
    ) -> tuple[CjkTranscript, CjkTimingAnchors]:
        raw_path = cjk_dir / "raw_transcript.json"
        timing_path = cjk_dir / "timing_anchors.json"

        cached_raw = None if ctx.force else _load_stage(raw_path, ws_hash)
        cached_timing = None if ctx.force else _load_stage(timing_path, ws_hash)

        if cached_raw is not None and cached_timing is not None:
            ctx.emit("CJK", "Stage 1: raw transcript + timing anchors (cached)")
            return (
                CjkTranscript.from_dict(cached_raw),
                CjkTimingAnchors.from_dict(cached_timing),
            )

        if ctx.transcript_text is not None:
            ctx.emit(
                "CJK",
                "Stage 1: split transcript "
                f"({ctx.transcript_source or 'transcript_only'} text + "
                f"{ctx.timing_backend or 'word_segments'} timing)",
            )
            transcript, timing = build_split_cjk_inputs(
                word_segments,
                transcript_text=ctx.transcript_text,
                transcript_source=ctx.transcript_source or "transcript_only",
                join_token=ctx.profile.join_token,
            )
        else:
            ctx.emit("CJK", "Stage 1: building raw transcript and timing anchors")
            transcript, timing = word_segments_to_cjk_inputs(
                word_segments, ctx.profile.join_token
            )
        _save_stage(raw_path, ws_hash, transcript.to_dict())
        _save_stage(timing_path, ws_hash, timing.to_dict())
        return transcript, timing

    # ------------------------------------------------------------------
    # Stage 2 — corrector
    # ------------------------------------------------------------------
    def _stage_correct(
        self,
        raw: CjkTranscript,
        ctx: StrategyContext,
        cjk_dir: Path,
        ws_hash: str,
    ) -> tuple[CjkTranscript, bool]:
        path = cjk_dir / "corrected_transcript.json"
        corrector_id = type(self.corrector).__name__
        input_hash = _hash(ws_hash, raw.text, corrector_id)
        cached = None if ctx.force else _load_stage(path, input_hash)
        if cached is not None:
            ctx.emit("CJK", f"Stage 2: corrected transcript (cached, {corrector_id})")
            return (
                CjkTranscript(text=cached["text"], source=cached.get("source", "corrector")),
                bool(cached.get("applied", False)),
            )

        ctx.emit("CJK", f"Stage 2: correcting transcript ({corrector_id})")
        applied = True
        try:
            corrected_text = self.corrector.correct(raw.text, ctx.profile.code)
            if not (isinstance(corrected_text, str) and corrected_text != ""):
                applied = False
                corrected_text = raw.text
        except Exception as exc:  # noqa: BLE001 — corrector boundary
            logger.warning(
                "Corrector %s raised: %s — using raw transcript", corrector_id, exc
            )
            ctx.emit(
                "CJK",
                f"Stage 2: corrector failed ({type(exc).__name__}); using raw text",
            )
            corrected_text = raw.text
            applied = False

        source = "corrector" if applied else "asr_raw"
        corrected = CjkTranscript(text=corrected_text, source=source)
        data = {
            "text": corrected.text,
            "source": corrected.source,
            "corrector": corrector_id,
            "applied": applied,
        }
        _save_stage(path, input_hash, data)
        return corrected, applied

    # ------------------------------------------------------------------
    # Stage 3 — sentence split (text only)
    # ------------------------------------------------------------------
    def _stage_split_sentences(
        self,
        transcript: CjkTranscript,
        ctx: StrategyContext,
        cjk_dir: Path,
        ws_hash: str,
    ) -> list[CjkSentence]:
        path = cjk_dir / "sentences.json"
        input_hash = _hash(
            ws_hash,
            transcript.text,
            "".join(sorted(ctx.profile.sentence_end)),
        )
        cached = None if ctx.force else _load_stage(path, input_hash)
        if cached is not None:
            ctx.emit("CJK", f"Stage 3: sentences (cached, {len(cached)})")
            return [CjkSentence.from_dict(s) for s in cached]

        ctx.emit("CJK", "Stage 3: splitting transcript into sentences")
        sentence_end = ctx.profile.sentence_end
        sentences: list[CjkSentence] = []
        text = transcript.text
        start = 0
        for i, ch in enumerate(text):
            if ch in sentence_end:
                sentences.append(CjkSentence(text[start : i + 1], start, i + 1))
                start = i + 1
        if start < len(text):
            sentences.append(CjkSentence(text[start:], start, len(text)))

        _save_stage(path, input_hash, [s.to_dict() for s in sentences])
        return sentences

    # ------------------------------------------------------------------
    # Stage 4 — alignment
    # ------------------------------------------------------------------
    def _stage_align(
        self,
        sentences: list[CjkSentence],
        raw: CjkTranscript,
        corrected: CjkTranscript,
        timing: CjkTimingAnchors,
        correction_applied: bool,
        ctx: StrategyContext,
        cjk_dir: Path,
        ws_hash: str,
    ) -> tuple[list[CjkAlignedCue], str | None]:
        path = cjk_dir / "alignment.json"
        input_hash = _hash(
            ws_hash,
            raw.text,
            corrected.text,
            json.dumps([s.to_dict() for s in sentences], ensure_ascii=False),
            timing.source,
            timing.status,
        )
        cached = None if ctx.force else _load_stage(path, input_hash)
        if cached is not None:
            ctx.emit("CJK", f"Stage 4: alignment (cached, {len(cached)} cues)")
            return [CjkAlignedCue.from_dict(c) for c in cached], None

        ctx.emit("CJK", "Stage 4: aligning sentences with timing anchors")

        try:
            corrected_to_raw = _map_corrected_to_raw(corrected.text, raw.text)
            corrected_to_timing = _map_corrected_to_raw(
                corrected.text, timing.text
            )
        except Exception as exc:  # noqa: BLE001 — alignment boundary
            logger.warning("Char-level alignment failed: %s", exc)
            ctx.emit(
                "Align",
                f"Char-level alignment failed ({type(exc).__name__}); falling back",
            )
            return [], "mapping_failed"

        cues: list[CjkAlignedCue] = []
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

        _save_stage(path, input_hash, [c.to_dict() for c in cues])
        return cues, None

    # ------------------------------------------------------------------
    # Stage 5 — shared refine / split / merge
    # ------------------------------------------------------------------
    def _finalize(
        self,
        chunks: list[list[dict]],
        ctx: StrategyContext,
    ) -> list[list[dict]]:
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

    # ------------------------------------------------------------------
    # Result persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _summarise_result(
        cues: list[CjkAlignedCue],
        timing: CjkTimingAnchors,
        raw_transcript: CjkTranscript,
        ctx: StrategyContext,
    ) -> dict:
        fallback_cues = [c for c in cues if c.fallback_reason is not None]
        text_sources = {c.text_source for c in cues}
        if len(text_sources) == 1:
            text_source = next(iter(text_sources))
        else:
            text_source = "mixed"
        anchored = [c for c in cues if c.fallback_reason is None]
        avg_conf = (
            sum(c.confidence for c in anchored) / len(anchored)
            if anchored
            else 0.0
        )
        result = CjkPipelineResult(
            cues=cues,
            text_source=text_source,
            timing_source=timing.source,
            timing_status=timing.status,
            fallback_used=bool(fallback_cues),
            fallback_reason=(
                fallback_cues[0].fallback_reason if fallback_cues else None
            ),
        )
        meta = {
            "mode": "transcript_first",
            "text_source": result.text_source,
            "timing_source": result.timing_source,
            "timing_status": result.timing_status,
            "fallback_used": result.fallback_used,
            "fallback_reason": result.fallback_reason,
            "transcript_backend": ctx.transcript_backend
            or ("sensevoice" if ctx.transcript_text is not None else "whisper"),
            "transcript_model": ctx.transcript_model,
            "transcript_provenance": raw_transcript.source,
            "transcript_length": len(raw_transcript.text),
            "timing_backend": ctx.timing_backend or "whisper",
            "timing_model": ctx.timing_model,
            "timing_text_length": len(timing.text),
            "transcript_fallback": ctx.transcript_fallback,
            "alignment_anchored_cues": len(anchored),
            "alignment_total_cues": len(cues),
            "alignment_avg_confidence": avg_conf,
        }
        return meta

    @staticmethod
    def _save_final_cues(
        cjk_dir: Path,
        chunks: list[list[dict]],
        *,
        meta: dict,
    ) -> None:
        path = cjk_dir / "final_cues.json"
        payload = {
            "meta": meta,
            "chunks": chunks,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cue(
    sent: CjkSentence,
    *,
    raw: CjkTranscript,
    corrected: CjkTranscript,
    timing: CjkTimingAnchors,
    corrected_to_raw: list[int | None],
    corrected_to_timing: list[int | None],
    correction_applied: bool,
) -> CjkAlignedCue:
    """Resolve one sentence into a fully-decorated :class:`CjkAlignedCue`."""
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
        # Corrector was rejected, or the cue lost its anchor — prefer the raw
        # ASR text so the user still gets something to read.
        display_text = raw_text if raw_text else sent.text
        text_source = "raw"

    return CjkAlignedCue(
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
    """Map each char index in *corrected* to the closest matching index in *raw*.

    Uses :class:`difflib.SequenceMatcher` so insertions/deletions/replacements
    introduced by the corrector still leave aligned regions intact. Positions
    in corrected text with no raw counterpart map to ``None``.
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
        # "delete" (chars only in corrected) and "insert" (chars only in raw)
        # leave the corrected positions unmapped.

    return mapping
