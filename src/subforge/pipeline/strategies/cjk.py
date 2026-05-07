"""Transcript-first CJK subtitle pipeline.

Unlike the English path — which relies on ASR word timestamps to drive
sentence segmentation through spaCy — the CJK strategy treats text quality
and timing as separate concerns. It runs five named stages, each of which
persists its artifact under ``project_dir/cjk/`` so a rerun can resume from
the latest valid stage:

  1. ``raw_transcript.json``       — text + per-character timings rebuilt
                                    from ASR ``word_segments``.
  2. ``corrected_transcript.json`` — output of the pluggable corrector
                                    (defaults to a no-op).
  3. ``sentences.json``            — sentence list split from text only,
                                    independent of timing.
  4. ``alignment.json``            — sentences re-attached to timing via
                                    char-level fuzzy alignment back to the
                                    raw transcript.
  5. final segments                — fed through the shared timing-refine,
                                    length-split and short-merge passes,
                                    and returned to the orchestrator.

Each stage degrades gracefully: a failed corrector falls back to raw text,
a failed alignment falls back to the legacy ``split_word_segments_by_
punctuation`` path, and missing per-character timing distributes uniformly.
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

logger = logging.getLogger(__name__)


_STAGE_FILES = (
    "raw_transcript.json",
    "corrected_transcript.json",
    "sentences.json",
    "alignment.json",
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
        # agnostic.
        if ctx.profile.code == "zh" and ctx.chinese_benchmark:
            from subforge.nlp.chinese_benchmark import hard_cut_chinese_segments

            ctx.emit(
                "NLP",
                f"Chinese benchmark mode: hard-cut segmentation "
                f"(hard_chars={CHINESE_BENCHMARK_HARD_CHARS}, "
                f"gap={CHINESE_BENCHMARK_GAP_SECONDS}s)",
            )
            return hard_cut_chinese_segments(
                word_segments,
                hard_chars=CHINESE_BENCHMARK_HARD_CHARS,
                gap_seconds=CHINESE_BENCHMARK_GAP_SECONDS,
            )

        ws_hash = _hash(json.dumps(word_segments, ensure_ascii=False, sort_keys=True))

        # Stage 1 — raw transcript (text + char timings + char→word index)
        raw = self._stage_raw_transcript(word_segments, ctx, cjk_dir, ws_hash)
        ctx.check_cancel()

        # Stage 2 — corrected transcript
        corrected_text = self._stage_correct(raw["text"], ctx, cjk_dir, ws_hash)
        ctx.check_cancel()

        # Stage 3 — sentence list (text only)
        sentences = self._stage_split_sentences(
            corrected_text, ctx, cjk_dir, ws_hash
        )
        ctx.check_cancel()

        # Stage 4 — re-attach timing
        aligned = self._stage_align(
            sentences, corrected_text, raw, word_segments, ctx, cjk_dir, ws_hash
        )
        ctx.check_cancel()

        if not aligned:
            ctx.emit(
                "Align",
                "Transcript-first alignment produced no chunks — "
                "falling back to word-segment punctuation split",
            )
            aligned = split_word_segments_by_punctuation(word_segments, ctx.profile)

        # Stage 5 — shared timing refinement / length split / short merge
        return self._finalize(aligned, ctx)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    def _stage_raw_transcript(
        self,
        word_segments: list[dict],
        ctx: StrategyContext,
        cjk_dir: Path,
        ws_hash: str,
    ) -> dict:
        path = cjk_dir / "raw_transcript.json"
        cached = None if ctx.force else _load_stage(path, ws_hash)
        if cached is not None:
            ctx.emit("CJK", "Stage 1: raw transcript (cached)")
            return cached

        ctx.emit("CJK", "Stage 1: building raw transcript")
        join_token = ctx.profile.join_token
        text_parts: list[str] = []
        char_timings: list[dict] = []
        char_to_word: list[int] = []

        for w_idx, seg in enumerate(word_segments):
            word = seg.get("word", "") or ""
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
            if end < start:
                end = start

            if w_idx > 0 and join_token:
                anchor = char_timings[-1]["end"] if char_timings else start
                for ch in join_token:
                    text_parts.append(ch)
                    char_timings.append({"ch": ch, "start": anchor, "end": anchor})
                    char_to_word.append(-1)

            if not word:
                continue

            n = len(word)
            step = (end - start) / n if n > 0 and end > start else 0.0
            for i, ch in enumerate(word):
                ch_start = start + i * step if step > 0 else start
                ch_end = start + (i + 1) * step if step > 0 else end
                text_parts.append(ch)
                char_timings.append({"ch": ch, "start": ch_start, "end": ch_end})
                char_to_word.append(w_idx)

        data = {
            "text": "".join(text_parts),
            "char_timings": char_timings,
            "char_to_word": char_to_word,
        }
        _save_stage(path, ws_hash, data)
        return data

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    def _stage_correct(
        self,
        raw_text: str,
        ctx: StrategyContext,
        cjk_dir: Path,
        ws_hash: str,
    ) -> str:
        path = cjk_dir / "corrected_transcript.json"
        corrector_id = type(self.corrector).__name__
        input_hash = _hash(ws_hash, raw_text, corrector_id)
        cached = None if ctx.force else _load_stage(path, input_hash)
        if cached is not None:
            ctx.emit("CJK", f"Stage 2: corrected transcript (cached, {corrector_id})")
            return cached["text"]

        ctx.emit("CJK", f"Stage 2: correcting transcript ({corrector_id})")
        try:
            corrected = self.corrector.correct(raw_text, ctx.profile.code)
            corrected_ok = isinstance(corrected, str) and corrected != ""
        except Exception as exc:
            logger.warning("Corrector %s raised: %s — using raw transcript", corrector_id, exc)
            ctx.emit("CJK", f"Stage 2: corrector failed ({type(exc).__name__}); using raw text")
            corrected = raw_text
            corrected_ok = False

        if not corrected_ok:
            corrected = raw_text

        data = {"text": corrected, "corrector": corrector_id, "applied": corrected_ok}
        _save_stage(path, input_hash, data)
        return corrected

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------
    def _stage_split_sentences(
        self,
        text: str,
        ctx: StrategyContext,
        cjk_dir: Path,
        ws_hash: str,
    ) -> list[str]:
        path = cjk_dir / "sentences.json"
        input_hash = _hash(ws_hash, text, "".join(sorted(ctx.profile.sentence_end)))
        cached = None if ctx.force else _load_stage(path, input_hash)
        if cached is not None:
            ctx.emit("CJK", f"Stage 3: sentences (cached, {len(cached)})")
            return cached

        ctx.emit("CJK", "Stage 3: splitting transcript into sentences")
        sentence_end = ctx.profile.sentence_end
        sentences: list[str] = []
        current = ""
        for ch in text:
            current += ch
            if ch in sentence_end:
                sentences.append(current)
                current = ""
        if current:
            sentences.append(current)

        _save_stage(path, input_hash, sentences)
        return sentences

    # ------------------------------------------------------------------
    # Stage 4
    # ------------------------------------------------------------------
    def _stage_align(
        self,
        sentences: list[str],
        corrected_text: str,
        raw: dict,
        word_segments: list[dict],
        ctx: StrategyContext,
        cjk_dir: Path,
        ws_hash: str,
    ) -> list[list[dict]]:
        path = cjk_dir / "alignment.json"
        input_hash = _hash(
            ws_hash,
            corrected_text,
            json.dumps(sentences, ensure_ascii=False),
        )
        cached = None if ctx.force else _load_stage(path, input_hash)
        if cached is not None:
            ctx.emit("CJK", f"Stage 4: alignment (cached, {len(cached)} chunks)")
            return cached

        ctx.emit("CJK", "Stage 4: aligning sentences with timing")
        char_to_word = raw["char_to_word"]
        raw_text = raw["text"]
        punct = ctx.profile.punctuation

        try:
            mapping = _map_corrected_to_raw(corrected_text, raw_text)
        except Exception as exc:
            logger.warning("Char-level alignment failed: %s", exc)
            ctx.emit("Align", f"Char-level alignment failed ({type(exc).__name__}); falling back")
            return []

        # Per-sentence raw-text span derived from the corrected→raw mapping.
        sentence_spans: list[tuple[int, int] | None] = []
        cursor = 0
        for sent in sentences:
            s, e = cursor, cursor + len(sent)
            cursor = e
            raw_indices = [
                mapping[i] for i in range(s, e)
                if 0 <= i < len(mapping) and mapping[i] is not None
            ]
            if raw_indices:
                sentence_spans.append((min(raw_indices), max(raw_indices)))
            else:
                sentence_spans.append(None)

        if not any(sentence_spans):
            # No sentence has a raw anchor — alignment is unusable.
            return []

        # Per-word raw-index range so we can route each ASR word to the right
        # sentence.
        word_first_raw = [-1] * len(word_segments)
        word_last_raw = [-1] * len(word_segments)
        for raw_i, w in enumerate(char_to_word):
            if w < 0:
                continue
            if word_first_raw[w] == -1:
                word_first_raw[w] = raw_i
            word_last_raw[w] = raw_i

        def _route(w_idx: int) -> int:
            raw_start = word_first_raw[w_idx]
            raw_end = word_last_raw[w_idx]
            if raw_start < 0:
                # Empty/whitespace word — attach to first sentence with a span.
                for s_idx, span in enumerate(sentence_spans):
                    if span is not None:
                        return s_idx
                return 0
            raw_mid = (raw_start + raw_end) // 2
            best, best_dist = 0, None
            for s_idx, span in enumerate(sentence_spans):
                if span is None:
                    continue
                s_lo, s_hi = span
                if s_lo <= raw_mid <= s_hi:
                    return s_idx
                d = min(abs(raw_mid - s_lo), abs(raw_mid - s_hi))
                if best_dist is None or d < best_dist:
                    best, best_dist = s_idx, d
            return best

        sentence_word_indices: list[list[int]] = [[] for _ in sentences]
        for w_idx in range(len(word_segments)):
            sentence_word_indices[_route(w_idx)].append(w_idx)

        chunks: list[list[dict]] = []
        for word_indices in sentence_word_indices:
            if not word_indices:
                continue
            word_indices.sort()
            tokens: list[dict] = []
            for w_idx in word_indices:
                seg = word_segments[w_idx]
                word = seg.get("word", "") or ""
                tokens.append({
                    "text": word,
                    "whitespace": "",
                    "is_punct": bool(word and word[-1] in punct),
                    "start": float(seg.get("start", 0.0) or 0.0),
                    "end": float(seg.get("end", 0.0) or 0.0),
                })
            if tokens:
                chunks.append(tokens)

        _save_stage(path, input_hash, chunks)
        return chunks

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
