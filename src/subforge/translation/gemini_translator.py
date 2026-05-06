from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, List

from subforge.translation.base import ProgressCallback, SubtitleChunk, TranslatedChunk

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger(__name__)

LANG_MAP: dict[str, str] = {
    "zho_Hant": "Traditional Chinese",
    "zho_Hans": "Simplified Chinese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "spa_Latn": "Spanish",
    "por_Latn": "Portuguese",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "eng_Latn": "English",
}

# Matches "1. text" or "1) text" -- captures the number and the rest
_NUMBERING_RE = re.compile(r"^\d+[\.\)]\s*")
_NUMBERED_LINE_RE = re.compile(r"^(\d+)[\.\)]\s*(.*)")

# Delimiters for block-based translation
_SENTENCE_DELIM = " ||| "       # between sentences within a block
_SENTENCE_DELIM_STRIP = "|||"
_BLOCK_DELIM = "\n\n===BLOCK===\n\n"  # between blocks
_BLOCK_DELIM_STRIP = "===BLOCK==="

MODEL_FALLBACK: list[str] = [
    "gemini-3.1-flash-lite-preview",  # 500 RPD
    "gemini-3-flash-preview",         # 20  RPD
    "gemini-2.5-flash-lite",          # 20  RPD
    "gemini-2.5-flash",               # 20  RPD
]

# Max chunks per API request
BATCH_SIZE = 500

# Target block size (sentences per block). Blocks are split at timestamp gaps.
BLOCK_TARGET_SIZE = 50

# Minimum timestamp gap (seconds) to consider as a block boundary
BLOCK_GAP_THRESHOLD = 1.5


def _resolve_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        from dotenv import dotenv_values

        vals = dotenv_values()
        key = vals.get("GEMINI_API_KEY")
        if key:
            return key
    except ImportError:
        pass
    raise ValueError(
        "Gemini API key not found. Provide it via the api_key parameter, "
        "the GEMINI_API_KEY environment variable, or a .env file."
    )


def _split_into_blocks(
    chunks: list[SubtitleChunk],
    target_size: int = BLOCK_TARGET_SIZE,
    gap_threshold: float = BLOCK_GAP_THRESHOLD,
) -> list[list[int]]:
    """Split chunk indices into blocks based on timestamp gaps.

    Returns list of blocks, where each block is a list of chunk indices.
    Prefers splitting at large timestamp gaps; falls back to target_size.
    """
    if not chunks:
        return []

    n = len(chunks)
    if n <= target_size:
        return [list(range(n))]

    # Find all gaps between consecutive chunks
    gaps: list[tuple[float, int]] = []  # (gap_seconds, index_after_gap)
    for i in range(1, n):
        gap = chunks[i]["start"] - chunks[i - 1]["end"]
        if gap >= gap_threshold:
            gaps.append((gap, i))

    # Sort gaps by size (largest first) to pick best split points
    gaps.sort(reverse=True)

    # Pick split points: want blocks of roughly target_size
    num_blocks_wanted = max(1, n // target_size)
    split_points = sorted(set(g[1] for g in gaps[: num_blocks_wanted - 1]))

    # If not enough natural gaps, add evenly-spaced splits
    if len(split_points) < num_blocks_wanted - 1:
        even_splits = set()
        for i in range(1, num_blocks_wanted):
            pos = i * n // num_blocks_wanted
            even_splits.add(pos)
        split_points = sorted(set(split_points) | even_splits)

    # Build blocks from split points
    blocks = []
    prev = 0
    for sp in split_points:
        if sp > prev:
            blocks.append(list(range(prev, sp)))
            prev = sp
    if prev < n:
        blocks.append(list(range(prev, n)))

    return blocks


class GeminiTranslator:
    """Translate subtitle chunks via Google Gemini API (google-genai SDK)."""

    def __init__(
        self,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "zho_Hant",
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = BATCH_SIZE,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._api_key = _resolve_api_key(api_key)
        self._models = [model] if model else list(MODEL_FALLBACK)
        self._client: genai.Client | None = None  # type: ignore[name-defined]
        self.batch_size = batch_size

    def _load(self):
        if self._client is None:
            from google import genai as _genai

            self._client = _genai.Client(api_key=self._api_key)

    # ------------------------------------------------------------------
    # Prompt / parse helpers
    # ------------------------------------------------------------------

    def _build_prompt_blocked(
        self, blocks: list[list[str]], src_name: str, tgt_name: str
    ) -> str:
        """Build a prompt with block structure.

        Each block's sentences are joined by |||.
        Blocks are separated by ===BLOCK===.
        """
        num_blocks = len(blocks)
        total_sentences = sum(len(b) for b in blocks)

        block_strs = []
        for block in blocks:
            block_strs.append(_SENTENCE_DELIM.join(block))

        joined = _BLOCK_DELIM.join(block_strs)

        return (
            f"You are a professional subtitle translator.\n"
            f"Translate the following {src_name} subtitles into {tgt_name}.\n"
            f"The text is organized into {num_blocks} blocks separated by ===BLOCK===.\n"
            f"Within each block, sentences are separated by ||| delimiters.\n\n"
            f"Rules:\n"
            f"- Preserve ALL ===BLOCK=== separators (exactly {num_blocks - 1})\n"
            f"- Within each block, preserve ALL ||| delimiters\n"
            f"- Do NOT add or remove any ===BLOCK=== or ||| delimiter\n"
            f"- Translate each sentence between ||| independently\n"
            f"- Even very short segments (single words) must be translated as-is\n"
            f"- Keep translations concise (suitable for subtitles)\n"
            f"- Output ONLY the translated text with all delimiters preserved\n\n"
            f"{joined}"
        )

    def _parse_blocked_response(
        self, text: str, blocks: list[list[str]]
    ) -> list[str] | None:
        """Parse a block-structured response.

        Returns flat list of translations if ALL blocks have correct count.
        Returns None only if block-level structure is wrong.
        If individual blocks have wrong ||| count, realigns per-block.
        """
        from subforge.translation.aligner import realign

        cleaned = text.strip()
        # Strip markdown fences if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        # Split into blocks
        response_blocks = cleaned.split(_BLOCK_DELIM_STRIP)
        response_blocks = [b.strip() for b in response_blocks]

        # Remove empty trailing blocks
        while response_blocks and not response_blocks[-1]:
            response_blocks.pop()

        if len(response_blocks) != len(blocks):
            logger.warning(
                "Block count mismatch: expected %d, got %d",
                len(blocks), len(response_blocks),
            )
            return None

        # Parse each block individually
        all_translations: list[str] = []
        blocks_realigned = 0

        for block_idx, (src_block, resp_block) in enumerate(
            zip(blocks, response_blocks)
        ):
            expected = len(src_block)
            segments = [s.strip() for s in resp_block.split(_SENTENCE_DELIM_STRIP)]

            if len(segments) == expected:
                all_translations.extend(segments)
            else:
                # Realign this block only
                logger.info(
                    "Block %d: ||| mismatch (%d vs %d), realigning",
                    block_idx + 1, len(segments), expected,
                )
                aligned = realign(src_block, segments)
                all_translations.extend(aligned)
                blocks_realigned += 1

        if blocks_realigned > 0:
            logger.info(
                "Parsed %d blocks, %d needed realignment",
                len(blocks), blocks_realigned,
            )
        else:
            logger.info("All %d blocks parsed perfectly", len(blocks))

        return all_translations

    def _call_api(self, prompt: str, model: str) -> str:
        """Call Gemini API with specified model and return the response text."""
        assert self._client is not None, "Call _load() before _call_api()"
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
        )
        if response.text is None:
            raise RuntimeError("Gemini returned empty response (no text)")
        return response.text

    # ------------------------------------------------------------------
    # Single-batch translation (with retry + model fallback)
    # ------------------------------------------------------------------

    def _translate_batch(
        self,
        texts: list[str],
        chunks: list[SubtitleChunk],
        src_name: str,
        tgt_name: str,
    ) -> list[str]:
        """Translate a single batch using block-based approach.

        Splits texts into blocks based on timestamp gaps, sends as one API call,
        and realigns per-block if needed.
        """
        from subforge.translation.aligner import realign

        # Split into blocks based on timestamps
        block_indices = _split_into_blocks(chunks)
        blocks: list[list[str]] = [[texts[i] for i in bi] for bi in block_indices]

        logger.info(
            "Batch: %d sentences -> %d blocks (sizes: %s)",
            len(texts),
            len(blocks),
            [len(b) for b in blocks],
        )

        prompt = self._build_prompt_blocked(blocks, src_name, tgt_name)
        logger.debug("=== PROMPT START ===\n%s\n=== PROMPT END ===", prompt)

        last_exc: Exception | None = None
        best_response: str | None = None

        for model_idx, model in enumerate(self._models):
            logger.info(
                "Trying model: %s (%d/%d)", model, model_idx + 1, len(self._models)
            )

            for attempt in range(3):
                try:
                    response_text = self._call_api(prompt, model)
                    logger.debug(
                        "=== RESPONSE (model=%s, attempt=%d) ===\n%s\n=== END ===",
                        model, attempt + 1, response_text,
                    )

                    parsed = self._parse_blocked_response(response_text, blocks)
                    if parsed is not None:
                        return parsed

                    # Block-level mismatch -- save for fallback realign
                    best_response = response_text
                    logger.warning(
                        "Block structure mismatch (model=%s, attempt %d/3)",
                        model, attempt + 1,
                    )

                except Exception as exc:
                    last_exc = exc
                    exc_str = str(exc)
                    is_rate_limit = "429" in exc_str or "quota" in exc_str.lower()
                    is_unavailable = "503" in exc_str or "UNAVAILABLE" in exc_str

                    if is_rate_limit or is_unavailable:
                        reason = (
                            "quota exceeded"
                            if is_rate_limit
                            else "unavailable (high demand)"
                        )
                        logger.warning(
                            "Model %s %s, switching: %s", model, reason, exc_str
                        )
                        break
                    elif attempt < 2:
                        wait = 2 ** attempt
                        logger.warning(
                            "API error (model=%s, attempt %d/3), retrying in %ds: %s",
                            model, attempt + 1, wait, exc_str,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            "Model %s failed after 3 attempts, switching: %s",
                            model, exc_str,
                        )
                        break

        # Fallback: if we have any response, do a full realign
        if best_response is not None:
            logger.warning("All block-parse attempts failed, doing full realign")
            # Extract whatever segments we can from the response
            cleaned = best_response.strip()
            # Try to split by any delimiter present
            all_segments = []
            for block_text in cleaned.split(_BLOCK_DELIM_STRIP):
                for seg in block_text.split(_SENTENCE_DELIM_STRIP):
                    seg = seg.strip()
                    if seg:
                        all_segments.append(seg)
            if all_segments:
                return realign(texts, all_segments)

        logger.error("Batch translation failed: all models exhausted")
        raise RuntimeError(
            "Translation failed: all models exhausted"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public API -- batched translation
    # ------------------------------------------------------------------

    def translate(
        self,
        chunks: List[SubtitleChunk],
        cache_dir: Path | None = None,
        force: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> List[TranslatedChunk]:
        """Translate all chunks, splitting into batches for long inputs.

        If cache_dir is provided, each batch result is saved/loaded as JSON.
        Set force=True to ignore cached batches.
        """
        if not chunks:
            return []
        self._load()
        src_name = LANG_MAP.get(self.src_lang, self.src_lang)
        tgt_name = LANG_MAP.get(self.tgt_lang, self.tgt_lang)

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        total = len(chunks)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        logger.info(
            "Translation: %d chunks, batch_size=%d, %d batch(es)",
            total, self.batch_size, num_batches,
        )

        all_results: List[TranslatedChunk] = []

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, total)
            batch_chunks = chunks[start:end]

            # Check batch cache
            batch_cache: Path | None = None
            if cache_dir is not None:
                batch_cache = cache_dir / f"batch_{batch_idx:03d}.json"
                if not force and batch_cache.exists():
                    import json

                    with open(batch_cache, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if len(cached) == len(batch_chunks):
                        logger.info(
                            "Batch %d/%d: loaded from cache",
                            batch_idx + 1, num_batches,
                        )
                        all_results.extend(cached)
                        if progress_callback:
                            done = len(all_results)
                            pct = done * 100 // total
                            progress_callback(
                                "Translate",
                                f"{pct}% ({done}/{total} chunks, "
                                f"batch {batch_idx + 1}/{num_batches} cached)",
                            )
                        continue
                    logger.warning(
                        "Batch %d/%d: cache size mismatch (%d vs %d), re-translating",
                        batch_idx + 1, num_batches, len(cached), len(batch_chunks),
                    )

            batch_texts = [c["segment"] for c in batch_chunks]
            logger.info(
                "Batch %d/%d: chunks %d-%d (%d items)",
                batch_idx + 1, num_batches, start + 1, end, len(batch_texts),
            )

            translations = self._translate_batch(
                batch_texts, batch_chunks, src_name, tgt_name
            )

            batch_results: list[TranslatedChunk] = [
                {**chunk, "translation": t}
                for chunk, t in zip(batch_chunks, translations)
            ]
            all_results.extend(batch_results)

            # Save batch cache
            if batch_cache is not None:
                import json

                with open(batch_cache, "w", encoding="utf-8") as f:
                    json.dump(batch_results, f, ensure_ascii=False, indent=2)

            done = len(all_results)
            pct = done * 100 // total
            logger.info(
                "Batch %d/%d complete (%d/%d total)",
                batch_idx + 1, num_batches, done, total,
            )
            if progress_callback:
                progress_callback(
                    "Translate",
                    f"{pct}% ({done}/{total} chunks, "
                    f"batch {batch_idx + 1}/{num_batches})",
                )

        return all_results
