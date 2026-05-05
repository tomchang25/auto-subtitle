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

# Matches "1. text" or "1) text" — captures the number and the rest
_NUMBERING_RE = re.compile(r"^\d+[\.\)]\s*")
_NUMBERED_LINE_RE = re.compile(r"^(\d+)[\.\)]\s*(.*)")

MODEL_FALLBACK: list[str] = [
    "gemini-3.1-flash-lite-preview",  # 500 RPD
    "gemini-3-flash-preview",  # 20  RPD
    "gemini-2.5-flash-lite",  # 20  RPD
    "gemini-2.5-flash",  # 20  RPD
]

# Max chunks per API request — keeps output within token limits
# Gemini flash-lite handles ~1500 lines reliably; 500 is a safe middle ground
BATCH_SIZE = 300


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

    def _build_prompt(self, texts: list[str], src_name: str, tgt_name: str) -> str:
        n = len(texts)
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        return (
            f"You are a professional subtitle translator.\n"
            f"Translate the following {n} {src_name} subtitles into {tgt_name}.\n"
            f"Rules:\n"
            f"- Output EXACTLY {n} lines, numbered 1 to {n}\n"
            f"- One translation per line — do NOT merge or split lines\n"
            f"- Even if a line is very short or seems incomplete, translate it as-is on its own line\n"
            f"- Keep translations concise (suitable for subtitles)\n"
            f"- Output ONLY the numbered translations, no explanations\n\n"
            f"{numbered}"
        )

    def _parse_response_strict(self, text: str, expected: int) -> list[str] | None:
        """Parse response expecting exact count match (fast path)."""
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        translations = [
            _NUMBERING_RE.sub("", line) for line in lines if _NUMBERING_RE.match(line)
        ]
        return translations if len(translations) == expected else None

    def _parse_response_by_number(self, text: str, expected: int) -> list[str]:
        """Parse response by matching line numbers, filling gaps with empty strings."""
        result = [""] * expected
        matched = 0
        for line in text.strip().splitlines():
            m = _NUMBERED_LINE_RE.match(line.strip())
            if not m:
                continue
            idx = int(m.group(1)) - 1  # 0-based
            if 0 <= idx < expected:
                result[idx] = m.group(2).strip()
                matched += 1
        logger.info("Number-based parse: matched %d/%d lines", matched, expected)
        return result

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
        self, texts: list[str], src_name: str, tgt_name: str
    ) -> list[str]:
        """Translate a single batch of texts. Returns list of translations."""
        prompt = self._build_prompt(texts, src_name, tgt_name)
        logger.debug("=== PROMPT START ===\n%s\n=== PROMPT END ===", prompt)

        last_exc: Exception | None = None

        for model_idx, model in enumerate(self._models):
            logger.info(
                "Trying model: %s (%d/%d)", model, model_idx + 1, len(self._models)
            )

            best_response: str | None = None
            best_match_count: int = 0

            for attempt in range(3):
                try:
                    response_text = self._call_api(prompt, model)
                    logger.debug(
                        "=== RESPONSE START (model=%s, attempt=%d) ===\n%s\n=== RESPONSE END ===",
                        model, attempt + 1, response_text,
                    )

                    # Fast path: exact count match
                    parsed = self._parse_response_strict(response_text, len(texts))
                    if parsed is not None:
                        return parsed

                    # Count mismatch — log details
                    response_lines = [
                        line.strip()
                        for line in response_text.strip().splitlines()
                        if line.strip()
                    ]
                    numbered_lines = [
                        l for l in response_lines if _NUMBERING_RE.match(l)
                    ]
                    logger.warning(
                        "Response count mismatch: expected %d, got %d numbered lines "
                        "(total lines: %d, model=%s, attempt %d/3)",
                        len(texts), len(numbered_lines), len(response_lines),
                        model, attempt + 1,
                    )
                    if numbered_lines:
                        preview = (
                            numbered_lines[:3]
                            + (["..."] if len(numbered_lines) > 6 else [])
                            + numbered_lines[-3:]
                        )
                        logger.info(
                            "Response preview:\n  %s", "\n  ".join(preview)
                        )

                    # Track best response across retries
                    if len(numbered_lines) > best_match_count:
                        best_match_count = len(numbered_lines)
                        best_response = response_text

                except Exception as exc:
                    last_exc = exc
                    exc_str = str(exc)
                    is_rate_limit = "429" in exc_str or "quota" in exc_str.lower()

                    error_code = None
                    code_match = re.search(r"\b([45]\d{2})\b", exc_str)
                    if code_match:
                        error_code = code_match.group(1)

                    is_unavailable = "503" in exc_str or "UNAVAILABLE" in exc_str

                    if is_rate_limit or is_unavailable:
                        reason = (
                            "quota exceeded"
                            if is_rate_limit
                            else "unavailable (high demand)"
                        )
                        logger.warning(
                            "Model %s %s (code=%s), switching to next model: %s",
                            model, reason, error_code or "unknown", exc_str,
                        )
                        break  # → next model immediately
                    elif attempt < 2:
                        wait = 2**attempt
                        logger.warning(
                            "API error (model=%s, attempt %d/3, code=%s), retrying in %ds: %s",
                            model, attempt + 1, error_code or "unknown", wait, exc_str,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            "Model %s failed after 3 attempts (code=%s), switching: %s",
                            model, error_code or "unknown", exc_str,
                        )
                        break  # → next model

            # After retries: fall back to number-based parse of best response
            if best_response is not None:
                logger.warning(
                    "Using best response (%d/%d lines) with number-based alignment",
                    best_match_count, len(texts),
                )
                translations = self._parse_response_by_number(
                    best_response, len(texts)
                )
                missing = [i + 1 for i, t in enumerate(translations) if not t]
                if missing:
                    logger.warning("Missing translations for lines: %s", missing)
                return translations

        logger.error("Batch translation failed: all models exhausted")
        raise RuntimeError(
            "Translation failed: all models exhausted"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public API — batched translation
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
                            "Batch %d/%d: loaded from cache", batch_idx + 1, num_batches
                        )
                        all_results.extend(cached)
                        if progress_callback:
                            done = len(all_results)
                            pct = done * 100 // total
                            progress_callback(
                                "Translate",
                                f"{pct}% ({done}/{total} chunks, batch {batch_idx + 1}/{num_batches} cached)",
                            )
                        continue
                    logger.warning(
                        "Batch %d/%d: cache size mismatch (%d vs %d), re-translating",
                        batch_idx + 1, num_batches, len(cached), len(batch_chunks),
                    )

            batch_texts = [c["segment"] for c in batch_chunks]
            logger.info(
                "Batch %d/%d: chunks %d–%d (%d items)",
                batch_idx + 1, num_batches, start + 1, end, len(batch_texts),
            )

            translations = self._translate_batch(batch_texts, src_name, tgt_name)

            batch_results = [
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
                    f"{pct}% ({done}/{total} chunks, batch {batch_idx + 1}/{num_batches})",
                )

        return all_results
