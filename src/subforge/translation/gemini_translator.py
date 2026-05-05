from __future__ import annotations

import logging
import os
import re
import time
from typing import TYPE_CHECKING, List

from subforge.translation.base import SubtitleChunk, TranslatedChunk

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

_NUMBERING_RE = re.compile(r"^\d+[\.\)]\s*")


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

    MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "zho_Hant",
        api_key: str | None = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._api_key = _resolve_api_key(api_key)
        self._client: genai.Client | None = None  # type: ignore[name-defined]

    def _load(self):
        if self._client is None:
            from google import genai as _genai

            self._client = _genai.Client(api_key=self._api_key)

    def _build_prompt(self, texts: list[str], src_name: str, tgt_name: str) -> str:
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        return (
            f"You are a professional subtitle translator.\n"
            f"Translate the following {src_name} subtitles into {tgt_name}.\n"
            f"Rules:\n"
            f"- Keep translations concise (suitable for subtitles)\n"
            f"- Maintain the same numbering\n"
            f"- Output ONLY the translations, one per line, prefixed with the number\n"
            f"- Do not add explanations\n\n"
            f"{numbered}"
        )

    def _parse_response(self, text: str, expected: int) -> list[str] | None:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        translations = [
            _NUMBERING_RE.sub("", line) for line in lines if _NUMBERING_RE.match(line)
        ]
        return translations if len(translations) == expected else None

    def _call_api(self, prompt: str) -> str:
        """Call Gemini API and return the response text."""
        assert self._client is not None, "Call _load() before _call_api()"
        response = self._client.models.generate_content(
            model=self.MODEL,
            contents=prompt,
        )
        if response.text is None:
            raise RuntimeError("Gemini returned empty response (no text)")
        return response.text

    def translate(self, chunks: List[SubtitleChunk]) -> List[TranslatedChunk]:
        """Translate all chunks in a single API call."""
        if not chunks:
            return []
        self._load()
        src_name = LANG_MAP.get(self.src_lang, self.src_lang)
        tgt_name = LANG_MAP.get(self.tgt_lang, self.tgt_lang)
        texts = [c["segment"] for c in chunks]

        prompt = self._build_prompt(texts, src_name, tgt_name)

        for attempt in range(3):
            try:
                response_text = self._call_api(prompt)
                parsed = self._parse_response(response_text, len(texts))
                if parsed is not None:
                    return [{**c, "translation": t} for c, t in zip(chunks, parsed)]
                if attempt < 2:
                    logger.warning(
                        "Response count mismatch (got lines, expected %d, attempt %d/3), retrying",
                        len(texts),
                        attempt + 1,
                    )
                else:
                    logger.warning(
                        "Count mismatch after 3 retries, returning partial results"
                    )
                    # Pad with empty strings if we got fewer translations
                    lines = [
                        line.strip()
                        for line in response_text.strip().splitlines()
                        if line.strip()
                    ]
                    translations = [
                        _NUMBERING_RE.sub("", line)
                        for line in lines
                        if _NUMBERING_RE.match(line)
                    ]
                    # Pad or truncate to match input length
                    translations = translations[: len(texts)]
                    translations.extend([""] * (len(texts) - len(translations)))
                    return [
                        {**c, "translation": t} for c, t in zip(chunks, translations)
                    ]
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "quota" in str(exc).lower()
                if attempt < 2:
                    wait = 30 if is_rate_limit else 2**attempt
                    logger.warning(
                        "API error (attempt %d/3), retrying in %.0fs",
                        attempt + 1,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Translation failed after 3 attempts: {exc}"
                    ) from exc

        raise RuntimeError("Translation failed unexpectedly")
