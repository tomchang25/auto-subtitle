from __future__ import annotations

import logging
import os
import re
import time
from typing import List

from subforge.translation.base import SubtitleChunk, TranslatedChunk

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
    """Translate subtitle chunks via Gemini 2.0 Flash API."""

    MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "zho_Hant",
        batch_size: int = 30,
        api_key: str | None = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self._api_key = _resolve_api_key(api_key)
        self._client = None

    def _load(self):
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self.MODEL)

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
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        translations = [_NUMBERING_RE.sub("", l) for l in lines if _NUMBERING_RE.match(l)]
        return translations if len(translations) == expected else None

    def _translate_one_by_one(
        self, texts: list[str], src_name: str, tgt_name: str
    ) -> list[str]:
        results = []
        for text in texts:
            prompt = self._build_prompt([text], src_name, tgt_name)
            try:
                response = self._client.generate_content(prompt)
                lines = [l.strip() for l in response.text.strip().splitlines() if l.strip()]
                translated = _NUMBERING_RE.sub("", lines[0]) if lines else ""
            except Exception as exc:
                logger.warning("Per-sentence translation failed: %s", exc)
                translated = ""
            results.append(translated)
        return results

    def _translate_batch(
        self, texts: list[str], src_name: str, tgt_name: str
    ) -> list[str]:
        prompt = self._build_prompt(texts, src_name, tgt_name)
        for attempt in range(3):
            try:
                response = self._client.generate_content(prompt)
                parsed = self._parse_response(response.text, len(texts))
                if parsed is not None:
                    return parsed
                if attempt < 2:
                    logger.warning(
                        "Response count mismatch (attempt %d/3), retrying", attempt + 1
                    )
                else:
                    logger.warning("Count mismatch after 3 retries, falling back to per-sentence")
                    return self._translate_one_by_one(texts, src_name, tgt_name)
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "quota" in str(exc).lower()
                if attempt < 2:
                    wait = 2 ** (attempt + 1) if is_rate_limit else 2 ** attempt
                    logger.warning(
                        "API error (attempt %d/3), retrying in %ds: %s", attempt + 1, wait, exc
                    )
                    time.sleep(wait)
                else:
                    logger.error("API error after 3 attempts: %s", exc)
                    return [""] * len(texts)
        return [""] * len(texts)

    def translate(self, chunks: List[SubtitleChunk]) -> List[TranslatedChunk]:
        if not chunks:
            return []
        self._load()
        src_name = LANG_MAP.get(self.src_lang, self.src_lang)
        tgt_name = LANG_MAP.get(self.tgt_lang, self.tgt_lang)
        texts = [c["segment"] for c in chunks]
        translations: list[str] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            translations.extend(self._translate_batch(batch, src_name, tgt_name))

        return [{**c, "translation": t} for c, t in zip(chunks, translations)]
