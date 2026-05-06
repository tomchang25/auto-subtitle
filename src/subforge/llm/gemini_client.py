"""Shared Gemini client with model fallback and API key resolution."""

import logging
import os
import time

logger = logging.getLogger(__name__)

MODEL_FALLBACK: list[str] = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve Gemini API key from argument, env var, or .env file."""
    if explicit:
        return explicit
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        from dotenv import dotenv_values

        key = dotenv_values().get("GEMINI_API_KEY")
        if key:
            return key
    except ImportError:
        pass
    raise ValueError(
        "Gemini API key not found. Provide it via the api_key parameter, "
        "the GEMINI_API_KEY environment variable, or a .env file."
    )


class GeminiClient:
    """Thin wrapper around google.genai with model fallback."""

    def __init__(self, api_key: str | None = None, models: list[str] | None = None):
        self._api_key = resolve_api_key(api_key)
        self._models = models if models else list(MODEL_FALLBACK)
        self._client = None

    def _load(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)

    def generate(self, prompt: str, max_retries: int = 2) -> str:
        """Call Gemini with model fallback. Returns response text.

        Tries each model in order. For each model, retries up to max_retries
        times on transient errors. Raises on complete failure.
        """
        self._load()
        assert self._client is not None
        last_exc: Exception | None = None

        for model_idx, model in enumerate(self._models):
            logger.info(
                "Trying model: %s (%d/%d)", model, model_idx + 1, len(self._models)
            )

            for attempt in range(max_retries + 1):
                try:
                    response = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    if response.text is None:
                        raise RuntimeError("Gemini returned empty response")
                    return response.text
                except Exception as e:
                    last_exc = e
                    err_msg = str(e)
                    if "429" in err_msg or "quota" in err_msg.lower():
                        logger.warning("Model %s rate limited, trying next", model)
                        break
                    if attempt < max_retries:
                        wait = 2**attempt
                        logger.warning(
                            "Model %s attempt %d failed: %s. Retrying in %ds...",
                            model,
                            attempt + 1,
                            e,
                            wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning("Model %s exhausted retries: %s", model, e)
                        break

        raise RuntimeError(
            f"All Gemini models failed. Last error: {last_exc}"
        ) from last_exc
