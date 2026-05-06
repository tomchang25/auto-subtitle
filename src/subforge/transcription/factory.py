from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

BACKENDS: dict[str, tuple[str, str]] = {
    "whisper": (
        "subforge.transcription.faster_whisper_transcriber",
        "transcribe_audio_word_level",
    ),
    "funasr": (
        "subforge.transcription.funasr_transcriber",
        "transcribe_audio_word_level",
    ),
}

DEFAULT_BACKEND = "whisper"

_FUNASR_LANGUAGES = frozenset({"zh"})

BACKEND_NAMES: list[str] = ["auto"] + list(BACKENDS)

_BACKEND_EXTRA: dict[str, str] = {
    "whisper": "full",
    "funasr": "funasr",
}


def resolve_backend(backend: str, source_language: str) -> str:
    """Return the concrete backend name (never 'auto')."""
    if backend == "auto":
        return "funasr" if source_language in _FUNASR_LANGUAGES else DEFAULT_BACKEND
    if backend not in BACKENDS:
        logger.warning(
            "Unknown ASR backend %r, falling back to %r", backend, DEFAULT_BACKEND
        )
        return DEFAULT_BACKEND
    return backend


def get_transcriber(backend: str, source_language: str = "auto"):
    """Return the transcribe_audio_word_level callable for the resolved backend."""
    concrete = resolve_backend(backend, source_language)
    module_path, fn_name = BACKENDS[concrete]
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        extra = _BACKEND_EXTRA.get(concrete, concrete)
        raise ImportError(
            f"ASR backend {concrete!r} requires extra dependencies that are not installed. "
            f"Install with: pip install subforge[{extra}]. Original error: {exc}"
        ) from exc
    return getattr(mod, fn_name)
