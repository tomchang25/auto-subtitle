from __future__ import annotations

import importlib
import logging

from subforge.config import (
    WHISPER_TIER_MAP,
    FUNASR_TIER_MAP,
    SENSEVOICE_TIER_MAP,
)

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
    "sensevoice": (
        "subforge.transcription.sensevoice_transcriber",
        "transcribe_audio_word_level",
    ),
}

DEFAULT_BACKEND = "whisper"

# No languages receive automatic FunASR routing any more.
# Paraformer (funasr) and SenseVoice are available as explicit backend choices
# for experimental / legacy use; the Chinese auto-default is now whisper.
_FUNASR_LANGUAGES: frozenset[str] = frozenset()

BACKEND_NAMES: list[str] = ["auto"] + list(BACKENDS)

_BACKEND_EXTRA: dict[str, str] = {
    "whisper": "full",
    "funasr": "funasr",
    "sensevoice": "funasr",   # SenseVoice ships via the funasr package
}

# Third-party package each backend actually needs at runtime.
_BACKEND_RUNTIME_PKG: dict[str, str] = {
    "whisper": "faster_whisper",
    "funasr": "funasr",
    "sensevoice": "funasr",
}

_TIER_MAPS: dict[str, dict[str, str]] = {
    "whisper": WHISPER_TIER_MAP,
    "funasr": FUNASR_TIER_MAP,
    "sensevoice": SENSEVOICE_TIER_MAP,
}


def resolve_model(model_name_or_tier: str, backend: str) -> str:
    """Resolve an abstract tier name or an explicit model name to a concrete model.

    If *model_name_or_tier* matches a tier key (accuracy/large/medium/small) the
    backend-specific concrete model is returned.  Otherwise the value is
    returned unchanged so that explicit model names remain valid.
    """
    tier_map = _TIER_MAPS.get(backend, WHISPER_TIER_MAP)
    return tier_map.get(model_name_or_tier, model_name_or_tier)


def resolve_backend(backend: str, source_language: str) -> str:
    """Return the concrete backend name (never 'auto').

    When *backend* is ``'auto'``, whisper is always selected regardless of
    *source_language*.  FunASR (Paraformer) and SenseVoice must be chosen
    explicitly for experimental / legacy Chinese benchmarks.
    """
    if backend == "auto":
        return "funasr" if source_language in _FUNASR_LANGUAGES else DEFAULT_BACKEND
    if backend not in BACKENDS:
        logger.warning(
            "Unknown ASR backend %r, falling back to %r", backend, DEFAULT_BACKEND
        )
        return DEFAULT_BACKEND
    return backend


def is_backend_available(backend: str) -> bool:
    """Return *True* if *backend*'s runtime dependency can be imported."""
    pkg = _BACKEND_RUNTIME_PKG.get(backend)
    if pkg is None:
        return False
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def get_transcriber(
    backend: str,
    source_language: str = "auto",
) -> tuple:
    """Return ``(transcribe_fn, actual_backend)`` for the resolved backend.

    If the requested backend's runtime dependency is missing and it is **not**
    the default, the function falls back to the default backend (whisper)
    instead of raising.  The caller can compare *actual_backend* with the
    originally resolved backend to detect a fallback.
    """
    concrete = resolve_backend(backend, source_language)

    if not is_backend_available(concrete):
        if concrete == DEFAULT_BACKEND:
            extra = _BACKEND_EXTRA.get(concrete, concrete)
            raise ImportError(
                f"ASR backend {concrete!r} requires extra dependencies. "
                f"Install with: pip install subforge[{extra}]"
            )
        extra = _BACKEND_EXTRA.get(concrete, concrete)
        logger.warning(
            "Backend %r is not installed (pip install subforge[%s]). "
            "Falling back to %r.",
            concrete, extra, DEFAULT_BACKEND,
        )
        concrete = DEFAULT_BACKEND

    module_path, fn_name = BACKENDS[concrete]
    mod = importlib.import_module(module_path)
    return getattr(mod, fn_name), concrete
