from __future__ import annotations

import importlib

from subforge.translation.base import Translator

BACKENDS: dict[str, tuple[str, str]] = {
    "nllb": ("subforge.translation.nllb_translator", "NLLBTranslator"),
}

DEFAULT = "nllb"
BACKEND_NAMES: list[str] = list(BACKENDS.keys())


def create_translator(method: str = DEFAULT, **kwargs) -> Translator:
    if method not in BACKENDS:
        raise ValueError(
            f"Unknown translation backend {method!r}. Choose from: {BACKEND_NAMES}"
        )
    module_path, class_name = BACKENDS[method]
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Backend {method!r} requires extra dependencies that are not installed. "
            f"Original error: {exc}"
        ) from exc
    return getattr(mod, class_name)(**kwargs)
