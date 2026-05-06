from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# AED-L supports word-level timestamps; LLM-L does not → AED-L is the default.
SUPPORTED_MODELS = (
    "FireRedASR-AED-L",
    "FireRedASR-LLM-L",
)
DEFAULT_MODEL = "FireRedASR-AED-L"

_loaded_models: dict = {}


def load_model(model_name: str):
    if model_name not in SUPPORTED_MODELS:
        logger.warning(
            "FireRedASR: unrecognized model %r, falling back to %r",
            model_name,
            DEFAULT_MODEL,
        )
        model_name = DEFAULT_MODEL
    if model_name not in _loaded_models:
        try:
            from fireredasr.models.fireredasr import FireRedASR
        except ImportError as exc:
            raise ImportError(
                "FireRedASR is not installed. Install with: pip install subforge[fireredasr]"
            ) from exc
        repo_id = f"FireRedTeam/{model_name}"
        logger.info("Loading FireRedASR model: %s (repo=%s)", model_name, repo_id)
        _loaded_models[model_name] = FireRedASR.from_pretrained(repo_id)
    return _loaded_models[model_name]


def transcribe_audio_word_level(
    wav_path: Path,
    model_name: str,
    progress_callback=None,
) -> tuple[list, str]:
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {wav_path}")

    if progress_callback:
        progress_callback("Transcribe", "Loading FireRedASR model…")

    model = load_model(model_name)

    logger.info("Transcribing with FireRedASR (%s): %s", model_name, wav_path)

    if progress_callback:
        progress_callback("Transcribe", "Running FireRedASR inference…")

    results = model.transcribe(
        [str(wav_path)],
        batch_size=1,
        decode_config={"use_timestamp": True, "beam_size": 3, "nbest": 1},
    )

    if not results:
        raise ValueError("FireRedASR did not return any results")

    logger.debug("FireRedASR raw output (first item): %r", results[0])

    segments = _extract_segments(results)

    if not segments:
        raise ValueError("FireRedASR did not return character-level timestamps")

    detected_lang = "zh"
    logger.info(
        "FireRedASR transcription complete: %d characters, backend=fireredasr, "
        "model=%s, lang=%s, has_timestamps=%s",
        len(segments),
        model_name,
        detected_lang,
        any(s["start"] > 0 or s["end"] > 0 for s in segments),
    )
    return segments, detected_lang


def _extract_segments(results: list[dict]) -> list[dict]:
    """Extract character-level word segments from FireRedASR output.

    FireRedASR-AED-L returns ``timestamp`` as a list of
    ``[character, start_sec, end_sec]`` triples (seconds, not milliseconds).
    FireRedASR-LLM-L may omit timestamps; characters are emitted with t=0 as
    a fallback so the transcript is still usable downstream.
    """
    segments: list[dict] = []

    for item in results:
        text = item.get("text", "")
        timestamps = item.get("timestamp")

        if timestamps:
            for entry in timestamps:
                if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                    continue
                char, start, end = entry[0], entry[1], entry[2]
                char = str(char).strip()
                if not char:
                    continue
                segments.append(
                    {"word": char, "start": float(start), "end": float(end)}
                )
        elif text:
            chars = [c for c in text if c.strip()]
            if chars:
                logger.warning(
                    "FireRedASR: no timestamps in result — emitting %d chars with t=0",
                    len(chars),
                )
                for char in chars:
                    segments.append({"word": char, "start": 0.0, "end": 0.0})

    return segments
