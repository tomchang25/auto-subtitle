import logging
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
)

_loaded_models = {}


def load_model(model_name: str) -> WhisperModel:
    if model_name not in _loaded_models:
        logger.info("Loading faster-whisper model: %s", model_name)
        _loaded_models[model_name] = WhisperModel(model_name)
    return _loaded_models[model_name]


def transcribe_audio_word_level(
    wav_path: Path,
    model_name: str,
    progress_callback=None,
) -> tuple[list, str]:
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {wav_path}")

    model = load_model(model_name)

    logger.info("Transcribing with faster-whisper: %s", wav_path)
    segments_iter, info = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        vad_filter=True,
    )

    duration = info.duration  # total audio duration in seconds
    detected_lang = info.language  # ISO 639-1 code, e.g. "en", "zh", "ja"
    logger.info(
        "Detected language: %s (probability %.2f)",
        detected_lang,
        info.language_probability,
    )
    segments = []
    last_reported = 0

    for segment in segments_iter:
        if not segment.words:
            continue
        for w in segment.words:
            text = w.word.strip()
            if not text:
                continue
            segments.append(
                {"word": text, "start": float(w.start), "end": float(w.end)}
            )

        # Report progress every 60 seconds of audio processed
        current_time = segment.end
        if duration > 0 and current_time - last_reported >= 60:
            pct = min(100, int(current_time / duration * 100))
            logger.info(
                "Transcription progress: %d%% (%d/%ds, %d words so far)",
                pct,
                int(current_time),
                int(duration),
                len(segments),
            )
            if progress_callback:
                progress_callback(
                    "Transcribe",
                    f"{pct}% ({int(current_time)}/{int(duration)}s)",
                )
            last_reported = current_time

    if not segments:
        raise ValueError("faster-whisper did not return word-level timestamps")

    logger.info(
        "Transcription complete: %d words, lang=%s", len(segments), detected_lang
    )
    return segments, detected_lang
