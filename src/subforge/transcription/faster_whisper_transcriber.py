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


def transcribe_audio_word_level(wav_path: Path, model_name: str) -> list:
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {wav_path}")

    model = load_model(model_name)

    logger.info("Transcribing with faster-whisper: %s", wav_path)
    segments_iter, _info = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        vad_filter=True,
    )

    segments = []
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

    if not segments:
        raise ValueError("faster-whisper did not return word-level timestamps")

    logger.info("Transcription complete: %d words", len(segments))
    return segments
