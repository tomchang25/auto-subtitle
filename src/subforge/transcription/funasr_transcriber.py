import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("paraformer-zh",)
DEFAULT_MODEL = "paraformer-zh"

_loaded_models = {}


def load_model(model_name: str):
    if model_name not in SUPPORTED_MODELS:
        logger.warning(
            "FunASR: unrecognized model %r, falling back to %r",
            model_name,
            DEFAULT_MODEL,
        )
        model_name = DEFAULT_MODEL
    if model_name not in _loaded_models:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ImportError(
                "FunASR is not installed. Install with: pip install subforge[funasr]"
            ) from exc
        logger.info("Loading FunASR model: %s", model_name)
        _loaded_models[model_name] = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
        )
    return _loaded_models[model_name]


def transcribe_audio_word_level(
    wav_path: Path,
    model_name: str,
    progress_callback=None,
) -> tuple[list, str]:
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {wav_path}")

    if progress_callback:
        progress_callback("Transcribe", "Loading FunASR model…")

    model = load_model(model_name)

    logger.info("Transcribing with FunASR (%s): %s", model_name, wav_path)

    if progress_callback:
        progress_callback("Transcribe", "Running FunASR inference…")

    res = model.generate(
        input=str(wav_path),
        batch_size_s=300,
        return_raw_text=True,
        is_final=True,
        timestamp_offset_mode="abs",
    )

    if not res:
        raise ValueError("FunASR did not return any results")

    segments = []
    for chunk in res:
        text = chunk.get("text", "")
        timestamps = chunk.get("timestamp", [])
        if not text or not timestamps:
            continue
        for char, ts in zip(text, timestamps):
            if not char.strip():
                continue
            start_ms, end_ms = ts
            segments.append(
                {
                    "word": char,
                    "start": start_ms / 1000.0,
                    "end": end_ms / 1000.0,
                }
            )

    if not segments:
        raise ValueError("FunASR did not return character-level timestamps")

    detected_lang = "zh"
    logger.info(
        "FunASR transcription complete: %d characters, lang=%s",
        len(segments),
        detected_lang,
    )
    return segments, detected_lang
