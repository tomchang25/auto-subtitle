from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("iic/SenseVoiceSmall",)
DEFAULT_MODEL = "iic/SenseVoiceSmall"

# SenseVoice prefixes each utterance with emotion / language / event tokens,
# e.g. "<|zh|><|NEUTRAL|><|Speech|><|woitn|>".  Strip all before processing.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")

_loaded_models: dict = {}


def load_model(model_name: str):
    if model_name not in SUPPORTED_MODELS:
        logger.warning(
            "SenseVoice: unrecognized model %r, falling back to %r",
            model_name,
            DEFAULT_MODEL,
        )
        model_name = DEFAULT_MODEL
    if model_name not in _loaded_models:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ImportError(
                "SenseVoice requires FunASR. Install with: pip install subforge[funasr]"
            ) from exc
        logger.info("Loading SenseVoice model: %s", model_name)
        _loaded_models[model_name] = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            hub="hf",
            vad_kwargs={"hub": "hf"},
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
        progress_callback("Transcribe", "Loading SenseVoice model…")

    model = load_model(model_name)

    logger.info("Transcribing with SenseVoice (%s): %s", model_name, wav_path)

    if progress_callback:
        progress_callback("Transcribe", "Running SenseVoice inference…")

    try:
        res = model.generate(
            input=str(wav_path),
            batch_size_s=300,
            pred_timestamp=True,
        )
    except KeyError as exc:
        logger.warning(
            "SenseVoice KeyError during generate(): %s — retrying without pred_timestamp",
            exc,
        )
        res = model.generate(
            input=str(wav_path),
            batch_size_s=300,
        )

    if not res:
        raise ValueError("SenseVoice did not return any results")

    logger.debug("SenseVoice raw output keys: %s", [list(c.keys()) for c in res])
    if res:
        sample = res[0]
        logger.debug(
            "SenseVoice sample chunk: text=%r, timestamp=%r, keys=%s",
            sample.get("text", "")[:100],
            sample.get("timestamp", [])[:5],
            list(sample.keys()),
        )

    segments = _extract_segments(res)

    if not segments:
        raise ValueError("SenseVoice did not return character-level timestamps")

    detected_lang = "zh"
    logger.info(
        "SenseVoice transcription complete: %d characters, backend=sensevoice, "
        "model=%s, lang=%s, has_timestamps=%s",
        len(segments),
        model_name,
        detected_lang,
        any(s["start"] > 0 or s["end"] > 0 for s in segments),
    )
    return segments, detected_lang


def _strip_special_tokens(text: str) -> str:
    return _SPECIAL_TOKEN_RE.sub("", text)


def _extract_segments(res: list[dict]) -> list[dict]:
    """Extract character-level segments from SenseVoice output.

    Structurally identical to the FunASR Paraformer extractor but strips
    SenseVoice-specific special tokens (emotion / language / event tags) from
    text before pairing characters with timestamps.

    Timestamp values are in milliseconds (same as Paraformer).
    """
    segments: list[dict] = []

    for chunk in res:
        raw_text = chunk.get("text", "")
        text = _strip_special_tokens(raw_text)
        if not text or not text.strip():
            continue

        # ── Strategy 1: top-level character timestamps ──
        timestamps = chunk.get("timestamp")
        if timestamps:
            chars = [c for c in text if c.strip()]
            pairs = min(len(chars), len(timestamps))
            for i in range(pairs):
                ts = timestamps[i]
                if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                    continue
                segments.append(
                    {
                        "word": chars[i],
                        "start": ts[0] / 1000.0,
                        "end": ts[1] / 1000.0,
                    }
                )
            if pairs > 0:
                continue

        # ── Strategy 2: sentence_info ──
        sentence_info = chunk.get("sentence_info")
        if sentence_info:
            handled = False
            for sent in sentence_info:
                sent_text = _strip_special_tokens(sent.get("text", ""))
                if not sent_text or not sent_text.strip():
                    continue
                chars = [c for c in sent_text if c.strip()]
                if not chars:
                    continue

                sent_ts = sent.get("timestamp")
                if sent_ts and isinstance(sent_ts, list):
                    pairs = min(len(chars), len(sent_ts))
                    for i in range(pairs):
                        ts = sent_ts[i]
                        if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                            continue
                        segments.append(
                            {
                                "word": chars[i],
                                "start": ts[0] / 1000.0,
                                "end": ts[1] / 1000.0,
                            }
                        )
                    handled = True
                else:
                    start_ms = sent.get("start", 0)
                    end_ms = sent.get("end", 0)
                    if isinstance(start_ms, (int, float)) and isinstance(
                        end_ms, (int, float)
                    ):
                        if start_ms != end_ms:
                            _distribute_chars(segments, chars, start_ms, end_ms)
                            handled = True
            if handled:
                continue

        # ── Strategy 3: text only ──
        chars = [c for c in text if c.strip()]
        if chars:
            logger.warning(
                "SenseVoice chunk has text (%d chars) but no usable timestamps — "
                "emitting with t=0. First 40 chars: %r",
                len(chars),
                text[:40],
            )
            for char in chars:
                segments.append({"word": char, "start": 0.0, "end": 0.0})

    return segments


def _distribute_chars(
    segments: list[dict],
    chars: list[str],
    start_ms: float,
    end_ms: float,
) -> None:
    n = len(chars)
    duration = end_ms - start_ms
    for i, char in enumerate(chars):
        c_start = start_ms + duration * i / n
        c_end = start_ms + duration * (i + 1) / n
        segments.append(
            {
                "word": char,
                "start": c_start / 1000.0,
                "end": c_end / 1000.0,
            }
        )
