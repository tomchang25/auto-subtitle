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
            hub="hf",
            vad_kwargs={"hub": "hf"},
            punc_kwargs={"hub": "hf"},
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

    # Key flags:
    # - pred_timestamp=True: tells the paraformer model to run
    #   ts_prediction_lfr6_standard() and include character-level timestamps
    #   in each result dict.  Without this, result only has {key, text}.
    # - sentence_timestamp=True: tells the VAD pipeline to POST-PROCESS
    #   timestamps into sentence_info via timestamp_sentence().  However,
    #   FunASR has a bug where it accesses result["timestamp"] without
    #   checking existence, causing KeyError if any VAD segment failed to
    #   produce timestamps.  So we do NOT pass sentence_timestamp and
    #   handle extraction ourselves in _extract_segments().
    try:
        res = model.generate(
            input=str(wav_path),
            batch_size_s=300,
            pred_timestamp=True,
        )
    except KeyError as exc:
        logger.warning(
            "FunASR KeyError during generate(): %s — retrying without pred_timestamp",
            exc,
        )
        res = model.generate(
            input=str(wav_path),
            batch_size_s=300,
        )

    if not res:
        raise ValueError("FunASR did not return any results")

    logger.debug("FunASR raw output keys: %s", [list(c.keys()) for c in res])
    if res:
        sample = res[0]
        logger.debug(
            "FunASR sample chunk: text=%r, timestamp=%r, " "sentence_info=%r, keys=%s",
            sample.get("text", "")[:100],
            sample.get("timestamp", [])[:5],
            str(sample.get("sentence_info", []))[:200],
            list(sample.keys()),
        )

    segments = _extract_segments(res)

    if not segments:
        raise ValueError("FunASR did not return character-level timestamps")

    detected_lang = "zh"
    logger.info(
        "FunASR transcription complete: %d characters, lang=%s",
        len(segments),
        detected_lang,
    )
    return segments, detected_lang


def _extract_segments(res: list[dict]) -> list[dict]:
    """Extract character-level segments from FunASR output.

    Handles all known FunASR output shapes robustly:

    1. Top-level ``timestamp`` — character-level timestamps from the ASR model.
       Match chars to timestamps; if counts differ, pair as many as possible.
    2. ``sentence_info`` — per-sentence dicts that may carry their own
       ``timestamp`` list or just ``start``/``end`` boundaries.
    3. Text-only fallback — if a chunk has text but no usable timing at all,
       we still emit the characters with ``start=end=0`` so downstream
       code can at least see the transcript (and warn).
    """
    segments: list[dict] = []

    for chunk in res:
        text = chunk.get("text", "")
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
            # If we got at least some timestamps from this chunk, move on
            if pairs > 0:
                continue

        # ── Strategy 2: sentence_info (from sentence_timestamp mode) ──
        sentence_info = chunk.get("sentence_info")
        if sentence_info:
            handled = False
            for sent in sentence_info:
                sent_text = sent.get("text", "")
                if not sent_text or not sent_text.strip():
                    continue
                chars = [c for c in sent_text if c.strip()]
                if not chars:
                    continue

                sent_ts = sent.get("timestamp")
                if sent_ts and isinstance(sent_ts, list):
                    # Per-character timestamps within this sentence
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
                    # No per-char timestamps — use sentence boundaries
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

        # ── Strategy 3: text exists but no timestamps at all ──
        # Emit characters with zero timing so the transcript is not lost.
        # Downstream grouping will still work; timing will be inaccurate.
        chars = [c for c in text if c.strip()]
        if chars:
            logger.warning(
                "FunASR chunk has text (%d chars) but no usable timestamps — "
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
):
    """Evenly distribute characters across a time range."""
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
