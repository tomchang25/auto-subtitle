"""Punctuation restoration using a local token-classification model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subforge.nlp.lang_profile import LanguageProfile

logger = logging.getLogger(__name__)

# Model that predicts punctuation after each token
_MODEL_NAME = "oliverguhr/fullstop-punctuation-multilang-large"

# Map model labels to punctuation characters
_LABEL_MAP = {
    "0": "",
    "O": "",
    ".": ".",
    ",": ",",
    "?": "?",
    "-": ",",
    ":": ",",
}

_DEFAULT_PUNCT_CHARS = set(".,!?;:")


def restore_punctuation(
    word_segments: list[dict],
    profile: LanguageProfile | None = None,
) -> list[dict]:
    """Add punctuation to word_segments using a local transformer model.

    Only adds punctuation where Whisper didn't already provide it.
    Preserves existing Whisper punctuation (which is generally more accurate
    for the specific audio context).
    """
    if not word_segments:
        return word_segments

    from transformers import pipeline

    logger.info("Loading punctuation model: %s", _MODEL_NAME)
    pipe = pipeline(
        "token-classification",
        model=_MODEL_NAME,
        aggregation_strategy="none",
    )

    # Build the punctuation character set from profile
    punct_chars = _DEFAULT_PUNCT_CHARS
    if profile is not None:
        punct_chars = set(profile.punctuation) | _DEFAULT_PUNCT_CHARS

    # Identify which words already have punctuation from Whisper
    has_punct = []
    clean_words = []
    for seg in word_segments:
        word = seg["word"]
        if word and word[-1] in punct_chars:
            has_punct.append(True)
            stripped = word
            while stripped and stripped[-1] in punct_chars:
                stripped = stripped[:-1]
            clean_words.append(stripped if stripped else word)
        else:
            has_punct.append(False)
            clean_words.append(word)

    # Process in chunks
    chunk_size = 200
    total = len(clean_words)
    all_labels = []

    for i in range(0, total, chunk_size):
        chunk = clean_words[i : i + chunk_size]
        text = " ".join(chunk)

        predictions = pipe(text)

        # Track word boundaries via character offsets
        word_boundaries = []
        pos = 0
        for w in chunk:
            start = text.index(w, pos)
            end = start + len(w)
            word_boundaries.append((start, end))
            pos = end

        # For each word, take the label of the last subword token
        chunk_labels = [""] * len(chunk)
        for pred in predictions:
            pred_start = pred["start"]
            pred_end = pred["end"]
            label = pred["entity"].split("-")[-1] if "-" in pred["entity"] else pred["entity"]

            for widx, (ws, we) in enumerate(word_boundaries):
                if pred_start >= ws and pred_end <= we:
                    chunk_labels[widx] = _LABEL_MAP.get(label, "")
                    break
                elif pred_start >= ws and pred_start < we:
                    chunk_labels[widx] = _LABEL_MAP.get(label, "")
                    break

        all_labels.extend(chunk_labels)

    # Apply: keep Whisper punctuation where it exists, add model's where it doesn't
    added = 0
    kept = 0
    for seg, clean, label, already_has in zip(word_segments, clean_words, all_labels, has_punct):
        if already_has:
            # Keep Whisper's original punctuation (don't touch)
            kept += 1
        elif label:
            # Whisper didn't add punctuation here, but model suggests one
            seg["word"] = clean + label
            added += 1
        # else: neither has punctuation, leave as-is

    logger.info(
        "Punctuation restoration done: kept %d (Whisper), added %d (model), "
        "%d unchanged, %d total",
        kept, added, total - kept - added, total,
    )
    return word_segments
