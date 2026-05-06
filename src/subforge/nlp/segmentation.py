from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subforge.nlp.lang_profile import LanguageProfile

# Fallback English values used when no profile is provided
_BREAK_WORDS = {
    "but",
    "and",
    "so",
    "because",
    "like",
    "then",
    "right",
    "now",
    "anyway",
    "although",
    "however",
}
_PUNCT = set(",.!?;:")
_SENTENCE_END = set(".!?")


def _get_break_words(profile: LanguageProfile | None) -> set[str]:
    if profile is not None:
        return set(profile.break_words)
    return _BREAK_WORDS


def _get_punctuation(profile: LanguageProfile | None) -> set[str]:
    if profile is not None:
        return set(profile.punctuation)
    return _PUNCT


def _get_sentence_end(profile: LanguageProfile | None) -> set[str]:
    if profile is not None:
        return set(profile.sentence_end)
    return _SENTENCE_END


def _use_char_count(profile: LanguageProfile | None) -> bool:
    if profile is not None:
        return profile.use_char_count
    return False


def _measure(token, use_chars: bool) -> int:
    if use_chars:
        return len(token.get("text", ""))
    return 1


def _measure_chunk(chunk, use_chars: bool) -> int:
    return sum(_measure(tok, use_chars) for tok in chunk)


def _cut_strength(token, next_token, pause_threshold, profile=None):
    """Return cut strength: 'strong', 'weak', or None.

    Strong (triggers at min_words): punctuation (. ! ? , ;) or timing pause.
    Weak (triggers at soft_words): next token is a break word.
    """
    punct = _get_punctuation(profile)
    break_words = _get_break_words(profile)

    text = token.get("text", "")
    if text and text[-1] in punct:
        return "strong"

    if next_token and pause_threshold > 0:
        t_end = token.get("end")
        n_start = next_token.get("start")
        if t_end is not None and n_start is not None:
            if n_start - t_end >= pause_threshold:
                return "strong"

    if next_token:
        next_text = next_token.get("text", "").lower()
        for ch in list(punct):
            next_text = next_text.rstrip(ch)
        if next_text in break_words:
            return "weak"

    return None


def split_long_sentences_by_length(
    sentence_chunks,
    min_words=4,
    max_words=15,
    soft_words=8,
    pause_threshold=0.25,
    profile: LanguageProfile | None = None,
):
    """Split chunks with min/soft/hard thresholds."""
    use_chars = _use_char_count(profile)
    new_chunks = []

    for chunk in sentence_chunks:
        chunk_size = _measure_chunk(chunk, use_chars)
        if chunk_size <= min_words:
            new_chunks.append(chunk)
            continue

        current = []
        current_size = 0
        for i, token in enumerate(chunk):
            current.append(token)
            current_size += _measure(token, use_chars)

            if current_size < min_words:
                continue

            remaining = chunk[i + 1 :]
            remaining_size = _measure_chunk(remaining, use_chars)

            # Hard cut
            if current_size >= max_words:
                if 0 < remaining_size < min_words:
                    current.extend(remaining)
                    break
                new_chunks.append(current)
                current = []
                current_size = 0
                continue

            # Check cut strength
            next_token = chunk[i + 1] if i + 1 < len(chunk) else None
            strength = _cut_strength(token, next_token, pause_threshold, profile)

            if strength == "strong" and current_size >= min_words:
                if remaining_size >= min_words or remaining_size == 0:
                    new_chunks.append(current)
                    current = []
                    current_size = 0
                    continue

            if strength == "weak" and current_size >= soft_words:
                if remaining_size >= min_words or remaining_size == 0:
                    new_chunks.append(current)
                    current = []
                    current_size = 0
                    continue

        if current:
            if _measure_chunk(current, use_chars) < min_words and new_chunks:
                new_chunks[-1].extend(current)
            else:
                new_chunks.append(current)

    return new_chunks


def merge_short_segments(
    sentence_chunks,
    max_words=15,
    max_duration=4.0,
    max_gap=1.0,
    profile: LanguageProfile | None = None,
):
    """Merge consecutive short segments if the result stays within limits.

    Only merges if:
      - Combined word count <= max_words
      - Combined duration (start of first to end of last) <= max_duration
      - Gap between segments <= max_gap
    """
    if not sentence_chunks:
        return []

    use_chars = _use_char_count(profile)
    sentence_end = _get_sentence_end(profile)
    merged = [sentence_chunks[0]]

    for chunk in sentence_chunks[1:]:
        prev = merged[-1]
        combined_size = _measure_chunk(prev, use_chars) + _measure_chunk(
            chunk, use_chars
        )

        prev_last_text = prev[-1].get("text", "")
        if prev_last_text and prev_last_text[-1] in sentence_end:
            merged.append(chunk)
            continue

        prev_end = prev[-1].get("end", 0)
        chunk_start = chunk[0].get("start", 0)
        gap = chunk_start - prev_end

        combined_start = prev[0].get("start", 0)
        combined_end = chunk[-1].get("end", 0)
        combined_duration = combined_end - combined_start

        if (
            combined_size <= max_words
            and combined_duration <= max_duration
            and gap <= max_gap
        ):
            merged[-1] = prev + chunk
        else:
            merged.append(chunk)

    return merged
