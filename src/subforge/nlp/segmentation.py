# Words that signal a natural sentence boundary in spoken English
_BREAK_WORDS = {"but", "and", "so", "because", "like", "then", "right", "now", "anyway", "although", "however"}


def _cut_strength(token, next_token, pause_threshold):
    """Return cut strength: 'strong', 'weak', or None.

    Strong (triggers at min_words): punctuation (. ! ? , ;) or timing pause.
    Weak (triggers at soft_words): next token is a break word.
    """
    text = token.get("text", "")
    if text and text[-1] in ",.!?;":
        return "strong"

    if next_token and pause_threshold > 0:
        t_end = token.get("end")
        n_start = next_token.get("start")
        if t_end is not None and n_start is not None:
            if n_start - t_end >= pause_threshold:
                return "strong"

    # Next token is a break word (cut BEFORE it)
    if next_token:
        next_text = next_token.get("text", "").lower().rstrip(",.!?;")
        if next_text in _BREAK_WORDS:
            return "weak"

    return None


def split_long_sentences_by_length(
    sentence_chunks,
    min_words=4,
    max_words=15,
    soft_words=8,
    pause_threshold=0.25,
):
    """Split chunks with min/soft/hard thresholds."""
    new_chunks = []

    for chunk in sentence_chunks:
        length = len(chunk)
        if length <= min_words:
            new_chunks.append(chunk)
            continue

        current = []
        for i, token in enumerate(chunk):
            current.append(token)
            word_count = len(current)

            if word_count < min_words:
                continue

            tokens_left = length - (i + 1)

            # Hard cut
            if word_count >= max_words:
                if 0 < tokens_left < min_words:
                    current.extend(chunk[i + 1:])
                    break
                new_chunks.append(current)
                current = []
                continue

            # Check cut strength
            next_token = chunk[i + 1] if i + 1 < length else None
            strength = _cut_strength(token, next_token, pause_threshold)

            if strength == "strong" and word_count >= min_words:
                if tokens_left >= min_words or tokens_left == 0:
                    new_chunks.append(current)
                    current = []
                    continue

            if strength == "weak" and word_count >= soft_words:
                if tokens_left >= min_words or tokens_left == 0:
                    new_chunks.append(current)
                    current = []
                    continue

        if current:
            if len(current) < min_words and new_chunks:
                new_chunks[-1].extend(current)
            else:
                new_chunks.append(current)

    return new_chunks


def merge_short_segments(
    sentence_chunks,
    max_words=15,
    max_duration=4.0,
    max_gap=1.0,
):
    """Merge consecutive short segments if the result stays within limits.

    Only merges if:
      - Combined word count <= max_words
      - Combined duration (start of first to end of last) <= max_duration
      - Gap between segments <= max_gap
    """
    if not sentence_chunks:
        return []

    merged = [sentence_chunks[0]]

    for chunk in sentence_chunks[1:]:
        prev = merged[-1]
        combined_words = len(prev) + len(chunk)

        # Don't merge across sentence boundaries (prev ends with . ! ?)
        prev_last_text = prev[-1].get("text", "")
        if prev_last_text and prev_last_text[-1] in ".!?":
            merged.append(chunk)
            continue

        # Check gap between segments
        prev_end = prev[-1].get("end", 0)
        chunk_start = chunk[0].get("start", 0)
        gap = chunk_start - prev_end

        # Check combined duration
        combined_start = prev[0].get("start", 0)
        combined_end = chunk[-1].get("end", 0)
        combined_duration = combined_end - combined_start

        if (combined_words <= max_words
                and combined_duration <= max_duration
                and gap <= max_gap):
            merged[-1] = prev + chunk
        else:
            merged.append(chunk)

    return merged
