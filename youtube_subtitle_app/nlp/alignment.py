import re


def _normalize(text):
    return re.sub(r"[^\w']", "", text.lower())


def _flatten_chunks(sentence_chunks):
    return [token["text"] for chunk in sentence_chunks for token in chunk]


def verify_word_order(word_segments, sentence_chunks, context=3):
    ws_words = [_normalize(ws["word"]) for ws in word_segments]
    sc_words = [_normalize(w) for w in _flatten_chunks(sentence_chunks)]

    if len(ws_words) != len(sc_words):
        print(
            f"[Verify] Length mismatch: word_segments={len(ws_words)}, chunks={len(sc_words)}"
        )

    for i, (w1, w2) in enumerate(zip(ws_words, sc_words)):
        if w1 != w2:
            print(f"[Mismatch at {i}]: '{w1}' != '{w2}'")
            raise ValueError("Word order mismatch")

    print("[Verify] Word order matches between ASR and NLP.")


def align_sentences_with_timestamps(word_segments, sentence_chunks):
    """
    Adds start/end timestamps to each sentence chunk based on aligned word_segments.
    Modifies each sentence_chunk (list of tokens) by adding:
    - sentence["start"]
    - sentence["end"]
    """
    import re

    def normalize(text):
        return re.sub(r"[^\w']+", "", text.lower())

    # Flatten sentence_chunks into a list of tokens
    flat_tokens = [token for chunk in sentence_chunks for token in chunk]
    ws_words = [normalize(w["word"]) for w in word_segments]
    tok_words = [normalize(t["text"]) for t in flat_tokens]

    # Verify length matches
    if len(ws_words) != len(tok_words):
        raise ValueError(
            f"[Align] Token count mismatch: ASR={len(ws_words)}, NLP={len(tok_words)}"
        )

    # Match confirmed: now assign timestamps
    for token, word in zip(flat_tokens, word_segments):
        token["start"] = int(word["start"] * 100) / 100
        token["end"] = int(word["end"] * 100) / 100

    return sentence_chunks


def refine_sentences_by_timing(
    sentence_chunks, min_duration=2.0, max_gap=1.0, force_split_if_gap=True
):
    """
    - Splits long sentences at word-level time gaps
    - Merges short chunks to previous
    """
    refined = []

    # Step 1: Split by time gap
    for chunk in sentence_chunks:
        current = [chunk[0]]

        for i in range(1, len(chunk)):
            gap = chunk[i]["start"] - chunk[i - 1]["end"]
            is_split = chunk[i]["is_punct"]

            if gap > max_gap and (is_split or force_split_if_gap):
                refined.append(current)
                current = [chunk[i]]
            else:
                current.append(chunk[i])

        if current:
            refined.append(current)

    # Step 2: Merge short-duration chunks
    merged = []
    for chunk in refined:
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        duration = end - start

        if duration < min_duration and merged:
            merged[-1].extend(chunk)
        else:
            merged.append(chunk)

    return merged
