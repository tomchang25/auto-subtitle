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


def refine_chunks_by_time(
    word_segments,
    sentence_chunks,
    max_gap=1.5,
    min_duration=1.5,
    force_split_if_gap=True,
):
    verify_word_order(word_segments, sentence_chunks)

    refined_chunks = []
    word_index = 0

    for chunk in sentence_chunks:
        matched = word_segments[word_index : word_index + len(chunk)]
        word_index += len(chunk)

        sub_chunks = []
        current = [matched[0]]

        for i in range(1, len(matched)):
            gap = matched[i]["start"] - matched[i - 1]["end"]
            is_split = chunk[i]["is_punct"] or chunk[i]["is_break"]

            if gap > max_gap and (is_split or force_split_if_gap):
                sub_chunks.append(current)
                current = [matched[i]]
            else:
                current.append(matched[i])

        if current:
            sub_chunks.append(current)

        refined_chunks.extend(sub_chunks)

    # Merge short segments
    merged_chunks = []
    for seg in refined_chunks:
        start = seg[0]["start"]
        end = seg[-1]["end"]
        duration = end - start
        if duration < min_duration and merged_chunks:
            merged_chunks[-1].extend(seg)
        else:
            merged_chunks.append(seg)

    return [
        {
            "segment": " ".join([w["word"] for w in chunk]),
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
        }
        for chunk in merged_chunks
    ]
