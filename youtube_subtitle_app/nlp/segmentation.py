def split_long_sentences_by_length(sentence_chunks, min_words=8, max_words=25):
    """
    Split any sentence into smaller chunks if it's too long (by token count).
    Ignores splits that would leave a final chunk shorter than min_words.
    """
    new_chunks = []

    for chunk in sentence_chunks:
        length = len(chunk)
        if length <= max_words:
            new_chunks.append(chunk)
            continue

        i = 0
        while i < length:
            end = i + max_words
            sub_chunk = chunk[i:end]

            # If this is the last chunk and too short, merge back
            if end >= length and len(sub_chunk) < min_words:
                new_chunks[-1].extend(sub_chunk)
                break

            new_chunks.append(sub_chunk)
            i += max_words

    return new_chunks
