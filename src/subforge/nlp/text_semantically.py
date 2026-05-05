import spacy
import functools

BREAK_WORD = ["so", "but"]


@functools.lru_cache(maxsize=1)
def _get_nlp_model():
    return spacy.load("en_core_web_sm")


def _merge_tokens(tokens):
    """
    Merges:
    - Contractions (e.g., "it", "'s" → "it's")
    - Glued compound words with no whitespace (e.g., "multi", "line" → "multiline")
    """
    merged = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        text = token["text"]
        whitespace = token["whitespace"]
        is_punct = token["is_punct"]

        j = i
        while j + 1 < len(tokens) and (
            tokens[j]["whitespace"] == ""
            or tokens[j + 1]["text"]
            in [
                "'s",
                "'re",
                "'ve",
                "'d",
                "'ll",
                "n't",
            ]
        ):
            text += tokens[j + 1]["text"]
            whitespace = tokens[j + 1]["whitespace"]
            is_punct = tokens[j + 1]["is_punct"]
            j += 1
        i = j

        merged.append(
            {
                "text": text,
                "whitespace": whitespace,
                "is_punct": is_punct,
            }
        )
        i += 1

    return merged


def split_to_sentences(text: str, punct_limit: int = 5):
    """
    Split a sentence based on punctuation breaks and soft word limits.
    No hard max word enforcement anymore.
    """
    doc = _get_nlp_model()(text)
    tokens = [
        {"text": t.text, "whitespace": t.whitespace_, "is_punct": t.is_punct}
        for t in doc
    ]
    tokens = _merge_tokens(tokens)

    chunks = []
    current = []
    count = 0

    for tok in tokens:
        current.append(
            {
                "text": tok["text"],
                "whitespace": tok["whitespace"],
                "is_punct": tok["is_punct"],
            }
        )
        count += 1

        if count >= punct_limit and tok["is_punct"]:
            chunks.append(current)
            current = []
            count = 0

    if current:
        chunks.append(current)

    return chunks
