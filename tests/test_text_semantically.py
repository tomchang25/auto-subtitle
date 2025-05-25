from youtube_subtitle_app.nlp.text_semantically import split_to_sentences
from pathlib import Path
import json


def test_nlp_with_contractions_and_hyphenated_compounds():
    test_text = (
        "It's a bright day, isn't it? "
        "We crossed the multi-line bridge. "
        "I'm sure they're waiting at the high-speed train station! "
        "Yes, that's exactly what happened."
    )

    print(test_text)
    chunks = split_to_sentences(test_text)

    assert isinstance(chunks, list)
    all_tokens = [token["text"] for chunk in chunks for token in chunk]

    # Check contractions
    assert any("it's" in w.lower() for w in all_tokens)
    assert any("isn't" in w.lower() for w in all_tokens)
    assert any("they're" in w.lower() for w in all_tokens)

    # Check compound hyphenated words
    assert any("multi-line" in w.lower() for w in all_tokens)
    assert any("high-speed" in w.lower() for w in all_tokens)

    # Check punctuation is attached correctly
    assert any(w.endswith(("?", ".", "!")) for w in all_tokens)


def load_word_segments(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_split_to_sentences():
    # Load ASR word segments
    segments_path = Path("tests/data/news_word_segments.json")
    word_segments = load_word_segments(segments_path)

    # Reconstruct the sentence
    sentence = " ".join(word["word"] for word in word_segments)

    # Run the sentence splitter
    chunks = split_to_sentences(sentence)

    # Assertions
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, list) for chunk in chunks)
    assert all(
        "text" in token and "whitespace" in token for chunk in chunks for token in chunk
    )

    # Optional: print or check chunk content
    for chunk in chunks:
        words = "".join(t["text"] + t["whitespace"] for t in chunk).strip()
        print(f"> {words}")


test_split_to_sentences()
