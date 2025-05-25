from pathlib import Path
import json
from youtube_subtitle_app.nlp.text_semantically import split_to_sentences
from youtube_subtitle_app.nlp.alignment import refine_chunks_by_time


def load_word_segments(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_refine_chunks_by_time():
    # Load test ASR data
    word_segments = load_word_segments(Path("tests/data/news_word_segments.json"))

    # Rebuild plain text from word segments
    full_text = " ".join(w["word"] for w in word_segments)

    # Convert to sentence chunks
    sentence_chunks = split_to_sentences(full_text)

    # Run alignment
    refined = refine_chunks_by_time(word_segments, sentence_chunks)

    # Assert structure
    assert isinstance(refined, list)
    for chunk in refined:
        assert "start" in chunk
        assert "end" in chunk
        assert "segment" in chunk
        assert isinstance(chunk["segment"], str)

    # Optional: print debug output
    for chunk in refined:
        print(f"[{chunk['start']:.2f} - {chunk['end']:.2f}] {chunk['segment']}")


test_refine_chunks_by_time()
