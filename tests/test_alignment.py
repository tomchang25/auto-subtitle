from pathlib import Path
import json
from subforge.nlp.text_semantically import split_to_sentences
from subforge.nlp.alignment import (
    align_sentences_with_timestamps,
    refine_sentences_by_timing,
)
from subforge.nlp.segmentation import split_long_sentences_by_length
from subforge.utils import get_bounds_and_text, save_to_json


def load_word_segments(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_align_sentences_with_timestamps():
    # Load word-level segments
    word_segments = load_word_segments(Path("tests/data/news_word_segments.json"))

    # Rebuild text and split into sentence chunks
    full_text = " ".join(w["word"] for w in word_segments)
    sentence_chunks = split_to_sentences(full_text)

    # Align and add timestamps to sentence chunks
    aligned_chunks = align_sentences_with_timestamps(word_segments, sentence_chunks)

    # Assertions
    assert isinstance(aligned_chunks, list)
    assert all(isinstance(chunk, list) for chunk in aligned_chunks)
    assert len(aligned_chunks) == len(sentence_chunks)

    total_tokens = 0

    for chunk in aligned_chunks:
        for token in chunk:
            assert "text" in token
            assert "start" in token
            assert "end" in token
            assert isinstance(token["start"], float)
            assert isinstance(token["end"], float)
            total_tokens += 1

    # Ensure alignment was complete
    assert total_tokens == len(word_segments)

    # Optional: preview a few aligned tokens
    for chunk in aligned_chunks:
        text = " ".join(t["text"] for t in chunk)
        print(f"[{chunk[0]['start']:.2f} - {chunk[-1]['end']:.2f}] {text}")


def test_refine_sentences_by_timing():
    word_segments = load_word_segments(Path("tests/data/news_word_segments.json"))
    full_text = " ".join(w["word"] for w in word_segments)

    # Step 1: NLP full-sentence split
    sentence_chunks = split_to_sentences(full_text)

    # Step 2: Timestamp alignment
    aligned = align_sentences_with_timestamps(word_segments, sentence_chunks)
    original_bounds = get_bounds_and_text(aligned)

    # Step 3: Refinement
    refined = refine_sentences_by_timing(aligned)
    refined_bounds = get_bounds_and_text(refined)

    print(f"\n[INFO] Original: {len(original_bounds)} sentences")
    print(f"[INFO] Refined : {len(refined_bounds)} chunks")

    # 🔪 Show splits
    print("\n🔪 Split sentences:")
    for orig in original_bounds:
        orig_start, orig_end, orig_text = orig["start"], orig["end"], orig["segment"]
        matching_refined = [
            r
            for r in refined_bounds
            if orig_start <= r["start"] and r["end"] <= orig_end
        ]
        if len(matching_refined) > 1:
            print(
                f"\nOriginal: [{orig_start:.2f} - {orig_end:.2f}] {orig_text}"
            )
            for r in matching_refined:
                print(f"  → [{r['start']:.2f} - {r['end']:.2f}] {r['segment']}")

    # 🧩 Show merges
    print("\n🧩 Merged sentences:")
    for new in refined_bounds:
        new_start, new_end, new_text = new["start"], new["end"], new["segment"]
        matching_original = [
            o
            for o in original_bounds
            if o["start"] >= new_start and o["end"] <= new_end
        ]
        if len(matching_original) > 1:
            print(f"\nRefined: [{new_start:.2f} - {new_end:.2f}] {new_text}")
            for o in matching_original:
                print(f"  ← [{o['start']:.2f} - {o['end']:.2f}] {o['segment']}")

    # Structural checks
    assert isinstance(refined, list)
    for chunk in refined:
        assert isinstance(chunk, list)
        assert all("text" in t and "start" in t and "end" in t for t in chunk)

    # Preview
    print("\n📝 Preview:")
    for chunk in refined:
        text = " ".join(t["text"] for t in chunk)
        print(f"[{chunk[0]['start']:.2f} - {chunk[-1]['end']:.2f}] {text}")


def test_split_long_sentences_by_length():
    word_segments = load_word_segments(Path("tests/data/news_word_segments.json"))
    full_text = " ".join(w["word"] for w in word_segments)

    # Step 1: NLP full-sentence split
    sentence_chunks = split_to_sentences(full_text)

    # Step 2: Timestamp alignment
    aligned = align_sentences_with_timestamps(word_segments, sentence_chunks)

    # Step 3: Refinement
    refined = refine_sentences_by_timing(aligned)

    # Step 4: Split long sentences by word length
    max_words = 15
    min_words = 8
    split_chunks = split_long_sentences_by_length(
        refined, min_words=min_words, max_words=max_words
    )

    print(f"\n[INFO] Refined sentences: {len(refined)}")
    print(f"[INFO] After split      : {len(split_chunks)}")

    # Check constraints
    for chunk in split_chunks:
        assert len(chunk) < max_words + min_words
        assert all("start" in tok and "end" in tok for tok in chunk)

    # Report actual splits
    print("\n📏 Split sentences (word count > max_words):")
    for chunk in refined:
        if len(chunk) > max_words:
            print(
                f"\nOriginal ({len(chunk)} words): "
                + " ".join(t["text"] for t in chunk)
            )
            new_subchunks = split_long_sentences_by_length(
                [chunk], min_words, max_words
            )
            for i, sub in enumerate(new_subchunks):
                print(
                    f"  → Chunk {i+1} ({len(sub)} words): "
                    + " ".join(t["text"] for t in sub)
                )


if __name__ == "__main__":
    # test_align_sentences_with_timestamps()
    # test_refine_sentences_by_timing()
    test_split_long_sentences_by_length()
