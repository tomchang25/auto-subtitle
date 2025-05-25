import json
import pathlib as Path


def save_word_segments(word_segments, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(word_segments, f, indent=2)
