import json
from pathlib import Path


def save_word_segments(word_segments, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(word_segments, f, indent=2)


def get_bounds_and_text(chunks):
    return [
        {
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "segment": " ".join(token["text"] for token in chunk),
        }
        for chunk in chunks
    ]


def save_to_json(bounds, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bounds, f, ensure_ascii=False, indent=2)

    print(f"[✅] Saved bounds to: {output_path}")
