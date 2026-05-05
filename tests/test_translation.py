from pathlib import Path
import json
from subforge.translation.translator import (
    translate_subtitles,
)


def test_translation_news_subtitles():
    input_path = Path("tests/data/news_refined.json")

    # Load English subtitle segments
    with open(input_path, "r", encoding="utf-8") as f:
        subtitles = json.load(f)

    # Translate
    # translated = translate_subtitles(subtitles)
    translated = translate_subtitles(subtitles, method="qwen")

    # Print each original + translated pair
    print("\n📘 Translated Subtitle Preview:")
    for chunk in translated:
        print(f"\nEN: {chunk['segment']}\nZH: {chunk['translation']}")


test_translation_news_subtitles()
