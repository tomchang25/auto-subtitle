from __future__ import annotations

from typing import List

from subforge.translation.base import SubtitleChunk, TranslatedChunk


class MarianTranslator:
    MODEL_NAME = "Helsinki-NLP/opus-mt-en-zh"

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import MarianMTModel, MarianTokenizer
            self._tokenizer = MarianTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = MarianMTModel.from_pretrained(self.MODEL_NAME)

    def translate(self, chunks: List[SubtitleChunk]) -> List[TranslatedChunk]:
        self._load()
        texts = [c["segment"] for c in chunks]
        all_translations: list[str] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )
            outputs = self._model.generate(**inputs)
            all_translations.extend(
                self._tokenizer.decode(t, skip_special_tokens=True) for t in outputs
            )
        return [{**c, "translation": zh} for c, zh in zip(chunks, all_translations)]
