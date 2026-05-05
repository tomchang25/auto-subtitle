from __future__ import annotations

from typing import List

from subforge.translation.base import SubtitleChunk, TranslatedChunk


class NLLBTranslator:
    MODEL_NAME = "facebook/nllb-200-1.3B"

    def __init__(
        self,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "zho_Hans",
        batch_size: int = 4,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)

    def translate(self, chunks: List[SubtitleChunk]) -> List[TranslatedChunk]:
        self._load()
        import torch
        from transformers import pipeline

        texts = [c["segment"] for c in chunks]
        pipe = pipeline(
            "translation",
            model=self._model,
            tokenizer=self._tokenizer,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            max_length=512,
            device=0 if torch.cuda.is_available() else -1,
        )
        translations = [r["translation_text"] for r in pipe(texts, batch_size=self.batch_size)]
        return [{**c, "translation": zh} for c, zh in zip(chunks, translations)]
