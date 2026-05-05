from __future__ import annotations

import logging
from typing import List

from subforge.translation.base import SubtitleChunk, TranslatedChunk

logger = logging.getLogger(__name__)

# Default batch sizes by device — GPU can handle more at once
_DEFAULT_BATCH_CPU = 4
_DEFAULT_BATCH_GPU = 16


class NLLBTranslator:
    MODEL_NAME = "facebook/nllb-200-1.3B"

    def __init__(
        self,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "zho_Hans",
        batch_size: int | None = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._explicit_batch_size = batch_size
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

        assert self._tokenizer is not None
        assert self._model is not None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        if self._explicit_batch_size is not None:
            batch_size = self._explicit_batch_size
        else:
            batch_size = _DEFAULT_BATCH_GPU if device.type == "cuda" else _DEFAULT_BATCH_CPU

        logger.info(
            "NLLB translation: %d chunks, batch_size=%d, device=%s",
            len(chunks), batch_size, device,
        )

        self._tokenizer.src_lang = self.src_lang
        texts = [c["segment"] for c in chunks]
        translations: list[str] = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            tgt_lang_id = self._tokenizer.convert_tokens_to_ids(self.tgt_lang)
            generated = self._model.generate(
                **inputs, forced_bos_token_id=tgt_lang_id, max_length=512
            )
            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            translations.extend(decoded)

            done = min(i + batch_size, total)
            logger.info("NLLB progress: %d/%d (%d%%)", done, total, done * 100 // total)

        return [{**c, "translation": zh} for c, zh in zip(chunks, translations)]
