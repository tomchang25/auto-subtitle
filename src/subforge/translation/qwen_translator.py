from __future__ import annotations

import re
from typing import List

from subforge.translation.base import SubtitleChunk, TranslatedChunk


class QwenTranslator:
    def __init__(
        self,
        model_path_or_id: str = "Qwen/Qwen3-4B",
        batch_size: int = 2,
        delimiter: str = " ### ",
    ):
        self.model_path_or_id = model_path_or_id
        self.batch_size = batch_size
        self.delimiter = delimiter
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path_or_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path_or_id,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

    def translate(self, chunks: List[SubtitleChunk]) -> List[TranslatedChunk]:
        self._load()
        texts = [c["segment"] for c in chunks]
        translations: list[str] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            joined = self.delimiter.join(batch)

            prompt = self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a professional translator."},
                    {
                        "role": "user",
                        "content": (
                            f"Translate the following English sentences into fluent Simplified "
                            f"Chinese. Use '{self.delimiter.strip()}' to separate each sentence:"
                            f"\n\n{joined}"
                        ),
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

            full_response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "assistant\n" in full_response:
                translated = full_response.split("assistant\n", 1)[1]
            else:
                translated = full_response

            translated = re.sub(r"<think>.*?</think>", "", translated, flags=re.DOTALL).strip()

            chunks_out = [s.strip() for s in translated.split(self.delimiter.strip()) if s.strip()]
            if len(chunks_out) < len(batch):
                chunks_out += [""] * (len(batch) - len(chunks_out))
            elif len(chunks_out) > len(batch):
                chunks_out = chunks_out[: len(batch)]

            translations.extend(chunks_out)

        return [{**c, "translation": zh} for c, zh in zip(chunks, translations)]
