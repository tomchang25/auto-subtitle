from transformers import MarianMTModel, MarianTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
import torch

# --- MarianMT Setup ---
_marian_model = "Helsinki-NLP/opus-mt-en-zh"
_marian_tokenizer = MarianTokenizer.from_pretrained(_marian_model)
_marian_model_obj = MarianMTModel.from_pretrained(_marian_model)


def _translate_marian(texts, batch_size=8):
    all_translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = _marian_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        )
        outputs = _marian_model_obj.generate(**inputs)
        translated_batch = [
            _marian_tokenizer.decode(t, skip_special_tokens=True) for t in outputs
        ]
        all_translations.extend(translated_batch)
    return all_translations


# --- NLLB-1.3B Setup ---

_nllb_model_name = "facebook/nllb-200-1.3B"

# Lazy-load so it doesn't block Marian/fairseq users
_nllb_tokenizer = None
_nllb_model = None


def _load_nllb_model():
    global _nllb_tokenizer, _nllb_model
    if _nllb_model is None or _nllb_tokenizer is None:
        _nllb_tokenizer = AutoTokenizer.from_pretrained(_nllb_model_name)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(_nllb_model_name)


def _translate_nllb(texts, src_lang="eng_Latn", tgt_lang="zho_Hans", batch_size=4):
    _load_nllb_model()
    pipe = pipeline(
        "translation",
        model=_nllb_model,
        tokenizer=_nllb_tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=512,
        device=0 if torch.cuda.is_available() else -1,
    )
    return [res["translation_text"] for res in pipe(texts, batch_size=batch_size)]


# --- Qwen3-8B Setup ---
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

_tokenizer_qwen = None
_model_qwen = None


def _load_qwen_transformers(model_path_or_id):
    global _tokenizer_qwen, _model_qwen
    if _tokenizer_qwen is None or _model_qwen is None:
        _tokenizer_qwen = AutoTokenizer.from_pretrained(model_path_or_id)
        _model_qwen = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )


def _translate_qwen(
    texts, model_path_or_id="Qwen/Qwen3-4B", batch_size=2, delimiter=" ### "
):
    _load_qwen_transformers(model_path_or_id)
    translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        joined = delimiter.join(batch)

        prompt = _tokenizer_qwen.apply_chat_template(
            [
                {"role": "system", "content": "You are a professional translator."},
                {
                    "role": "user",
                    "content": f"Translate the following English sentences into fluent Simplified Chinese. Use '{delimiter.strip()}' to separate each sentence:\n\n{joined}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = _tokenizer_qwen(prompt, return_tensors="pt").to(_model_qwen.device)
        outputs = _model_qwen.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=_tokenizer_qwen.eos_token_id,
        )

        full_response = _tokenizer_qwen.decode(outputs[0], skip_special_tokens=True)

        if "assistant\n" in full_response:
            translated = full_response.split("assistant\n", 1)[1]
        else:
            translated = full_response

        translated = re.sub(
            r"<think>.*?</think>", "", translated, flags=re.DOTALL
        ).strip()

        chunks = [s.strip() for s in translated.split(delimiter.strip()) if s.strip()]
        if len(chunks) < len(batch):
            chunks += [""] * (len(batch) - len(chunks))
        elif len(chunks) > len(batch):
            chunks = chunks[: len(batch)]

        translations.extend(chunks)

    return translations


# --- Unified interface ---
def translate_subtitles(chunks, method="marian", **kwargs):
    texts = [chunk["segment"] for chunk in chunks]

    if method == "marian":
        translations = _translate_marian(texts, **kwargs)
    elif method == "nllb":
        translations = _translate_nllb(texts, **kwargs)
    elif method == "qwen":
        translations = _translate_qwen(texts, **kwargs)

    else:
        raise ValueError(f"Unknown translation method: {method}")

    for chunk, zh in zip(chunks, translations):
        chunk["translation"] = zh

    return chunks


if __name__ == "__main__":
    print("0")
    from google import genai

    print("1")
    client = genai.Client(api_key="AIzaSyBZjna_LeqnINgH4jSW2zWI7nkNvWeNRp0")
    print("2")
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents="Explain how AI works in a few words"
    )
    print(response.text)
