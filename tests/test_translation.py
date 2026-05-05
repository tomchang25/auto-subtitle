import sys
from unittest.mock import MagicMock, patch

import pytest

SAMPLE_CHUNKS = [
    {"start": 0.0, "end": 1.0, "segment": "Hello world."},
    {"start": 1.0, "end": 2.0, "segment": "How are you?"},
]


# --- Factory ---

def test_factory_unknown_backend():
    from subforge.translation.factory import create_translator
    with pytest.raises(ValueError, match="Unknown translation backend"):
        create_translator("nonexistent")


def test_factory_known_backends_listed():
    from subforge.translation.factory import BACKEND_NAMES
    assert "nllb" in BACKEND_NAMES


# --- NLLBTranslator ---

def test_nllb_translate_output_shape():
    mock_tf = MagicMock()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device.return_value = "cpu"

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.convert_tokens_to_ids.return_value = 256047  # fake id
    mock_tokenizer.return_value = MagicMock(to=MagicMock(return_value={"input_ids": None}))
    mock_tokenizer.batch_decode.return_value = ["你好世界。", "你好嗎？"]
    mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    # Mock model
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2], [3, 4]]
    mock_tf.AutoModelForSeq2SeqLM.from_pretrained.return_value = mock_model

    with patch.dict(sys.modules, {"transformers": mock_tf, "torch": mock_torch}):
        from subforge.translation.nllb_translator import NLLBTranslator
        t = NLLBTranslator()
        result = t.translate(SAMPLE_CHUNKS)

    assert len(result) == 2
    assert result[0]["translation"] == "你好世界。"
    assert result[1]["translation"] == "你好嗎？"
    assert result[0]["segment"] == "Hello world."


# --- formatter bilingual support ---

def test_format_srt_bilingual():
    from subforge.subtitle.formatter import format_srt
    segments = [{"start": 0.16, "end": 2.0, "segment": "Hello.", "translation": "你好。"}]
    output = format_srt(segments)
    assert "Hello." in output
    assert "你好。" in output


def test_format_srt_monolingual_unchanged():
    from subforge.subtitle.formatter import format_srt
    segments = [{"start": 0.0, "end": 1.0, "segment": "Hello."}]
    output = format_srt(segments)
    non_empty = [line for line in output.strip().splitlines() if line.strip()]
    # index + timestamp + text = 3 lines
    assert len(non_empty) == 3


def test_format_srt_empty_translation_ignored():
    from subforge.subtitle.formatter import format_srt
    segments = [{"start": 0.0, "end": 1.0, "segment": "Hello.", "translation": ""}]
    output = format_srt(segments)
    non_empty = [line for line in output.strip().splitlines() if line.strip()]
    assert len(non_empty) == 3
