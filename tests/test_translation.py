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
    assert "marian" in BACKEND_NAMES
    assert "nllb" in BACKEND_NAMES
    assert "qwen" in BACKEND_NAMES


# --- MarianTranslator ---

def _make_mock_transformers_marian():
    mock_tf = MagicMock()
    mock_tok = mock_tf.MarianTokenizer.from_pretrained.return_value
    mock_model = mock_tf.MarianMTModel.from_pretrained.return_value
    mock_tok.return_value = {}
    mock_model.generate.return_value = [MagicMock(), MagicMock()]
    mock_tok.decode.side_effect = ["你好世界。", "你好嗎？"]
    return mock_tf, mock_tok, mock_model


def test_marian_translate_output_shape():
    mock_tf, mock_tok, mock_model = _make_mock_transformers_marian()
    with patch.dict(sys.modules, {"transformers": mock_tf}):
        from subforge.translation.marian_translator import MarianTranslator
        t = MarianTranslator()
        result = t.translate(SAMPLE_CHUNKS)

    assert len(result) == 2
    assert result[0]["translation"] == "你好世界。"
    assert result[1]["translation"] == "你好嗎？"
    assert result[0]["segment"] == "Hello world."
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 1.0


def test_marian_lazy_load_called_once():
    mock_tf, mock_tok, mock_model = _make_mock_transformers_marian()
    # reset side_effect so repeated calls don't exhaust it
    mock_tok.decode.side_effect = None
    mock_tok.decode.return_value = "翻譯"

    with patch.dict(sys.modules, {"transformers": mock_tf}):
        from subforge.translation.marian_translator import MarianTranslator
        t = MarianTranslator()
        t.translate(SAMPLE_CHUNKS[:1])
        t.translate(SAMPLE_CHUNKS[:1])

    mock_tf.MarianTokenizer.from_pretrained.assert_called_once()
    mock_tf.MarianMTModel.from_pretrained.assert_called_once()


# --- NLLBTranslator ---

def test_nllb_translate_output_shape():
    mock_tf = MagicMock()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    mock_pipe_fn = MagicMock()
    mock_pipe_fn.return_value = [
        {"translation_text": "你好世界。"},
        {"translation_text": "你好嗎？"},
    ]
    mock_tf.pipeline.return_value = mock_pipe_fn

    with patch.dict(sys.modules, {"transformers": mock_tf, "torch": mock_torch}):
        from subforge.translation.nllb_translator import NLLBTranslator
        t = NLLBTranslator()
        result = t.translate(SAMPLE_CHUNKS)

    assert len(result) == 2
    assert result[0]["translation"] == "你好世界。"
    assert result[1]["translation"] == "你好嗎？"
    assert result[0]["segment"] == "Hello world."


# --- QwenTranslator ---

def test_qwen_translate_output_shape():
    mock_tf = MagicMock()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.float32 = "float32"

    mock_tok = mock_tf.AutoTokenizer.from_pretrained.return_value
    mock_model = mock_tf.AutoModelForCausalLM.from_pretrained.return_value
    mock_model.device = "cpu"
    mock_tok.apply_chat_template.return_value = "prompt"
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = {}
    mock_tok.return_value = mock_inputs
    mock_model.generate.return_value = [MagicMock()]
    mock_tok.decode.return_value = "assistant\n你好世界。 ### 你好嗎？"
    mock_tok.eos_token_id = 0

    with patch.dict(sys.modules, {"transformers": mock_tf, "torch": mock_torch}):
        from subforge.translation.qwen_translator import QwenTranslator
        t = QwenTranslator()
        result = t.translate(SAMPLE_CHUNKS)

    assert len(result) == 2
    assert result[0]["segment"] == "Hello world."
    assert result[1]["segment"] == "How are you?"


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
    non_empty = [l for l in output.strip().splitlines() if l.strip()]
    # index + timestamp + text = 3 lines
    assert len(non_empty) == 3


def test_format_srt_empty_translation_ignored():
    from subforge.subtitle.formatter import format_srt
    segments = [{"start": 0.0, "end": 1.0, "segment": "Hello.", "translation": ""}]
    output = format_srt(segments)
    non_empty = [l for l in output.strip().splitlines() if l.strip()]
    assert len(non_empty) == 3
