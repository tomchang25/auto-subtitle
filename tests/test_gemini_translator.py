from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from subforge.translation.factory import BACKEND_NAMES
from subforge.translation.base import SubtitleChunk
from subforge.translation.gemini_translator import (
    LANG_MAP,
    GeminiTranslator,
    _resolve_api_key,
)


def _make_chunks(texts: list[str]) -> list[SubtitleChunk]:
    return [
        SubtitleChunk(start=i * 2.0, end=i * 2.0 + 1.5, segment=t)
        for i, t in enumerate(texts)
    ]


def _make_translator(**kwargs) -> GeminiTranslator:
    kwargs.setdefault("api_key", "test-key")
    return GeminiTranslator(**kwargs)


def _inject_model(t: GeminiTranslator, responses: list[str]) -> MagicMock:
    """Set a mock client directly, bypassing _load() and the real import."""
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = [MagicMock(text=r) for r in responses]
    t._client = mock_model
    return mock_model


# --- Factory registration ---


def test_gemini_registered_in_factory():
    assert "gemini" in BACKEND_NAMES


# --- LANG_MAP ---


def test_lang_map_has_required_codes():
    for code in ("zho_Hant", "zho_Hans", "jpn_Jpan", "kor_Hang", "eng_Latn"):
        assert code in LANG_MAP


def test_lang_map_unknown_code_falls_back_to_code_itself():
    _make_translator()
    src = LANG_MAP.get("xxx_Unkn", "xxx_Unkn")
    assert src == "xxx_Unkn"


# --- API key resolution ---


def test_resolve_key_explicit_wins(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert _resolve_api_key("explicit") == "explicit"


def test_resolve_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert _resolve_api_key(None) == "env-key"


def test_resolve_key_missing_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch.dict(sys.modules, {"dotenv": None}):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            _resolve_api_key(None)


# --- _parse_response ---


def test_parse_response_period_prefix():
    t = GeminiTranslator.__new__(GeminiTranslator)
    assert t._parse_response("1. 你好\n2. 世界", 2) == ["你好", "世界"]


def test_parse_response_paren_prefix():
    t = GeminiTranslator.__new__(GeminiTranslator)
    assert t._parse_response("1) foo\n2) bar", 2) == ["foo", "bar"]


def test_parse_response_skips_blank_lines():
    t = GeminiTranslator.__new__(GeminiTranslator)
    assert t._parse_response("1. a\n\n2. b", 2) == ["a", "b"]


def test_parse_response_wrong_count_returns_none():
    t = GeminiTranslator.__new__(GeminiTranslator)
    assert t._parse_response("1. only one", 3) is None


def test_parse_response_single_item():
    t = GeminiTranslator.__new__(GeminiTranslator)
    assert t._parse_response("1. translated", 1) == ["translated"]


# --- translate ---


def test_translate_basic():
    chunks = _make_chunks(["Hello", "World"])
    t = _make_translator(tgt_lang="zho_Hant")
    _inject_model(t, ["1. 你好\n2. 世界"])

    result = t.translate(chunks)

    assert len(result) == 2
    assert result[0]["translation"] == "你好"
    assert result[1]["translation"] == "世界"
    assert result[0]["segment"] == "Hello"


def test_translate_preserves_timing():
    chunks = _make_chunks(["Hello"])
    t = _make_translator()
    _inject_model(t, ["1. 你好"])

    result = t.translate(chunks)

    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 1.5


def test_translate_empty_chunks():
    t = _make_translator()
    assert t.translate([]) == []


def test_translate_count_mismatch_falls_back_per_sentence():
    chunks = _make_chunks(["Hello", "World", "Bye"])
    t = _make_translator(tgt_lang="zho_Hant")
    # 3 batch attempts each return wrong count, then 3 per-sentence calls
    mock_model = _inject_model(
        t,
        [
            "1. 你好",  # batch attempt 0 — missing lines 2 and 3
            "1. 你好",  # batch attempt 1 — still wrong
            "1. 你好",  # batch attempt 2 — wrong → triggers per-sentence fallback
            "1. 你好",  # per-sentence: Hello
            "1. 世界",  # per-sentence: World
            "1. 再見",  # per-sentence: Bye
        ],
    )

    result = t.translate(chunks)

    assert len(result) == 3
    assert mock_model.generate_content.call_count == 6
    assert result[0]["translation"] == "你好"
    assert result[1]["translation"] == "世界"
    assert result[2]["translation"] == "再見"


def test_translate_api_error_returns_empty():
    chunks = _make_chunks(["Hello"])
    t = _make_translator()
    t._client = MagicMock()
    t._client.generate_content.side_effect = Exception("500 Server Error")

    with patch("subforge.translation.gemini_translator.time.sleep"):
        result = t.translate(chunks)

    assert len(result) == 1
    assert result[0]["translation"] == ""


def test_translate_batching_single_call():
    chunks = _make_chunks([f"Sentence {i}" for i in range(5)])
    t = _make_translator(tgt_lang="zho_Hant", batch_size=10)
    numbered = "\n".join(f"{i + 1}. trans_{i}" for i in range(5))
    mock_model = _inject_model(t, [numbered])

    result = t.translate(chunks)

    assert len(result) == 5
    assert mock_model.generate_content.call_count == 1


def test_translate_multiple_batches():
    chunks = _make_chunks([f"Sentence {i}" for i in range(7)])
    t = _make_translator(tgt_lang="zho_Hant", batch_size=4)
    batch1 = "\n".join(f"{i + 1}. trans_{i}" for i in range(4))
    batch2 = "\n".join(f"{i + 1}. trans_{i + 4}" for i in range(3))
    mock_model = _inject_model(t, [batch1, batch2])

    result = t.translate(chunks)

    assert len(result) == 7
    assert mock_model.generate_content.call_count == 2
