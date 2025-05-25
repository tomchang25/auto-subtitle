from pathlib import Path
from youtube_subtitle_app.subtitle.formatter import format_srt
from youtube_subtitle_app.subtitle.writer import write_srt


def dummy_segments():
    return [
        {"start": 0.0, "end": 2.4, "segment": "Hello world."},
        {"start": 2.5, "end": 5.0, "segment": "This is a test."},
    ]


def test_format_srt():
    srt = format_srt(dummy_segments())
    assert isinstance(srt, str)
    assert "00:00:00,000 --> 00:00:02,400" in srt
    assert "Hello world." in srt


def test_write_srt(tmp_path):
    output_path = tmp_path / "test_output.srt"
    segments = dummy_segments()
    write_srt(segments, output_path)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "This is a test." in content
