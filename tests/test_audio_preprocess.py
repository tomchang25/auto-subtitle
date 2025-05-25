import pytest
from pathlib import Path
from youtube_subtitle_app.audio.preprocess import preprocess_audio


def test_preprocess_audio_creates_mono_wav(tmp_path):
    # Arrange: copy or generate a small test MP3
    from pydub.generators import Sine

    audio_path = tmp_path / "test_input.mp3"
    tone = Sine(440).to_audio_segment(duration=1000)  # 1 second sine wave
    tone.export(audio_path, format="mp3")

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Act
    wav_path = preprocess_audio(audio_path, project_dir, use_demucs=False)

    # Assert
    assert wav_path.exists()
    assert wav_path.suffix == ".wav"
    assert "mono_wav" in str(wav_path)
