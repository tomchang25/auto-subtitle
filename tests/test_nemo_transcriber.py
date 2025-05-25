from youtube_subtitle_app.transcription.nemo_transcriber import (
    transcribe_audio_word_level,
)
from youtube_subtitle_app.config import DEFAULT_MODEL
from pathlib import Path


def test_transcribe_real_audio():
    # Load real audio file
    audio_path = Path(__file__).parent / "data" / "news.wav"

    # Make sure test audio file exists
    assert audio_path.exists(), f"Test audio file not found: {audio_path}"

    # Transcribe
    result = transcribe_audio_word_level(audio_path, model_name=DEFAULT_MODEL)

    # Assert structure
    assert isinstance(result, list)
    assert all("word" in w and "start" in w and "end" in w for w in result)
