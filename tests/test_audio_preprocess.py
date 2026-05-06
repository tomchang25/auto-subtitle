import shutil
import subprocess

from subforge.audio.preprocess import preprocess_audio


def _generate_test_audio(output_path):
    """Generate a 1-second sine wave using ffmpeg (no pydub dependency)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )


def test_preprocess_audio_creates_mono_wav(tmp_path):
    # Arrange: generate a small test MP3 via ffmpeg
    audio_path = tmp_path / "test_input.mp3"
    _generate_test_audio(audio_path)

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Act
    wav_path = preprocess_audio(audio_path, project_dir, use_demucs=False)

    # Assert
    assert wav_path.exists()
    assert wav_path.suffix == ".wav"
    assert "mono_wav" in str(wav_path)
