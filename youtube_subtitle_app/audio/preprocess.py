from pathlib import Path
from pydub import AudioSegment
from youtube_subtitle_app.audio.demucs_wrapper import run_demucs


def preprocess_audio(audio_path: Path, project_dir: Path, use_demucs=True) -> Path:
    """
    Runs preprocessing: Demucs (optional) + mono conversion.
    Returns the final WAV file for transcription.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    print(f"[Preprocess] Starting on {audio_path}")

    if use_demucs:
        audio_path = run_demucs(audio_path, project_dir / "demucs_output")

    # Convert to mono WAV
    mono_output_dir = project_dir / "mono_wav"
    mono_output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = mono_output_dir / f"{audio_path.stem}.wav"
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    audio.export(wav_path, format="wav")

    print(f"[Preprocess] Mono WAV saved: {wav_path}")
    return wav_path
