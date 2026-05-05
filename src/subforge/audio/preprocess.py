import logging
from pathlib import Path

from pydub import AudioSegment

from subforge.audio.demucs_wrapper import run_demucs, get_demucs_vocals_path

logger = logging.getLogger(__name__)


def preprocess_audio(
    audio_path: Path, project_dir: Path, use_demucs: bool = True, force: bool = False
) -> Path:
    """
    Runs preprocessing: Demucs (optional) + mono conversion.
    Returns the final WAV file for transcription.

    If force=True, re-runs all steps regardless of cached outputs.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    logger.info("Starting on %s", audio_path)

    if use_demucs:
        demucs_out_dir = project_dir / "demucs_output"
        expected = get_demucs_vocals_path(demucs_out_dir)
        if not force and expected.exists():
            logger.info("Demucs output already exists, skipping: %s", expected)
            audio_path = expected
        else:
            audio_path = run_demucs(audio_path, demucs_out_dir)

    # Convert to mono WAV
    mono_output_dir = project_dir / "mono_wav"
    mono_output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = mono_output_dir / f"{audio_path.stem}.wav"
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    audio.export(wav_path, format="wav")

    logger.info("Mono WAV saved: %s", wav_path)
    return wav_path
