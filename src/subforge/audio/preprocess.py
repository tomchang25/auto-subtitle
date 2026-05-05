import logging
import shutil
import subprocess
from pathlib import Path

from subforge.audio.demucs_wrapper import run_demucs, get_demucs_vocals_path

logger = logging.getLogger(__name__)


def _find_ffmpeg() -> str:
    """Return the ffmpeg binary path, or raise if not found."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg and make sure it's accessible."
        )
    return ffmpeg


def _convert_to_mono_wav(input_path: Path, output_path: Path) -> None:
    """Convert audio to 16kHz mono WAV using ffmpeg (streams, no RAM spike)."""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",               # overwrite output
        "-i", str(input_path),
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz (what whisper expects)
        "-f", "wav",
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode('utf-8', errors='replace')}")


def preprocess_audio(
    audio_path: Path, project_dir: Path, use_demucs: bool = True, force: bool = False
) -> Path:
    """
    Runs preprocessing: Demucs (optional) + mono WAV conversion via ffmpeg.
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

    # Convert to mono 16kHz WAV via ffmpeg (streaming, no full-file RAM load)
    mono_output_dir = project_dir / "mono_wav"
    mono_output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = mono_output_dir / f"{audio_path.stem}.wav"
    if not force and wav_path.exists():
        logger.info("Mono WAV already exists, skipping: %s", wav_path)
    else:
        _convert_to_mono_wav(audio_path, wav_path)
        logger.info("Mono WAV saved: %s", wav_path)

    return wav_path
