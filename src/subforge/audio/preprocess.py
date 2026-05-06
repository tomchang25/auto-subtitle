import logging
import shutil
import subprocess
from pathlib import Path

from subforge.audio.demucs_wrapper import run_demucs, get_demucs_vocals_path
from subforge.config import DEMUCS_CHUNK_MINUTES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str:
    """Return the ffmpeg binary path, or raise if not found.

    Checks the system PATH first, then falls back to a bundled copy in the
    project's ``tools/`` directory (downloaded automatically by setup.bat).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg

    # Fallback: check <project_root>/tools/ffmpeg.exe
    tools_ffmpeg = Path(__file__).resolve().parents[2] / "tools" / "ffmpeg.exe"
    if tools_ffmpeg.is_file():
        return str(tools_ffmpeg)

    raise FileNotFoundError(
        "ffmpeg not found on PATH or in tools/. "
        "Run setup.bat or install ffmpeg manually."
    )


def _get_duration_seconds(audio_path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe not found on PATH.")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed:\n{result.stderr.decode('utf-8', errors='replace')}"
        )
    return float(result.stdout.strip())


def _split_audio(audio_path: Path, output_dir: Path, chunk_seconds: int) -> list[Path]:
    """Split audio into chunks of *chunk_seconds* using ffmpeg. Returns chunk paths."""
    ffmpeg = _find_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    # ffmpeg segment muxer: produces chunk_000.mp3, chunk_001.mp3, …
    pattern = str(output_dir / "chunk_%03d.mp3")
    cmd = [
        ffmpeg, "-y",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",          # no re-encode, just cut
        pattern,
    ]
    logger.info("Splitting audio into %d-second chunks", chunk_seconds)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg split failed:\n{result.stderr.decode('utf-8', errors='replace')}"
        )
    chunks = sorted(output_dir.glob("chunk_*.mp3"))
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def _concat_audio(parts: list[Path], output_path: Path) -> None:
    """Concatenate audio files via ffmpeg concat demuxer."""
    ffmpeg = _find_ffmpeg()
    # Write concat list file
    list_file = output_path.parent / "concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            # ffmpeg concat requires forward slashes or escaped backslashes
            safe = str(p).replace("\\", "/")
            f.write(f"file '{safe}'\n")
    cmd = [
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    logger.info("Concatenating %d vocal tracks", len(parts))
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed:\n{result.stderr.decode('utf-8', errors='replace')}"
        )
    list_file.unlink(missing_ok=True)


def _convert_to_mono_wav(input_path: Path, output_path: Path) -> None:
    """Convert audio to 16kHz mono WAV using ffmpeg (streams, no RAM spike)."""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz (what whisper expects)
        "-f", "wav",
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed:\n{result.stderr.decode('utf-8', errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Demucs (with chunked processing for long files)
# ---------------------------------------------------------------------------

def _run_demucs_chunked(
    audio_path: Path, demucs_out_dir: Path, force: bool = False
) -> Path:
    """
    Run demucs on audio, splitting into chunks for long files.

    For files <= DEMUCS_CHUNK_MINUTES: run demucs directly (same as before).
    For longer files: split → demucs each chunk → concat vocal tracks.

    Returns the path to the final vocals file.
    """
    chunk_seconds = DEMUCS_CHUNK_MINUTES * 60

    # Check if final concatenated output already exists
    concat_vocals = demucs_out_dir / "vocals_concat.mp3"
    single_vocals = get_demucs_vocals_path(demucs_out_dir)

    if not force and concat_vocals.exists():
        logger.info("Concatenated vocals already exist, skipping: %s", concat_vocals)
        return concat_vocals
    if not force and single_vocals.exists():
        logger.info("Demucs output already exists, skipping: %s", single_vocals)
        return single_vocals

    # Get duration to decide whether to chunk
    duration = _get_duration_seconds(audio_path)
    logger.info("Audio duration: %.0fs (%.1f min)", duration, duration / 60)

    if duration <= chunk_seconds:
        # Short file: run demucs directly
        return run_demucs(audio_path, demucs_out_dir)

    # Long file: split → demucs each chunk → concat
    chunks_dir = demucs_out_dir / "chunks"
    audio_chunks = _split_audio(audio_path, chunks_dir, chunk_seconds)

    vocal_parts: list[Path] = []
    for i, chunk_path in enumerate(audio_chunks):
        chunk_out_dir = demucs_out_dir / f"chunk_{i:03d}"
        chunk_vocals = get_demucs_vocals_path(chunk_out_dir)

        if not force and chunk_vocals.exists():
            logger.info(
                "Chunk %d/%d: cached, skipping", i + 1, len(audio_chunks)
            )
            vocal_parts.append(chunk_vocals)
        else:
            logger.info(
                "Chunk %d/%d: running demucs", i + 1, len(audio_chunks)
            )
            result = run_demucs(chunk_path, chunk_out_dir)
            vocal_parts.append(result)

    # Concat all vocal parts
    _concat_audio(vocal_parts, concat_vocals)
    logger.info("Concatenated vocals saved: %s", concat_vocals)
    return concat_vocals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_audio(
    audio_path: Path, project_dir: Path, use_demucs: bool = True, force: bool = False
) -> Path:
    """
    Runs preprocessing: Demucs (optional, chunked for long files) + mono WAV conversion.
    Returns the final WAV file for transcription.

    If force=True, re-runs all steps regardless of cached outputs.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    logger.info("Starting on %s", audio_path)

    if use_demucs:
        demucs_out_dir = project_dir / "demucs_output"
        audio_path = _run_demucs_chunked(audio_path, demucs_out_dir, force=force)

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
