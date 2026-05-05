import logging
from pathlib import Path

import demucs.separate

from subforge.config import DEMUCS_MODEL, DEMUCS_VOCALS_FILENAME

logger = logging.getLogger(__name__)


def get_demucs_vocals_path(output_dir: Path) -> Path:
    """Canonical path where demucs vocals are expected."""
    return output_dir / DEMUCS_MODEL / DEMUCS_VOCALS_FILENAME


def run_demucs(input_path: Path, output_dir: Path) -> Path:
    """
    Runs Demucs to separate vocals from a given audio file.
    Returns the path to the separated vocals (or the original file if separation fails).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--mp3",
        "--two-stems",
        "vocals",
        "-o",
        str(output_dir),
        "--filename",
        "{stem}.{ext}",
        str(input_path),
    ]

    logger.info("Processing: %s", input_path)
    try:
        demucs.separate.main(args)
        expected = get_demucs_vocals_path(output_dir)
        if expected.exists():
            logger.info("Vocal track saved: %s", expected)
            return expected
        logger.warning(
            "Demucs output not found at %s. Using original.", expected
        )
        return input_path
    except Exception as e:
        logger.error("Error: %s", e)
        return input_path
