from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subforge.translation.base import SubtitleChunk

logger = logging.getLogger(__name__)

# Modules required by each entry point, mapped to their pip package names.
_CORE_DEPS = {
    "faster_whisper": "faster-whisper",
    "yt_dlp": "yt-dlp",
    "ffmpeg": "ffmpeg-python",
    "demucs": "demucs",
}
_GUI_DEPS = {
    "PySide6": "PySide6-Essentials",
}


def check_dependencies(*, gui: bool = False) -> None:
    """Check that required optional packages are installed.

    Exits with a clear message if anything is missing, so the user doesn't
    see a raw ImportError traceback.
    """
    required = dict(_CORE_DEPS)
    if gui:
        required.update(_GUI_DEPS)

    missing = [
        pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None
    ]
    if missing:
        print(
            f"\n✗ Missing dependencies: {', '.join(missing)}\n"
            f'  Install with:  pip install -e ".[full]"\n'
        )
        sys.exit(1)


def save_word_segments(word_segments, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(word_segments, f, indent=2)


def get_bounds_and_text(chunks) -> list[SubtitleChunk]:
    return [
        {
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "segment": " ".join(token["text"] for token in chunk),
        }
        for chunk in chunks
    ]


def save_to_json(bounds, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bounds, f, ensure_ascii=False, indent=2)

    logger.info("Saved bounds to: %s", output_path)
