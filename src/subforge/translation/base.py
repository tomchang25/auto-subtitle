from pathlib import Path
from typing import Callable, Protocol
from typing_extensions import TypedDict

ProgressCallback = Callable[[str, str], None]


class SubtitleChunk(TypedDict):
    start: float
    end: float
    segment: str


class TranslatedChunk(TypedDict):
    start: float
    end: float
    segment: str
    translation: str


class Translator(Protocol):
    def translate(
        self,
        chunks: list[SubtitleChunk],
        cache_dir: Path | None = None,
        force: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> list[TranslatedChunk]: ...
