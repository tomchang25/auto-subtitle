from pathlib import Path
from typing import List, Protocol
from typing_extensions import TypedDict


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
        chunks: List[SubtitleChunk],
        cache_dir: Path | None = None,
        force: bool = False,
    ) -> List[TranslatedChunk]: ...
