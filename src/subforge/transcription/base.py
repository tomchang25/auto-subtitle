from pathlib import Path
from typing import Protocol


WordSegment = dict[str, object]


class WordLevelTranscriber(Protocol):
    """Shared contract for word-level transcribers.

    Implementations expose a callable that accepts a path to a mono WAV file
    and a model identifier, and returns a list of dicts shaped like::

        {"word": str, "start": float, "end": float}

    Timestamps are seconds from the start of the audio.
    """

    def __call__(self, wav_path: Path, model_name: str) -> list[WordSegment]: ...
