"""Strategy interface for language-specific subtitle pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from subforge.nlp.lang_profile import LanguageProfile


EmitFn = Callable[[str, str], None]
CancelFn = Callable[[], None]


@dataclass
class StrategyContext:
    """Per-run context passed from the orchestrator to a strategy."""

    profile: LanguageProfile
    project_dir: Path
    force: bool
    emit: EmitFn
    check_cancel: CancelFn
    chinese_benchmark: bool = False
    options: dict = field(default_factory=dict)


class LanguagePipelineStrategy(ABC):
    """Convert ASR ``word_segments`` into refined sentence chunks.

    The returned value is a ``list[list[token]]`` where each token has at
    least ``text``, ``start``, ``end``, and ``is_punct`` keys — the same
    shape consumed by :func:`subforge.utils.get_bounds_and_text`.
    """

    @abstractmethod
    def run(
        self,
        word_segments: list[dict],
        ctx: StrategyContext,
    ) -> list[list[dict]]:
        ...
