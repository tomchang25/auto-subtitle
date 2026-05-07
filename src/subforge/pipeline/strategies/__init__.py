"""Language-specific subtitle pipeline strategies.

The transcription stage of :class:`SubtitlePipeline` produces a single
``word_segments`` list regardless of language. From there the processing
diverges between alphabetic languages (English) and CJK
(Chinese/Japanese/Korean). Each language family is expressed as a strategy
object so the orchestrator can stay flat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from subforge.pipeline.strategies.base import (
    LanguagePipelineStrategy,
    StrategyContext,
)
from subforge.pipeline.strategies.cjk import CjkPipelineStrategy
from subforge.pipeline.strategies.english import EnglishPipelineStrategy

if TYPE_CHECKING:
    from subforge.nlp.cjk_corrector import Corrector
    from subforge.nlp.lang_profile import LanguageProfile


def get_strategy(
    profile: LanguageProfile,
    *,
    corrector: Corrector | None = None,
) -> LanguagePipelineStrategy:
    """Return the strategy that handles *profile*'s language family.

    CJK languages (and any other profile that opts out of the spaCy path)
    use :class:`CjkPipelineStrategy`; everything else uses the English path.
    """
    if not profile.use_spacy:
        return CjkPipelineStrategy(corrector=corrector)
    return EnglishPipelineStrategy()


__all__ = [
    "CjkPipelineStrategy",
    "EnglishPipelineStrategy",
    "LanguagePipelineStrategy",
    "StrategyContext",
    "get_strategy",
]
