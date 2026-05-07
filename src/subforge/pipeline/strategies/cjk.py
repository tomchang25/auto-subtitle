"""Compatibility wrapper for the CJK subtitle pipeline strategy.

The CJK orchestration is implemented as a generic
:class:`subforge.pipeline.stages.runner.StagedPipelineRunner` driven by
:class:`subforge.pipeline.stages.cjk_policy.CjkPolicy`. This module now
exists purely to keep the historical strategy import path stable for
callers and tests:

* ``CjkPipelineStrategy`` — public strategy class, used by
  :func:`subforge.pipeline.strategies.get_strategy`.
* ``CjkPolicy`` — re-exported because PR2 introduced this class on this
  module path.
* ``_map_corrected_to_raw`` — re-exported because tests import it from
  here directly.

All the CJK-specific stage behavior lives in
``subforge.pipeline.stages.cjk_policy``.
"""

from __future__ import annotations

from subforge.nlp.cjk_corrector import Corrector, NoOpCorrector
from subforge.pipeline.stages.cjk_policy import (
    CjkPolicy,
    _map_corrected_to_raw,
)
from subforge.pipeline.stages.runner import StagedPipelineRunner
from subforge.pipeline.strategies.base import (
    LanguagePipelineStrategy,
    StrategyContext,
)


class CjkPipelineStrategy(LanguagePipelineStrategy):
    """Bind :class:`CjkPolicy` to :class:`StagedPipelineRunner`."""

    def __init__(self, corrector: Corrector | None = None):
        self.corrector = corrector or NoOpCorrector()
        self._runner = StagedPipelineRunner(
            CjkPolicy(corrector=self.corrector)
        )

    def run(
        self,
        word_segments: list[dict],
        ctx: StrategyContext,
    ) -> list[list[dict]]:
        return self._runner.run(word_segments, ctx)


__all__ = [
    "CjkPipelineStrategy",
    "CjkPolicy",
    "_map_corrected_to_raw",
]
