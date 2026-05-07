"""Compatibility wrapper for the English subtitle pipeline strategy.

The English orchestration is implemented as a generic
:class:`subforge.pipeline.stages.runner.StagedPipelineRunner` driven by
:class:`subforge.pipeline.stages.english_policy.EnglishPolicy`. This
module now exists purely to keep the historical strategy import path
stable for callers and tests.
"""

from __future__ import annotations

from subforge.pipeline.stages.english_policy import EnglishPolicy
from subforge.pipeline.stages.runner import StagedPipelineRunner
from subforge.pipeline.strategies.base import (
    LanguagePipelineStrategy,
    StrategyContext,
)


class EnglishPipelineStrategy(LanguagePipelineStrategy):
    """Bind :class:`EnglishPolicy` to :class:`StagedPipelineRunner`."""

    def run(
        self,
        word_segments: list[dict],
        ctx: StrategyContext,
    ) -> list[list[dict]]:
        # Pre-seed word_segments on the policy so cache-hit paths that
        # skip ``build_inputs`` still have them available for alignment.
        policy = EnglishPolicy(word_segments=word_segments)
        runner = StagedPipelineRunner(policy)
        return runner.run(word_segments, ctx)


__all__ = ["EnglishPipelineStrategy"]
