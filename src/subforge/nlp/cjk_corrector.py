"""Corrector seam for the CJK subtitle pipeline.

The pipeline treats text quality and timing as separate concerns: stage 2
runs the raw transcript through a :class:`Corrector` so a later pre-plan can
plug in an LLM-based or rule-based corrector without further changes to the
pipeline shape. The default :class:`NoOpCorrector` returns the input
unchanged so the seam exists with no extra runtime dependency.
"""

from __future__ import annotations

from typing import Protocol


class Corrector(Protocol):
    """Strategy for cleaning up an ASR transcript before sentence splitting."""

    def correct(self, text: str, lang: str) -> str:
        ...


class NoOpCorrector:
    """Default corrector that returns its input verbatim."""

    def correct(self, text: str, lang: str) -> str:
        return text
