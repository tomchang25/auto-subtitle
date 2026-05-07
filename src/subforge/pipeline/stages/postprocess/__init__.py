"""Postprocess stage implementations.

* :mod:`word_count` — token-aware refine / split-long / merge-short
  pass used by the English pipeline (and the CJK fallback path).
* :mod:`display_width` — display-width-aware postprocess used by the
  CJK transcript-first pipeline.

The module re-exports :func:`finalize_token_chunks` so callers that
previously imported it from ``stages.postprocess_helpers`` can be
redirected to ``stages.postprocess`` without rebinding their imports
to a deeper module path.
"""

from __future__ import annotations

from subforge.pipeline.stages.postprocess.word_count import (
    finalize_token_chunks,
)

__all__ = ["finalize_token_chunks"]
