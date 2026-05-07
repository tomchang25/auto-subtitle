"""Display-width-aware postprocess for the CJK staged pipeline.

CJK cues use display width (CJK chars count as 2) rather than word
count for length budgeting, so the English token-aware postprocess is
not appropriate. This module is the thin adapter between the CJK
policy's :class:`AlignedCue` list and the
:func:`subforge.nlp.cjk_postprocess.postprocess_cjk_cues` primitive,
followed by the writer-chunk conversion.
"""

from __future__ import annotations

from subforge.config import (
    CJK_POSTPROCESS_MAX_DURATION,
    CJK_POSTPROCESS_MAX_WIDTH,
    CJK_POSTPROCESS_MERGE_MAX_DURATION,
    CJK_POSTPROCESS_MERGE_MAX_GAP,
    CJK_POSTPROCESS_MERGE_MAX_WIDTH,
    CJK_POSTPROCESS_MIN_DURATION,
    CJK_POSTPROCESS_SHORT_CUE_WIDTH,
)
from subforge.nlp.cjk_postprocess import (
    PostprocessConfig,
    postprocess_cjk_cues,
    postprocess_cues_to_writer_chunks,
)
from subforge.nlp.lang_profile import LanguageProfile
from subforge.pipeline.stages.models import AlignedCue


def postprocess_cjk(
    cues: list[AlignedCue],
    profile: LanguageProfile,
) -> tuple[list[list[dict]], dict]:
    """Run the display-width-aware postprocess over CJK aligned cues."""
    cfg = PostprocessConfig(
        max_display_width=CJK_POSTPROCESS_MAX_WIDTH,
        min_duration=CJK_POSTPROCESS_MIN_DURATION,
        max_duration=CJK_POSTPROCESS_MAX_DURATION,
        merge_max_width=CJK_POSTPROCESS_MERGE_MAX_WIDTH,
        merge_max_duration=CJK_POSTPROCESS_MERGE_MAX_DURATION,
        merge_max_gap=CJK_POSTPROCESS_MERGE_MAX_GAP,
        short_cue_width=CJK_POSTPROCESS_SHORT_CUE_WIDTH,
    )
    post_cues, post_diag = postprocess_cjk_cues(cues, profile, cfg)
    chunks = postprocess_cues_to_writer_chunks(post_cues, profile)
    return chunks, post_diag


__all__ = ["postprocess_cjk"]
