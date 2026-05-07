"""Required metadata key contract for ``final_cues.json``.

The runner writes a metadata block at the head of ``final_cues.json``
for every run. ``REQUIRED_META_KEYS`` is the language-agnostic set of
keys downstream tooling can rely on regardless of which policy
(English, CJK, fallback) produced the file. Policies are free to add
language-specific keys on top, but they must always emit the required
set.
"""

from __future__ import annotations

REQUIRED_META_KEYS: frozenset[str] = frozenset(
    {
        "mode",
        "profile",
        "text_source",
        "timing_source",
        "timing_status",
        "transcript_backend",
        "timing_backend",
        "correction_mode",
        "correction_applied",
        "fallback_used",
        "fallback_reason",
        "alignment_total_cues",
        "alignment_anchored_cues",
        "transcript_length",
    }
)


def validate_meta(meta: dict) -> list[str]:
    """Return the sorted list of required keys missing from ``meta``."""
    return sorted(REQUIRED_META_KEYS - set(meta.keys()))


__all__ = ["REQUIRED_META_KEYS", "validate_meta"]
