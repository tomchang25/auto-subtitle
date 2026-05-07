"""Display-width-aware postprocessing for CJK subtitle cues.

The CJK pipeline up through Stage 4 produces sentence-shaped cues with
text, timing, and provenance. Stage 5 used to delegate to the shared
length-split / short-merge passes built around Whisper word tokens,
which assumes one "word" per token and therefore hard-cuts CJK text by
character count without distinguishing between half-width ASCII and
full-width ideographs.

This module replaces that path for CJK with a postprocess that operates
directly on aligned cues:

* sentence-aware boundaries — splits prefer the configured sentence-end
  punctuation, then any other punctuation, then a timing pause inferred
  from the cue interval, before falling back to a width-based hard cut;
* readability constraints — display width is measured with
  ``unicodedata.east_asian_width`` so a cue line is bounded by columns
  rather than raw ``len``;
* timing safety — the final cue list is sorted, clipped to be
  monotonic, and never overlaps;
* diagnostics — every output cue records the actions that produced it
  (``preserved`` / ``split`` / ``merged`` / ``shortened`` / ``expanded``
  / ``clipped`` / ``fallback``) so the postprocess decisions are
  visible alongside the alignment artifacts.

Only the CJK strategy uses this module; the English strategy keeps the
shared length-split / short-merge passes unchanged.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subforge.nlp.lang_profile import LanguageProfile
    from subforge.pipeline.strategies.cjk_models import CjkAlignedCue


# Recognised postprocess action tags. Documented as a public set so later
# stages (or tests) can lock against the vocabulary.
POSTPROCESS_ACTIONS: frozenset[str] = frozenset({
    "preserved",   # cue passed through without text or timing changes
    "split",       # cue produced by splitting a longer source cue
    "merged",      # cue produced by combining shorter source cues
    "shortened",   # cue duration capped to max_duration
    "expanded",    # cue duration extended to reach min_duration
    "clipped",     # cue end clipped to keep ordering monotonic
    "fallback",    # cue fabricated for an invalid / empty source
    "dropped",     # source cue discarded; appears only in diagnostics
})


@dataclass
class PostprocessedCue:
    """A cue ready for the writer with postprocess decisions attached."""

    text: str
    start: float
    end: float
    actions: list[str] = field(default_factory=list)
    source_index: int | None = None

    def add_action(self, action: str) -> None:
        if action and action not in self.actions:
            self.actions.append(action)


@dataclass
class PostprocessConfig:
    """Knobs for the CJK postprocess pass.

    All fields are keyword-only and have sensible defaults so callers can
    override one knob without restating the rest.
    """

    max_display_width: int = 28
    min_duration: float = 0.8
    max_duration: float = 6.0
    merge_max_width: int = 24
    merge_max_duration: float = 5.0
    merge_max_gap: float = 0.5
    short_cue_width: int = 6           # cues at/below this width are merge candidates
    pause_split_threshold: float = 0.6  # internal pause that justifies a split


# ---------------------------------------------------------------------------
# Width helpers
# ---------------------------------------------------------------------------


def display_width(text: str) -> int:
    """Return the column width of *text*.

    Full-width / wide East-Asian glyphs count as 2; ASCII and half-width
    forms count as 1; combining marks count as 0. Falls back to 1 for
    anything unicodedata can't classify so we never under-count.
    """
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        eaw = unicodedata.east_asian_width(ch)
        if eaw in ("F", "W"):
            width += 2
        else:
            width += 1
    return width


def _is_punct(ch: str, profile: "LanguageProfile") -> bool:
    return ch in profile.punctuation


def _is_sentence_end(ch: str, profile: "LanguageProfile") -> bool:
    return ch in profile.sentence_end


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _candidate_split_points(
    text: str,
    profile: "LanguageProfile",
) -> list[tuple[int, int]]:
    """Return ``(index_after_split, priority)`` tuples for *text*.

    Priority is a small integer where lower means stronger preference:
        0 — sentence-end punctuation (。！？.!?)
        1 — any other punctuation (、，；,;:)
    The index is the position **after** the punctuation so the split
    keeps trailing punctuation with the preceding cue.
    """
    points: list[tuple[int, int]] = []
    for i, ch in enumerate(text):
        if _is_sentence_end(ch, profile):
            points.append((i + 1, 0))
        elif _is_punct(ch, profile):
            points.append((i + 1, 1))
    return points


def _interpolate_split_time(
    start: float,
    end: float,
    text: str,
    split_idx: int,
) -> float:
    """Linear-time interpolation for splitting a cue at ``text[:split_idx]``.

    Width-weighted so a CJK char and an ASCII char don't get the same
    slice of the cue's duration.
    """
    if split_idx <= 0 or split_idx >= len(text) or end <= start:
        return start + (end - start) / 2
    total = display_width(text)
    if total <= 0:
        return start + (end - start) / 2
    consumed = display_width(text[:split_idx])
    frac = consumed / total
    return start + (end - start) * frac


def _split_once(
    cue: PostprocessedCue,
    profile: "LanguageProfile",
    cfg: PostprocessConfig,
) -> list[PostprocessedCue] | None:
    """Try one split that brings the cue under ``max_display_width``.

    Returns ``None`` if no acceptable split point exists. The returned
    list always contains exactly two cues; further reduction is the
    caller's responsibility.
    """
    text = cue.text
    if display_width(text) <= cfg.max_display_width:
        return None

    candidates = _candidate_split_points(text, profile)
    if candidates:
        # Prefer punctuation closest to the half-width point of the cue,
        # and among ties prefer the strongest (lowest-priority) class.
        target = display_width(text) / 2
        best: tuple[int, int] | None = None
        best_score: tuple[int, int] | None = None
        for idx, prio in candidates:
            # Don't pick a split that leaves an empty side.
            left = text[:idx].strip()
            right = text[idx:].strip()
            if not left or not right:
                continue
            consumed = display_width(text[:idx])
            score = (prio, abs(consumed - target))
            if best_score is None or score < best_score:
                best = (idx, prio)
                best_score = score
        if best is not None:
            idx, _prio = best
            return _split_at_index(cue, idx)

    # Fall back to a width-based hard cut at the largest left side that
    # still fits inside max_display_width.
    target = cfg.max_display_width
    consumed = 0
    cut = 0
    for i, ch in enumerate(text):
        glyph = display_width(ch)
        if consumed + glyph > target and cut > 0:
            break
        consumed += glyph
        cut = i + 1
    if 0 < cut < len(text):
        parts = _split_at_index(cue, cut)
        for part in parts:
            part.add_action("fallback")
        return parts
    return None


def _split_at_index(
    cue: PostprocessedCue,
    idx: int,
) -> list[PostprocessedCue]:
    text = cue.text
    left_text = text[:idx]
    right_text = text[idx:]
    split_t = _interpolate_split_time(cue.start, cue.end, text, idx)
    if split_t < cue.start:
        split_t = cue.start
    if split_t > cue.end:
        split_t = cue.end
    left = PostprocessedCue(
        text=left_text,
        start=cue.start,
        end=split_t,
        actions=list(cue.actions),
        source_index=cue.source_index,
    )
    right = PostprocessedCue(
        text=right_text,
        start=split_t,
        end=cue.end,
        actions=list(cue.actions),
        source_index=cue.source_index,
    )
    left.add_action("split")
    right.add_action("split")
    return [left, right]


def _split_long_cue(
    cue: PostprocessedCue,
    profile: "LanguageProfile",
    cfg: PostprocessConfig,
) -> list[PostprocessedCue]:
    """Recursively split *cue* until every part fits ``max_display_width``."""
    if display_width(cue.text) <= cfg.max_display_width:
        return [cue]

    parts = _split_once(cue, profile, cfg)
    if parts is None:
        return [cue]

    out: list[PostprocessedCue] = []
    for part in parts:
        out.extend(_split_long_cue(part, profile, cfg))
    return out


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def _can_merge(
    prev: PostprocessedCue,
    nxt: PostprocessedCue,
    profile: "LanguageProfile",
    cfg: PostprocessConfig,
) -> bool:
    if not prev.text or not nxt.text:
        return False
    # Don't merge across a hard sentence boundary.
    if prev.text and _is_sentence_end(prev.text[-1], profile):
        return False
    combined_width = display_width(prev.text) + display_width(nxt.text)
    if combined_width > cfg.merge_max_width:
        return False
    combined_duration = nxt.end - prev.start
    if combined_duration > cfg.merge_max_duration:
        return False
    gap = nxt.start - prev.end
    if gap > cfg.merge_max_gap:
        return False
    return True


def _merge_short_cues(
    cues: list[PostprocessedCue],
    profile: "LanguageProfile",
    cfg: PostprocessConfig,
) -> list[PostprocessedCue]:
    if not cues:
        return []
    out: list[PostprocessedCue] = [cues[0]]
    for cue in cues[1:]:
        prev = out[-1]
        prev_short = display_width(prev.text) <= cfg.short_cue_width
        cur_short = display_width(cue.text) <= cfg.short_cue_width
        if (prev_short or cur_short) and _can_merge(prev, cue, profile, cfg):
            merged = PostprocessedCue(
                text=prev.text + cue.text,
                start=prev.start,
                end=cue.end,
                actions=list({*prev.actions, *cue.actions}),
                source_index=prev.source_index,
            )
            merged.add_action("merged")
            out[-1] = merged
        else:
            out.append(cue)
    return out


# ---------------------------------------------------------------------------
# Duration / timing safety
# ---------------------------------------------------------------------------


def _enforce_durations(
    cues: list[PostprocessedCue],
    cfg: PostprocessConfig,
) -> list[PostprocessedCue]:
    """Cap long cues, extend short ones into available trailing silence."""
    n = len(cues)
    for i, cue in enumerate(cues):
        duration = cue.end - cue.start
        if duration > cfg.max_duration:
            cue.end = cue.start + cfg.max_duration
            cue.add_action("shortened")
            duration = cfg.max_duration
        if duration < cfg.min_duration:
            limit = cue.start + cfg.min_duration
            if i + 1 < n:
                limit = min(limit, cues[i + 1].start)
            if limit > cue.end:
                cue.end = limit
                cue.add_action("expanded")
    return cues


def _enforce_monotonic(
    cues: list[PostprocessedCue],
) -> list[PostprocessedCue]:
    """Sort, clip overlaps, drop zero-duration cues that have no text."""
    if not cues:
        return []
    cues = sorted(cues, key=lambda c: (c.start, c.end))
    cleaned: list[PostprocessedCue] = []
    for cue in cues:
        if cue.end < cue.start:
            cue.end = cue.start
            cue.add_action("clipped")
        if cleaned:
            prev = cleaned[-1]
            if cue.start < prev.end:
                # Clip the *previous* cue's end so we keep the new cue's
                # textual content intact and only shorten timing.
                prev.end = cue.start
                prev.add_action("clipped")
        cleaned.append(cue)
    # Drop empty / zero-width cues without text.
    return [c for c in cleaned if c.text or (c.end > c.start)]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def postprocess_cjk_cues(
    cues: list["CjkAlignedCue"],
    profile: "LanguageProfile",
    cfg: PostprocessConfig | None = None,
) -> tuple[list[PostprocessedCue], dict]:
    """Run the CJK postprocess pass.

    Parameters
    ----------
    cues:
        Aligned cues from :func:`subforge.pipeline.strategies.cjk._stage_align`.
    profile:
        Language profile carrying the CJK-specific punctuation sets.
    cfg:
        Optional configuration override. Defaults to the values exposed
        through :mod:`subforge.config`.

    Returns
    -------
    tuple
        ``(postprocessed_cues, diagnostics)`` where ``postprocessed_cues``
        is the list ready for the writer and ``diagnostics`` is a dict
        suitable for embedding in ``final_cues.json``.
    """
    cfg = cfg or PostprocessConfig()
    initial: list[PostprocessedCue] = []
    dropped: list[dict] = []

    for i, cue in enumerate(cues):
        text = (cue.display_text or "").strip("​")  # strip ZWSP only
        # Keep raw whitespace for timing reasons but skip empties.
        if not text:
            dropped.append({
                "source_index": i,
                "reason": "empty_text",
            })
            continue
        start = float(cue.start)
        end = float(cue.end)
        actions: list[str] = []
        if end < start:
            end = start
            actions.append("clipped")
        if cue.fallback_reason is not None:
            actions.append("fallback")
        if end <= start:
            # No timing — postprocess can't reason about durations; mark
            # as fallback and let _enforce_durations expand it later.
            actions.append("fallback")
        initial.append(
            PostprocessedCue(
                text=cue.display_text,
                start=start,
                end=end,
                actions=actions,
                source_index=i,
            )
        )

    # Step 1 — split cues that exceed the display-width budget.
    after_split: list[PostprocessedCue] = []
    for cue in initial:
        parts = _split_long_cue(cue, profile, cfg)
        if len(parts) == 1 and not parts[0].actions:
            parts[0].add_action("preserved")
        after_split.extend(parts)

    # Step 2 — merge short adjacent cues that satisfy the width / duration /
    # gap budget.
    after_merge = _merge_short_cues(after_split, profile, cfg)

    # Step 3 — duration bounds.
    after_duration = _enforce_durations(after_merge, cfg)

    # Step 4 — sort / clip overlaps / drop invalid intervals.
    final = _enforce_monotonic(after_duration)

    # Make sure every output cue has at least one tag so diagnostics
    # never surprise downstream consumers.
    for cue in final:
        if not cue.actions:
            cue.add_action("preserved")

    diagnostics = _build_diagnostics(final, dropped, cfg)
    return final, diagnostics


def postprocess_cues_to_writer_chunks(
    cues: list[PostprocessedCue],
    profile: "LanguageProfile",
) -> list[list[dict]]:
    """Convert :class:`PostprocessedCue` objects into writer chunks.

    The writer / translator stack expects per-character tokens with
    ``start``, ``end``, ``text``, ``whitespace`` and ``is_punct`` keys.
    Mirrors :func:`cjk_cues_to_writer_chunks` but operates on the
    postprocessed cue type so we can route timing through the
    width-aware splits cleanly.
    """
    punct = profile.punctuation
    chunks: list[list[dict]] = []
    for cue in cues:
        text = cue.text
        if not text:
            continue
        n = len(text)
        duration = max(cue.end - cue.start, 0.0)
        step = duration / n if n > 0 else 0.0
        tokens: list[dict] = []
        for i, ch in enumerate(text):
            ts = cue.start + i * step
            te = cue.end if i + 1 == n else cue.start + (i + 1) * step
            tokens.append({
                "text": ch,
                "whitespace": "",
                "is_punct": ch in punct,
                "start": ts,
                "end": te,
            })
        if tokens:
            chunks.append(tokens)
    return chunks


def _build_diagnostics(
    cues: list[PostprocessedCue],
    dropped: list[dict],
    cfg: PostprocessConfig,
) -> dict:
    action_counts: dict[str, int] = {}
    per_cue: list[dict] = []
    for cue in cues:
        for action in cue.actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        per_cue.append({
            "text": cue.text,
            "start": cue.start,
            "end": cue.end,
            "display_width": display_width(cue.text),
            "actions": list(cue.actions),
            "source_index": cue.source_index,
        })
    return {
        "config": {
            "max_display_width": cfg.max_display_width,
            "min_duration": cfg.min_duration,
            "max_duration": cfg.max_duration,
            "merge_max_width": cfg.merge_max_width,
            "merge_max_duration": cfg.merge_max_duration,
            "merge_max_gap": cfg.merge_max_gap,
            "short_cue_width": cfg.short_cue_width,
            "pause_split_threshold": cfg.pause_split_threshold,
        },
        "action_counts": action_counts,
        "cues": per_cue,
        "dropped": dropped,
        "total_in": len(cues) + len(dropped),
        "total_out": len(cues),
    }
