"""Rule-based CJK sentence-boundary restoration.

CJK ASR backends (notably SenseVoice) sometimes emit a long unpunctuated
transcript span. The rest of the CJK pipeline assumes sentence-ish
boundaries before splitting and aligning, so a missing run of punctuation
causes downstream stages to either produce one giant cue or to recover
through late hard cuts.

This module provides a deterministic, language-aware restorer that only
*inserts* punctuation: it never deletes, replaces, or otherwise rewrites
the speaker's words. Boundaries can be triggered by:

* ``existing`` — the transcript already had the punctuation; the mark is
  recorded for diagnostics but no character is inserted.
* ``pause``    — the timing track shows a silence gap large enough to be
  treated as a phrase or sentence break.
* ``length``   — the running character budget exceeded
  :attr:`BoundaryConfig.soft_phrase_chars` or
  :attr:`BoundaryConfig.max_sentence_chars`.
* ``fallback`` — the input ended without any sentence-end character;
  one is appended so downstream sentence splitting always sees a
  terminator.

When :attr:`BoundaryConfig.mode` is ``"none"`` the restorer is a pure
no-op, so the CJK pipeline can still be exercised without restoration
for benchmark / debugging runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subforge.nlp.lang_profile import LanguageProfile


BOUNDARY_REASONS: frozenset[str] = frozenset({
    "existing",  # punctuation was already in the input transcript
    "pause",     # timing-side silence gap triggered insertion
    "length",    # length budget exceeded; phrase or sentence punct inserted
    "fallback",  # trailing sentence-end appended because none existed
})


BOUNDARY_MODES: frozenset[str] = frozenset({"rule", "none"})


@dataclass
class BoundaryMark:
    """A single boundary decision recorded for diagnostics."""

    position: int   # index in the *output* text of the punctuation char
    reason: str     # one of BOUNDARY_REASONS
    char: str       # the punctuation character itself
    inserted: bool  # True if the restorer added it; False if pre-existing


@dataclass
class BoundaryConfig:
    """Knobs for the rule-based boundary restorer."""

    mode: str = "rule"
    max_sentence_chars: int = 30
    soft_phrase_chars: int = 15
    pause_threshold: float = 0.6
    min_chars_between_breaks: int = 4
    # When None, the punct chars are derived from the language profile so
    # Korean (Latin-style punct) does not receive Chinese full-width punct.
    sentence_punct: str | None = None
    phrase_punct: str | None = None


def _resolve_punct(
    cfg: BoundaryConfig,
    profile: "LanguageProfile",
) -> tuple[str, str]:
    sentence = cfg.sentence_punct
    if sentence is None:
        if "。" in profile.sentence_end:
            sentence = "。"
        elif "." in profile.sentence_end:
            sentence = "."
        else:
            sentence = next(iter(profile.sentence_end), ".")
    phrase = cfg.phrase_punct
    if phrase is None:
        non_terminal = profile.punctuation - profile.sentence_end
        if "，" in non_terminal:
            phrase = "，"
        elif "," in non_terminal:
            phrase = ","
        else:
            phrase = next(iter(non_terminal), ",")
    return sentence, phrase


def _empty_diagnostics(cfg: BoundaryConfig, applied: bool) -> dict:
    return {
        "applied": applied,
        "mode": cfg.mode,
        "config": _config_snapshot(cfg),
        "reason_counts": {},
        "marks": [],
        "total_marks": 0,
        "inserted_count": 0,
    }


def _config_snapshot(cfg: BoundaryConfig) -> dict:
    return {
        "mode": cfg.mode,
        "max_sentence_chars": cfg.max_sentence_chars,
        "soft_phrase_chars": cfg.soft_phrase_chars,
        "pause_threshold": cfg.pause_threshold,
        "min_chars_between_breaks": cfg.min_chars_between_breaks,
        "sentence_punct": cfg.sentence_punct,
        "phrase_punct": cfg.phrase_punct,
    }


def restore_cjk_boundaries(
    text: str,
    profile: "LanguageProfile",
    *,
    gap_after: list[float | None] | None = None,
    cfg: BoundaryConfig | None = None,
) -> tuple[str, list[BoundaryMark], dict]:
    """Insert CJK sentence/phrase punctuation into *text*.

    Parameters
    ----------
    text:
        The transcript to restore. Returned verbatim when ``cfg.mode`` is
        ``"none"`` or when the text is empty.
    profile:
        The :class:`LanguageProfile` used for punctuation classification.
    gap_after:
        Optional list parallel to *text* where ``gap_after[i]`` is the
        silence gap (seconds) between ``text[i]`` and ``text[i + 1]``.
        Used to trigger pause-based boundaries when available.
    cfg:
        Optional configuration override; defaults to :class:`BoundaryConfig`.

    Returns
    -------
    tuple
        ``(restored_text, marks, diagnostics)`` where ``marks`` is the
        ordered list of :class:`BoundaryMark` records and ``diagnostics``
        is a dict suitable for embedding in a stage artifact.
    """
    cfg = cfg or BoundaryConfig()
    if cfg.mode not in BOUNDARY_MODES:
        raise ValueError(f"Unknown boundary mode: {cfg.mode!r}")
    if cfg.mode == "none" or not text:
        return text, [], _empty_diagnostics(cfg, applied=False)

    sentence_punct, phrase_punct = _resolve_punct(cfg, profile)

    out_chars: list[str] = []
    marks: list[BoundaryMark] = []
    sentence_chars = 0  # since the last sentence-end (existing or inserted)
    phrase_chars = 0    # since the last break of any kind

    for i, ch in enumerate(text):
        out_chars.append(ch)
        sentence_chars += 1
        phrase_chars += 1

        if ch in profile.sentence_end:
            marks.append(BoundaryMark(
                position=len(out_chars) - 1,
                reason="existing",
                char=ch,
                inserted=False,
            ))
            sentence_chars = 0
            phrase_chars = 0
            continue

        if ch in profile.punctuation:
            marks.append(BoundaryMark(
                position=len(out_chars) - 1,
                reason="existing",
                char=ch,
                inserted=False,
            ))
            # Phrase-level existing punct does not reset the sentence budget.
            phrase_chars = 0
            continue

        # Don't insert a break right before the end of the input — the
        # fallback rule will append a terminator if needed.
        if i + 1 >= len(text):
            continue
        # Don't double up — if the next char is itself punctuation, let
        # it fire on its own iteration as ``existing``.
        if text[i + 1] in profile.punctuation:
            continue
        if phrase_chars < cfg.min_chars_between_breaks:
            continue

        gap = None
        if gap_after is not None and 0 <= i < len(gap_after):
            gap = gap_after[i]

        if gap is not None and gap >= cfg.pause_threshold:
            if sentence_chars >= cfg.soft_phrase_chars:
                punct = sentence_punct
                out_chars.append(punct)
                marks.append(BoundaryMark(
                    position=len(out_chars) - 1,
                    reason="pause",
                    char=punct,
                    inserted=True,
                ))
                sentence_chars = 0
                phrase_chars = 0
            else:
                punct = phrase_punct
                out_chars.append(punct)
                marks.append(BoundaryMark(
                    position=len(out_chars) - 1,
                    reason="pause",
                    char=punct,
                    inserted=True,
                ))
                phrase_chars = 0
            continue

        if sentence_chars >= cfg.max_sentence_chars:
            punct = sentence_punct
            out_chars.append(punct)
            marks.append(BoundaryMark(
                position=len(out_chars) - 1,
                reason="length",
                char=punct,
                inserted=True,
            ))
            sentence_chars = 0
            phrase_chars = 0
            continue

        if phrase_chars >= cfg.soft_phrase_chars:
            punct = phrase_punct
            out_chars.append(punct)
            marks.append(BoundaryMark(
                position=len(out_chars) - 1,
                reason="length",
                char=punct,
                inserted=True,
            ))
            phrase_chars = 0

    out_text = "".join(out_chars)

    # Fallback: ensure the output ends with a sentence terminator so the
    # sentence splitter always sees one. Empty text is handled above.
    if out_text and out_text[-1] not in profile.sentence_end:
        out_text += sentence_punct
        marks.append(BoundaryMark(
            position=len(out_text) - 1,
            reason="fallback",
            char=sentence_punct,
            inserted=True,
        ))

    counts: dict[str, int] = {}
    for m in marks:
        counts[m.reason] = counts.get(m.reason, 0) + 1
    diagnostics = {
        "applied": True,
        "mode": cfg.mode,
        "config": _config_snapshot(cfg),
        "reason_counts": counts,
        "marks": [
            {
                "position": m.position,
                "reason": m.reason,
                "char": m.char,
                "inserted": m.inserted,
            }
            for m in marks
        ],
        "total_marks": len(marks),
        "inserted_count": sum(1 for m in marks if m.inserted),
        "input_length": len(text),
        "output_length": len(out_text),
    }
    return out_text, marks, diagnostics
