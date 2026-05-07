"""Punctuation-driven sentence splitter for the CJK pipeline.

CJK transcripts have no whitespace separators, so the splitter walks
the corrected text character-by-character and cuts whenever a profile
sentence-end character is encountered. The trailing fragment (if any)
becomes the final sentence.
"""

from __future__ import annotations

from subforge.pipeline.stages.models import Sentence


def split_by_punctuation(
    corrected_text: str,
    sentence_end: frozenset[str],
) -> list[Sentence]:
    """Split *corrected_text* at every *sentence_end* character.

    The terminating punctuation character is kept on the preceding
    sentence so the writer can reproduce it verbatim. Any trailing run
    without a terminator becomes one final sentence.
    """
    sentences: list[Sentence] = []
    start = 0
    for i, ch in enumerate(corrected_text):
        if ch in sentence_end:
            sentences.append(
                Sentence(corrected_text[start : i + 1], start, i + 1)
            )
            start = i + 1
    if start < len(corrected_text):
        sentences.append(
            Sentence(corrected_text[start:], start, len(corrected_text))
        )
    return sentences


__all__ = ["split_by_punctuation"]
