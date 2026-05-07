"""spaCy-driven sentence splitter for the English pipeline.

The splitter returns both :class:`Sentence` objects (for the staged
pipeline contract) and the raw spaCy token chunks (needed by word-level
alignment). The English policy captures the latter on the policy
instance so that alignment can walk word-level timing onto each token.
"""

from __future__ import annotations

from subforge.nlp.text_semantically import split_to_sentences
from subforge.pipeline.stages.models import Sentence


# ``split_to_sentences`` hardcodes its punctuation chunk limit. Mirror it
# here so the English policy's ``split_signature`` can fold the value
# into the cache key without re-importing the constant.
_SPACY_PUNCT_LIMIT = 5


def split_spacy(
    corrected_text: str,
) -> tuple[list[Sentence], list[list[dict]]]:
    """Split *corrected_text* into spaCy sentences and token chunks.

    Returns a pair of ``(sentences, token_chunks)``. ``token_chunks`` is
    the spaCy ``list[list[token_dict]]`` payload needed by alignment;
    ``sentences`` is the materialised :class:`Sentence` view consumed by
    the rest of the staged pipeline.
    """
    token_chunks = split_to_sentences(corrected_text)
    sentences = sentences_from_token_chunks(token_chunks, corrected_text)
    return sentences, token_chunks


def sentences_from_token_chunks(
    token_chunks: list[list[dict]],
    transcript_text: str,
) -> list[Sentence]:
    """Materialise :class:`Sentence` objects from spaCy token chunks.

    The spaCy split returns ``list[list[token_dict]]`` with ``text`` and
    ``whitespace`` per token. We reconstruct each sentence text by
    concatenating those, then locate the substring in the transcript by
    forward scanning so duplicate sentences resolve to distinct offsets.
    Failure to find the reconstructed sentence indicates a bug in the
    splitter's invariants and is raised loudly rather than silently
    producing wrong offsets.
    """
    sentences: list[Sentence] = []
    offset = 0
    for chunk in token_chunks:
        sent_text = "".join(t["text"] + t["whitespace"] for t in chunk)
        idx = transcript_text.find(sent_text, offset)
        if idx < 0:
            raise ValueError(
                "Reconstructed sentence not found in transcript at "
                f"offset {offset}: {sent_text!r:.80}"
            )
        sentences.append(
            Sentence(
                text=sent_text,
                char_start=idx,
                char_end=idx + len(sent_text),
            )
        )
        offset = idx + len(sent_text)
    return sentences


__all__ = [
    "split_spacy",
    "sentences_from_token_chunks",
    "_SPACY_PUNCT_LIMIT",
]
