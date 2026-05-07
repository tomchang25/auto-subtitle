"""Sentence-to-timing alignment stage implementations.

* :mod:`char_level` — char-level alignment via :class:`difflib.SequenceMatcher`
  used by the CJK pipeline.
* :mod:`word_level` — word-level alignment via the spaCy/ASR token pipeline
  used by the English pipeline.

Each module exposes a single high-level entry point that takes the data
the policy can supply and returns the staged-pipeline aligned-cue list.
"""
