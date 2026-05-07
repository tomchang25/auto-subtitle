"""Sentence-splitter stage implementations.

Each module here implements one concrete sentence-splitting strategy:

* :mod:`spacy_splitter` — spaCy-driven splitting with token-chunk
  preservation (English).
* :mod:`punctuation` — character iteration over a sentence-end set
  (CJK).

Policies select and call the appropriate splitter; the splitters
themselves are pure functions that take explicit data and return
``Sentence`` objects.
"""
