"""Internal data contract for the CJK subtitle pipeline.

The CJK strategy does not assume a single ASR output is both the transcript
and the timing source. These dataclasses make that boundary explicit:

* :class:`CjkTranscript`     — text + provenance (ASR raw, corrector, future
                               SenseVoice, …)
* :class:`CjkTimingAnchors`  — per-character timing track + provenance
* :class:`CjkSentence`       — sentence text with offsets back into the
                               transcript it was split from
* :class:`CjkAlignedCue`     — sentence + timing interval + display/raw text
                               + confidence and fallback metadata
* :class:`CjkPipelineResult` — aggregated, debuggable view of a pipeline run

These types are internal — nothing outside the CJK strategy imports them
yet. They exist so that adding SenseVoice (transcript-only) or Whisper
(timing-only) backends in later plans does not need to re-shape the contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subforge.nlp.lang_profile import LanguageProfile


# Recognised values for ``CjkTimingAnchors.status`` and ``CjkAlignedCue.timing_status``.
# Documented as a public set so later Whisper / forced-alignment backends have
# a stable vocabulary to land on.
TIMING_STATUSES: frozenset[str] = frozenset({
    "word_timing",        # per-word timestamps from ASR
    "char_timing",        # per-character timestamps (e.g. forced alignment)
    "segment_timing",     # only segment-level start/end available
    "uniform_estimated",  # interpolated within a coarser interval
    "missing",            # no timing information at all
    "invalid",            # timing present but rejected (non-monotonic, NaN, …)
    "fallback",           # final cue produced by a non-aligned fallback path
})


@dataclass
class CjkTimingAnchor:
    """Per-character timing interval and its provenance."""

    start: float
    end: float
    source: str  # e.g. "word_segments" | "interpolated" | "join_token"

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "source": self.source}


@dataclass
class CjkTimingAnchors:
    """A timing track with the text it is parallel to.

    ``anchors[i]`` is the timing interval for character ``text[i]``. When the
    transcript and timing come from the same backend (Whisper-only), ``text``
    matches :attr:`CjkTranscript.text`. When they come from different
    backends (SenseVoice transcript + Whisper timing), ``text`` is the
    timing-side string and downstream stages must align transcript sentences
    to it before reading anchors.

    ``status`` describes the granularity of the underlying source so later
    stages can degrade gracefully without recomputing it.
    """

    anchors: list[CjkTimingAnchor]
    source: str  # "word_segments" | "whisper" | "missing" | …
    status: str  # one of TIMING_STATUSES
    text: str = ""
    char_to_word: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "anchors": [a.to_dict() for a in self.anchors],
            "source": self.source,
            "status": self.status,
            "text": self.text,
            "char_to_word": list(self.char_to_word),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CjkTimingAnchors":
        return cls(
            anchors=[
                CjkTimingAnchor(a["start"], a["end"], a["source"])
                for a in data.get("anchors", [])
            ],
            source=data.get("source", ""),
            status=data.get("status", "missing"),
            text=data.get("text", ""),
            char_to_word=list(data.get("char_to_word", [])),
        )


@dataclass
class CjkTranscript:
    """A textual transcript with provenance."""

    text: str
    source: str  # "asr_raw" | "corrector" | "sensevoice" | …

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict) -> "CjkTranscript":
        return cls(text=data.get("text", ""), source=data.get("source", ""))


@dataclass
class CjkSentence:
    """A sentence-sized text unit with offsets in its source transcript."""

    text: str
    char_start: int  # inclusive offset in source transcript
    char_end: int    # exclusive offset in source transcript

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CjkSentence":
        return cls(
            text=data["text"],
            char_start=int(data["char_start"]),
            char_end=int(data["char_end"]),
        )


@dataclass
class CjkAlignedCue:
    """A sentence with timing, display text, and fallback metadata.

    ``display_text`` is what the subtitle writer should ultimately show. It
    defaults to ``corrected_text`` when correction was applied and alignment
    succeeded, and falls back to ``raw_text`` otherwise. ``raw_text`` is the
    best-effort reconstruction of what the original ASR said for this span,
    derived from the corrected→raw character mapping.
    """

    raw_text: str
    corrected_text: str
    display_text: str
    start: float
    end: float
    confidence: float
    fallback_reason: str | None
    text_source: str  # "corrected" | "raw"
    timing_source: str  # mirrors CjkTimingAnchors.source
    timing_status: str  # one of TIMING_STATUSES

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CjkAlignedCue":
        return cls(
            raw_text=data["raw_text"],
            corrected_text=data["corrected_text"],
            display_text=data["display_text"],
            start=float(data["start"]),
            end=float(data["end"]),
            confidence=float(data["confidence"]),
            fallback_reason=data.get("fallback_reason"),
            text_source=data["text_source"],
            timing_source=data["timing_source"],
            timing_status=data["timing_status"],
        )


@dataclass
class CjkPipelineResult:
    """Aggregated, debuggable view of a CJK pipeline run."""

    cues: list[CjkAlignedCue]
    text_source: str
    timing_source: str
    timing_status: str
    fallback_used: bool
    fallback_reason: str | None

    def to_dict(self) -> dict:
        return {
            "cues": [c.to_dict() for c in self.cues],
            "text_source": self.text_source,
            "timing_source": self.timing_source,
            "timing_status": self.timing_status,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def word_segments_to_cjk_inputs(
    word_segments: list[dict],
    join_token: str,
) -> tuple[CjkTranscript, CjkTimingAnchors]:
    """Adapter from ASR ``word_segments`` to a CJK transcript + timing track.

    Splits a single ASR output into two first-class artifacts so transcript
    and timing can later come from different backends without changing the
    pipeline. Per-character timings are interpolated linearly inside each
    word interval; the source is recorded as ``"word_segments"`` so the
    timing status is :data:`"word_timing"`.
    """
    text_parts: list[str] = []
    anchors: list[CjkTimingAnchor] = []
    char_to_word: list[int] = []

    for w_idx, seg in enumerate(word_segments):
        word = seg.get("word", "") or ""
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        if end < start:
            end = start

        if w_idx > 0 and join_token:
            anchor_t = anchors[-1].end if anchors else start
            for ch in join_token:
                text_parts.append(ch)
                anchors.append(CjkTimingAnchor(anchor_t, anchor_t, "join_token"))
                char_to_word.append(-1)

        if not word:
            continue

        n = len(word)
        step = (end - start) / n if n > 0 and end > start else 0.0
        for i, ch in enumerate(word):
            ch_start = start + i * step if step > 0 else start
            ch_end = start + (i + 1) * step if step > 0 else end
            text_parts.append(ch)
            anchors.append(CjkTimingAnchor(ch_start, ch_end, "word_segments"))
            char_to_word.append(w_idx)

    text = "".join(text_parts)
    transcript = CjkTranscript(text=text, source="asr_raw")
    timing = CjkTimingAnchors(
        anchors=anchors,
        source="word_segments",
        status="word_timing" if anchors else "missing",
        text=text,
        char_to_word=char_to_word,
    )
    return transcript, timing


def build_split_cjk_inputs(
    word_segments: list[dict],
    transcript_text: str,
    transcript_source: str,
    join_token: str,
) -> tuple[CjkTranscript, CjkTimingAnchors]:
    """Build CJK inputs from a separate transcript backend and timing backend.

    Used when the transcript text comes from one backend (e.g. SenseVoice)
    and the timing anchors come from another (e.g. Whisper word_segments).
    The returned :class:`CjkTimingAnchors` carries its own ``text`` field —
    the Whisper-side string that the anchors are parallel to — so later
    alignment stages can map transcript sentences onto timing positions
    without assuming the two strings are identical.
    """
    _whisper_transcript, timing = word_segments_to_cjk_inputs(
        word_segments, join_token=join_token
    )
    transcript = CjkTranscript(
        text=transcript_text or "",
        source=transcript_source or "asr_raw",
    )
    return transcript, timing


# ---------------------------------------------------------------------------
# Writer-compatibility bridge
# ---------------------------------------------------------------------------


def cjk_cues_to_writer_chunks(
    cues: list[CjkAlignedCue],
    profile: "LanguageProfile",
) -> list[list[dict]]:
    """Convert aligned CJK cues into the writer's per-character token format.

    The shared length-split / short-merge passes operate on
    ``list[list[token]]`` where every token has ``text``, ``start``, ``end``
    and ``is_punct``. Rather than reusing English-style word tokens, each
    cue's ``display_text`` is exploded into per-character tokens with timing
    distributed uniformly across ``[cue.start, cue.end]``. Cues with no
    display text are skipped.
    """
    punct = profile.punctuation
    chunks: list[list[dict]] = []
    for cue in cues:
        text = cue.display_text
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
