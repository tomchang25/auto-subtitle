"""Language-agnostic data contract for the subtitle pipeline.

The pipeline does not assume a single ASR output is both the transcript
and the timing source. These dataclasses make that boundary explicit:

* :class:`Transcript`     — text + provenance (ASR raw, corrector, future
                            SenseVoice, …)
* :class:`TimingAnchors`  — per-character timing track + provenance
* :class:`Sentence`       — sentence text with offsets back into the
                            transcript it was split from
* :class:`TokenInterval`  — typed per-token timing entry, used as the
                            optional ``tokens`` payload on aligned cues
* :class:`AlignedCue`     — sentence + timing interval + display/raw text
                            + optional per-token timing + confidence and
                            fallback metadata
* :class:`PipelineResult` — aggregated, debuggable view of a pipeline run

These types are used internally by the language strategies. The legacy
CJK names (``CjkTranscript``, ``CjkTimingAnchors``, …) live in
:mod:`subforge.pipeline.strategies.cjk_models` as compatibility aliases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# Recognised values for ``TimingAnchors.status`` and ``AlignedCue.timing_status``.
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
class TimingAnchor:
    """Per-character timing interval and its provenance."""

    start: float
    end: float
    source: str  # e.g. "word_segments" | "interpolated" | "join_token"

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "source": self.source}


@dataclass
class TimingAnchors:
    """A timing track with the text it is parallel to.

    ``anchors[i]`` is the timing interval for character ``text[i]``. When the
    transcript and timing come from the same backend (Whisper-only), ``text``
    matches :attr:`Transcript.text`. When they come from different backends
    (SenseVoice transcript + Whisper timing), ``text`` is the timing-side
    string and downstream stages must align transcript sentences to it
    before reading anchors.

    ``status`` describes the granularity of the underlying source so later
    stages can degrade gracefully without recomputing it.
    """

    anchors: list[TimingAnchor]
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
    def from_dict(cls, data: dict) -> "TimingAnchors":
        return cls(
            anchors=[
                TimingAnchor(a["start"], a["end"], a["source"])
                for a in data.get("anchors", [])
            ],
            source=data.get("source", ""),
            status=data.get("status", "missing"),
            text=data.get("text", ""),
            char_to_word=list(data.get("char_to_word", [])),
        )


@dataclass
class Transcript:
    """A textual transcript with provenance."""

    text: str
    source: str  # "asr_raw" | "corrector" | "sensevoice" | …

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls(text=data.get("text", ""), source=data.get("source", ""))


@dataclass
class Sentence:
    """A sentence-sized text unit with offsets in its source transcript."""

    text: str
    char_start: int  # inclusive offset in source transcript
    char_end: int    # exclusive offset in source transcript

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Sentence":
        return cls(
            text=data["text"],
            char_start=int(data["char_start"]),
            char_end=int(data["char_end"]),
        )


@dataclass
class TokenInterval:
    """Typed per-token timing entry.

    Used as the optional ``tokens`` payload on :class:`AlignedCue` so future
    English token-aware cues can carry word-level timing without falling
    back to raw dicts. ``source`` records where the timing came from
    (e.g. ``"asr_word"``, ``"char_split"``, ``"force_aligned"``).
    """

    text: str
    start: float
    end: float
    is_punct: bool = False
    whitespace: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TokenInterval":
        return cls(
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
            is_punct=bool(data.get("is_punct", False)),
            whitespace=data.get("whitespace", ""),
            source=data.get("source", ""),
        )


@dataclass
class AlignedCue:
    """A sentence with timing, display text, and fallback metadata.

    ``display_text`` is what the subtitle writer should ultimately show. It
    defaults to ``corrected_text`` when correction was applied and alignment
    succeeded, and falls back to ``raw_text`` otherwise. ``raw_text`` is the
    best-effort reconstruction of what the original ASR said for this span,
    derived from the corrected→raw character mapping.

    ``tokens`` is an optional list of typed per-token timing entries. When
    omitted (the default) the cue serializes without a ``tokens`` key, so
    artifacts produced by callers that do not populate per-token timing
    keep their existing JSON shape.
    """

    raw_text: str
    corrected_text: str
    display_text: str
    start: float
    end: float
    confidence: float
    fallback_reason: str | None
    text_source: str  # "corrected" | "raw"
    timing_source: str  # mirrors TimingAnchors.source
    timing_status: str  # one of TIMING_STATUSES
    tokens: list[TokenInterval] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.tokens is None:
            d.pop("tokens", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "AlignedCue":
        tokens_data = data.get("tokens")
        tokens = (
            [TokenInterval.from_dict(t) for t in tokens_data]
            if tokens_data is not None
            else None
        )
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
            tokens=tokens,
        )


@dataclass
class PipelineResult:
    """Aggregated, debuggable view of a pipeline run."""

    cues: list[AlignedCue]
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
# Adapters
# ---------------------------------------------------------------------------


def word_segments_to_inputs(
    word_segments: list[dict],
    join_token: str,
) -> tuple[Transcript, TimingAnchors]:
    """Adapter from ASR ``word_segments`` to a transcript + timing track.

    Splits a single ASR output into two first-class artifacts so transcript
    and timing can later come from different backends without changing the
    pipeline. Per-character timings are interpolated linearly inside each
    word interval; the source is recorded as ``"word_segments"`` so the
    timing status is :data:`"word_timing"`.
    """
    text_parts: list[str] = []
    anchors: list[TimingAnchor] = []
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
                anchors.append(TimingAnchor(anchor_t, anchor_t, "join_token"))
                char_to_word.append(-1)

        if not word:
            continue

        n = len(word)
        step = (end - start) / n if n > 0 and end > start else 0.0
        for i, ch in enumerate(word):
            ch_start = start + i * step if step > 0 else start
            ch_end = start + (i + 1) * step if step > 0 else end
            text_parts.append(ch)
            anchors.append(TimingAnchor(ch_start, ch_end, "word_segments"))
            char_to_word.append(w_idx)

    text = "".join(text_parts)
    transcript = Transcript(text=text, source="asr_raw")
    timing = TimingAnchors(
        anchors=anchors,
        source="word_segments",
        status="word_timing" if anchors else "missing",
        text=text,
        char_to_word=char_to_word,
    )
    return transcript, timing


def build_split_inputs(
    word_segments: list[dict],
    transcript_text: str,
    transcript_source: str,
    join_token: str,
) -> tuple[Transcript, TimingAnchors]:
    """Build inputs from a separate transcript backend and timing backend.

    Used when the transcript text comes from one backend (e.g. SenseVoice)
    and the timing anchors come from another (e.g. Whisper word_segments).
    The returned :class:`TimingAnchors` carries its own ``text`` field —
    the Whisper-side string that the anchors are parallel to — so later
    alignment stages can map transcript sentences onto timing positions
    without assuming the two strings are identical.
    """
    _whisper_transcript, timing = word_segments_to_inputs(
        word_segments, join_token=join_token
    )
    transcript = Transcript(
        text=transcript_text or "",
        source=transcript_source or "asr_raw",
    )
    return transcript, timing
