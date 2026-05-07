# PR1 — Extract Shared Stage Models (No Behavior Changes)

Branch: `claude/extract-shared-stage-models-f9egB`

## Context

The CJK strategy already encodes a useful transcript-first data contract: transcript text, timing anchors, sentence units, aligned cues, pipeline result metadata, and adapters from ASR `word_segments` to those types. Today these live in a single CJK-named module (`pipeline/strategies/cjk_models.py`) and are only imported from CJK code paths.

The future staged subtitle pipeline needs the same vocabulary to be language-agnostic and to carry typed per-token timing information for the English path. PR1 is the structural prerequisite: lift the existing dataclasses into a shared module under language-agnostic names, add a typed `TokenInterval`, and keep every existing CJK import working through a thin compatibility shim. **No orchestration, alignment, postprocess, translation, artifact, or strategy behavior changes.** Later PRs will route English through the new types and add the staged runner.

## Lay of the land (from exploration)

Source of truth (321 LOC): `src/subforge/pipeline/strategies/cjk_models.py`
- Constant: `TIMING_STATUSES`
- Dataclasses: `CjkTimingAnchor`, `CjkTimingAnchors`, `CjkTranscript`, `CjkSentence`, `CjkAlignedCue`, `CjkPipelineResult`
- Adapters: `word_segments_to_cjk_inputs`, `build_split_cjk_inputs`
- Writer bridge: `cjk_cues_to_writer_chunks` (returns writer-format raw dicts; not part of the shared vocabulary)

Importers (only four sites, all keep their current import paths):
- `src/subforge/pipeline/strategies/cjk.py:79–86` — full bulk import
- `src/subforge/nlp/cjk_postprocess.py:38` — `CjkAlignedCue`, TYPE_CHECKING
- `tests/test_cjk_pipeline.py:17–24`
- `tests/test_cjk_postprocess.py:18`

No `tokens` field exists on cues today. No `pipeline/stages` package exists.

## Decisions confirmed with the user

1. Shared models live in a package: `src/subforge/pipeline/stages/` (`__init__.py` + `models.py`), leaving room for the staged runner in a later PR without a follow-up rename.
2. `TokenInterval` is re-exported from the `cjk_models` compatibility shim in addition to `pipeline/stages`.

## File plan

### NEW — `src/subforge/pipeline/stages/__init__.py`
Re-export the public stage-model surface so future callers can `from subforge.pipeline.stages import AlignedCue, TokenInterval, ...`.

### NEW — `src/subforge/pipeline/stages/models.py`
Language-agnostic dataclasses and adapters. Bodies copied verbatim from `cjk_models.py`, renamed and lightly adjusted:

| New name                  | Replaces / Notes                                            |
|---------------------------|-------------------------------------------------------------|
| `TIMING_STATUSES`         | moved as-is                                                 |
| `TimingAnchor`            | was `CjkTimingAnchor`                                       |
| `TimingAnchors`           | was `CjkTimingAnchors`                                      |
| `Transcript`              | was `CjkTranscript`                                         |
| `Sentence`                | was `CjkSentence`                                           |
| `TokenInterval`           | NEW — typed dataclass (see below)                           |
| `AlignedCue`              | was `CjkAlignedCue`, plus `tokens: list[TokenInterval] \| None = None` |
| `PipelineResult`          | was `CjkPipelineResult`                                     |
| `word_segments_to_inputs` | was `word_segments_to_cjk_inputs`                           |
| `build_split_inputs`      | was `build_split_cjk_inputs`                                |

Inside the bodies, replace the few internal `Cjk*` references (constructor calls inside `word_segments_to_inputs`, return types in `from_dict`, docstring references) with the new names. Field order, types, and runtime behavior are unchanged.

`TokenInterval`:

```python
@dataclass
class TokenInterval:
    text: str
    start: float
    end: float
    is_punct: bool = False
    whitespace: str = ""
    source: str = ""

    def to_dict(self) -> dict: return asdict(self)

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
```

`AlignedCue` extension — append `tokens` as the **last** field (positional safety) and adjust serialization to keep current JSON byte-identical when no tokens are present:

```python
tokens: list[TokenInterval] | None = None

def to_dict(self) -> dict:
    d = asdict(self)              # asdict recurses into TokenInterval
    if self.tokens is None:
        d.pop("tokens", None)     # preserve current artifact shape
    return d

@classmethod
def from_dict(cls, data: dict) -> "AlignedCue":
    tokens_data = data.get("tokens")
    tokens = (
        [TokenInterval.from_dict(t) for t in tokens_data]
        if tokens_data is not None else None
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
```

### MODIFIED — `src/subforge/pipeline/strategies/cjk_models.py`
Shrink to ~40 lines: a compatibility shim that re-exports shared symbols under the legacy `Cjk*` names plus the unchanged `cjk_cues_to_writer_chunks` helper (not part of the shared vocabulary; current tests import it from this exact path).

```python
from subforge.pipeline.stages.models import (
    TIMING_STATUSES,
    AlignedCue as CjkAlignedCue,
    PipelineResult as CjkPipelineResult,
    Sentence as CjkSentence,
    TimingAnchor as CjkTimingAnchor,
    TimingAnchors as CjkTimingAnchors,
    TokenInterval,
    Transcript as CjkTranscript,
    build_split_inputs as build_split_cjk_inputs,
    word_segments_to_inputs as word_segments_to_cjk_inputs,
)
# def cjk_cues_to_writer_chunks(...): body copied verbatim
__all__ = [
    "TIMING_STATUSES", "CjkAlignedCue", "CjkPipelineResult", "CjkSentence",
    "CjkTimingAnchor", "CjkTimingAnchors", "CjkTranscript", "TokenInterval",
    "build_split_cjk_inputs", "cjk_cues_to_writer_chunks",
    "word_segments_to_cjk_inputs",
]
```

Because the legacy names are aliases (`CjkAlignedCue is AlignedCue`), `isinstance` and `from_dict` resolve identically — `cjk.py` and the existing tests need zero changes.

### UNTOUCHED (verified, no edits)
- `src/subforge/pipeline/strategies/cjk.py` — keeps its current `from subforge.pipeline.strategies.cjk_models import (...)` block
- `src/subforge/pipeline/strategies/english.py`
- `src/subforge/pipeline/processor.py` (Non-Goal #8)
- `src/subforge/nlp/cjk_postprocess.py` (TYPE_CHECKING import resolves through the shim)
- `tests/test_cjk_pipeline.py`, `tests/test_cjk_postprocess.py`

## New tests — `tests/test_stage_models.py`

Six focused cases mapping directly to the PR's testing requirements:

1. `test_token_interval_roundtrip` — construct, `to_dict`, `from_dict`, defaults (`is_punct=False`, `whitespace=""`, `source=""`).
2. `test_aligned_cue_roundtrip_without_tokens` — `to_dict` omits `tokens`; `from_dict` of legacy dict (no `tokens` key) yields `tokens is None`. Locks current CJK JSON shape.
3. `test_aligned_cue_roundtrip_with_tokens` — round-trip preserves typed `TokenInterval` list; dataclass values compare equal.
4. `test_cjk_compat_aliases_resolve` — import each legacy name from `subforge.pipeline.strategies.cjk_models` and assert it `is` the shared class object (covers all dataclasses + `TokenInterval` + adapter helpers).
5. `test_word_segments_adapter_unchanged` — exercise the same fixture as `tests/test_cjk_pipeline.py`'s adapter test through the new shared name; assert identical text, anchor count, `char_to_word`, and timing status.
6. `test_build_split_inputs_unchanged` — mirror `test_build_split_cjk_inputs_separates_sources` through the shared name.

## Verification

1. `pytest tests/test_stage_models.py -q` — new tests pass.
2. `pytest tests/test_cjk_pipeline.py tests/test_cjk_postprocess.py -q` — existing CJK tests pass unchanged.
3. `pytest -q` — full suite green.
4. Sanity grep: `grep -rn "Cjk\(Transcript\|TimingAnchor\|Sentence\|AlignedCue\|PipelineResult\)\|word_segments_to_cjk_inputs\|build_split_cjk_inputs" src tests` — every hit must still resolve through the shim.
5. Manual diff-check: any committed cached `alignment.json` fixtures keep the same key set (no new `tokens` key for cues that didn't set one).
6. Commit on `claude/extract-shared-stage-models-f9egB`, push with `git push -u origin claude/extract-shared-stage-models-f9egB`. Do not open a PR (per global instructions).

## Acceptance-criteria mapping

| Criterion                                                   | Met by                                                |
|-------------------------------------------------------------|-------------------------------------------------------|
| Shared language-agnostic stage models exist                 | `pipeline/stages/models.py`                           |
| CJK imports remain backward-compatible                      | `cjk_models.py` shim, untouched call sites            |
| `AlignedCue` supports optional typed token intervals        | `tokens: list[TokenInterval] \| None = None`          |
| JSON round-trips with and without tokens                    | Updated `to_dict`/`from_dict`; tests 2 and 3          |
| Existing CJK tests pass without behavioral changes          | Untouched test files; tests 5 and 6 re-prove adapters |
| English pipeline behavior untouched                         | `english.py` and `processor.py` not modified          |
| No processor / strategy / artifact / correction / alignment / postprocess / translation / fallback changes | Scope limited to the two new files + shim shrinkage   |

## Guardrails honored

- No staged runner introduced.
- No language policies introduced.
- English path unchanged; no English imports of the new module in this PR.
- No artifact directory or filename changes.
- No opportunistic cleanup of unrelated code.
- `cjk_cues_to_writer_chunks` deliberately stays in the CJK shim — its raw-dict writer format is out of scope for the shared vocabulary.
