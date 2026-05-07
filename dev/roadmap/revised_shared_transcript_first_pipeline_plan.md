# Plan: Unify English and CJK Subtitle Processing into a Shared Transcript-First Staged Pipeline

## Context

The subtitle pipeline currently splits into two parallel orchestration paths after transcription:

- The English path owns a spaCy-based flow: sentence split → token-level timing attach → timing-gap refinement → word-count split → short-segment merge.
- The CJK path owns a transcript-first flow with explicit artifacts: raw transcript + timing anchors → correction seam → sentence split → alignment → display-width-aware postprocess.

This split makes future work difficult to add consistently. LLM correction, punctuation restoration, translation timing projection, force alignment, and richer diagnostics should target shared stages rather than separate end-to-end language pipelines.

Outcome: one staged orchestration scaffold, language-specific policies, consistent stage artifacts, and behavior-compatible subtitle output. Byte-identical output should be required only for targeted deterministic regression fixtures, not as a blanket requirement for every language and fixture.

## Architecture Decisions

1. **Shared staged pipeline skeleton.** English and CJK should both pass through the same high-level stage names: inputs, correction, sentence units, alignment, postprocess, and final cues.

2. **Policy-based language behavior.** Language-specific behavior should live in policies and stage implementations, not in separate end-to-end orchestration classes.

3. **Typed token intervals.** `AlignedCue` should include an optional `tokens` field, but the token type should be explicit rather than raw dictionaries. English populates token intervals; CJK may leave them empty and rely on cue interval plus display text.

4. **Consistent stage signatures.** Stage protocols should expose consistent signatures so the runner stays generic. Differences between English and CJK should be handled inside implementations, not through runner-level language branching.

5. **Canonical artifact directory.** Shared artifacts should be read from one canonical stage directory. Legacy CJK artifact output may be mirrored for compatibility, but the runner should not read caches from both locations.

6. **Strategy compatibility shims.** Existing `EnglishPipelineStrategy` and `CjkPipelineStrategy` should remain importable public entry points, but their orchestration bodies should become thin wrappers around the shared staged pipeline.

7. **Incremental PR slicing.** This refactor should land in small behavior-preserving slices rather than one large all-at-once rewrite.

## Shared Data Model

Introduce language-agnostic stage models:

- `Transcript`
- `TimingAnchor`
- `TimingAnchors`
- `SentenceUnit`
- `TokenInterval`
- `AlignedCue`
- `PostprocessResult`
- `PipelineResult`

`TokenInterval` should include at least:

- `text`
- `start`
- `end`
- `is_punct`
- `whitespace`
- `source`
- optional confidence or provenance metadata

`AlignedCue` should include at least:

- raw text
- corrected text
- display text
- start and end interval
- confidence
- fallback reason
- text source
- timing source
- timing status
- optional `tokens: list[TokenInterval] | None`

English writer output should treat `tokens` as the source of truth when available. `display_text` for English is primarily an artifact/debug view and must not introduce punctuation-spacing regressions.

## Shared Stage Contract

Each stage should return explicit data rather than passing hidden state through mutable context side channels.

Recommended stage shapes:

- Input stage: word segments plus optional transcript override → transcript and timing anchors.
- Correction stage: transcript plus policy → corrected transcript and correction metadata.
- Sentence stage: corrected transcript plus policy → sentence units.
- Alignment stage: sentence units, timing anchors, original word segments, and policy → aligned cues and alignment metadata.
- Postprocess stage: aligned cues and policy → final writer chunks plus postprocess diagnostics.

Do not pass English spaCy sentence tokens through generic context options. If sentence-level token data is needed by alignment, attach it to the sentence unit or an explicit sentence-stage result.

## Artifact Policy

Use a canonical shared artifact directory for both English and CJK runs.

Canonical artifact names:

- `raw_transcript.json`
- `timing_anchors.json`
- `corrected_transcript.json`
- `sentences.json`
- `alignment.json`
- `final_cues.json`

CJK may mirror these files to the legacy CJK artifact directory for backward compatibility, but that directory should be write-only from the runner’s perspective.

Rules:

1. The runner reads only canonical stage artifacts.
2. Legacy artifact directories are compatibility mirrors, not cache sources.
3. Force-clear and schema invalidation operate on the canonical directory first.
4. Legacy mirrors are overwritten from canonical artifacts after successful writes.
5. Stage schema version should be bumped so stale artifacts from previous layouts are ignored.

Minimum shared metadata should include:

- mode
- language/profile code
- text source
- timing source
- transcript backend
- timing backend
- correction mode
- correction applied
- alignment status
- fallback used
- fallback reason
- timing status
- postprocess action counts when postprocess ran

## Implementation Plan

### PR 1 — Shared models and compatibility re-exports

Create shared stage models and cache helpers without changing behavior.

Tasks:

1. Introduce shared transcript, timing, sentence, token, cue, and result models.
2. Add typed token interval support and JSON round-trip behavior.
3. Re-export existing CJK model names from the shared models so existing imports continue to work.
4. Keep the existing English and CJK strategy bodies unchanged.
5. Add focused tests for model serialization, including aligned cues with and without tokens.

Completion criteria:

- Existing tests pass unchanged.
- CJK model imports still resolve.
- New shared cue and token models can round-trip through JSON.

### PR 2 — Shared staged runner for CJK only

Move CJK orchestration onto the shared runner while preserving behavior and legacy artifacts.

Tasks:

1. Introduce the policy object and stage protocol definitions.
2. Implement the staged runner using the shared stage contract.
3. Extract existing CJK input, correction, sentence split, alignment, fallback, and display-width postprocess behavior into policy-driven stage implementations.
4. Keep `CjkPipelineStrategy` as a thin compatibility shim.
5. Write canonical artifacts to the shared stage directory and mirror them to the legacy CJK directory.
6. Ensure the runner reads only canonical artifacts.
7. Preserve Chinese benchmark short-circuit behavior and its existing metadata shape.

Completion criteria:

- Existing CJK tests pass unchanged.
- CJK artifacts appear in both canonical and legacy locations.
- Canonical artifacts are the only cache source.
- Benchmark mode still bypasses normal stages and records the same high-level intent.

### PR 3 — English policy through the staged runner

Route English through the shared runner without changing default behavior.

Tasks:

1. Implement English sentence splitting as a stage result that explicitly carries any token data needed by alignment.
2. Implement English word-level alignment that populates `AlignedCue.tokens`.
3. Implement English postprocess that reads token intervals and calls existing timing refinement, long-split, and short-merge primitives.
4. Keep `EnglishPipelineStrategy` as a thin compatibility shim.
5. Add English stage artifacts under the canonical stage directory.
6. Add targeted deterministic regression fixtures for behavior-compatible or byte-identical English output where appropriate.

Completion criteria:

- Existing English primitive tests pass.
- English staged runs produce canonical artifacts.
- Aligned English cues have populated token intervals.
- Writer output does not regress punctuation spacing.
- Deterministic English fixture output matches the legacy path where the fixture is stable enough for byte comparison.

### PR 4 — Cleanup duplicated orchestration and tighten diagnostics

After both languages route through the staged runner, remove duplicated orchestration logic and normalize diagnostics.

Tasks:

1. Remove old end-to-end strategy orchestration bodies after shims are stable.
2. Ensure postprocess diagnostics have a comparable shape across English and CJK.
3. Ensure alignment fallback metadata is consistent across language policies.
4. Audit stage metadata for required shared fields.
5. Update developer documentation to explain the staged pipeline and language policy model.

Completion criteria:

- Strategies are compatibility wrappers only.
- Runner does not contain language-specific orchestration branches except policy selection and explicitly documented benchmark gates.
- Final cue metadata is comparable across English and CJK.
- Future correction, punctuation, translation projection, and force-alignment work can target explicit stages.

## Language Policies

### English policy

English policy should provide:

- spaCy or equivalent sentence splitting
- word-token alignment from ASR word segments
- token-aware timing refinement
- token-aware long-cue splitting and short-cue merging
- no default transcript correction
- canonical artifact output only

English postprocess should preserve existing token timing behavior. When tokens are present, postprocess should use tokens rather than interpolating from cue-level intervals.

### CJK policy

CJK policy should provide:

- transcript/timing separation
- no-op correction by default
- punctuation-based sentence splitting from corrected transcript
- char-level or fuzzy text-to-timing alignment
- display-width-aware postprocess
- legacy artifact mirroring for compatibility

CJK should not require English-style word tokens. If tokens are unavailable, postprocess should operate on display text and cue interval.

## Fallback Policy

Fallback behavior should be policy-driven and should avoid reintroducing hidden alternate pipelines.

Rules:

1. Alignment failures should return structured fallback metadata.
2. Fallback output should flow through the same postprocess stage where possible.
3. Fallback should avoid writing invalid zero-duration cues as normal subtitles.
4. CJK fallback should not bypass display-width-aware postprocess by returning to legacy hard-cut behavior unless explicitly marked as a benchmark or emergency fallback.
5. English fallback may use punctuation-based word segment splitting when spaCy alignment fails, but it should still produce the shared result shape.

## Processor Integration

Avoid changing the main processor unless policy construction requires new options. The existing strategy selection surface should remain stable initially.

Current compatibility requirements:

- `get_strategy(profile, *, corrector=None)` remains available.
- `EnglishPipelineStrategy` remains importable.
- `CjkPipelineStrategy` remains importable.
- Existing SenseVoice transcript override fields in strategy context continue to work.
- Existing benchmark and force flags continue to work.

If future policy options are required, they should be added through explicit configuration and stage metadata rather than hidden context mutation.

## Non-Goals

1. Do not implement LLM transcript correction in this refactor.
2. Do not implement rule-based CJK boundary restoration in this refactor.
3. Do not add neural punctuation restoration in this refactor.
4. Do not implement force alignment in this refactor.
5. Do not redesign translation quality or translation prompts in this refactor.
6. Do not remove public strategy imports in the initial migration.
7. Do not make legacy CJK artifacts a second cache source.

## Risk Areas and Guardrails

### Risk: Hidden stage coupling through context mutation

Guardrail: Stage outputs must explicitly carry data needed by later stages. Avoid using generic context options as a data transport layer.

### Risk: English spacing regressions

Guardrail: English writer output should be derived from token intervals when tokens exist. Do not reconstruct English subtitle text with naive string joining.

### Risk: Cache confusion between canonical and legacy artifacts

Guardrail: Read only canonical stage artifacts. Treat legacy CJK artifact output as a compatibility mirror.

### Risk: Runner becomes a language switchboard

Guardrail: The runner should call policy stages through shared contracts. Any language-specific decision should live in policy construction or implementation classes.

### Risk: Refactor becomes too large

Guardrail: Land models, CJK runner, English runner, and cleanup as separate PRs with green tests at each step.

## Verification

1. Run the full existing test suite after each PR slice.
2. Add model serialization tests for typed token intervals and aligned cues.
3. Add CJK staged-run tests verifying canonical artifacts and legacy mirrors.
4. Add English staged-run tests verifying token population and canonical artifacts.
5. Add deterministic English fixture regression checks where byte equality is stable.
6. Add metadata checks for shared final cue fields across English and CJK.
7. Run an end-to-end CJK smoke test using split transcript/timing input and verify that invalid zero-duration cues are not emitted as normal subtitles.
8. Run an end-to-end English smoke test and verify that punctuation spacing and timing-gap behavior remain behavior-compatible with the legacy path.
