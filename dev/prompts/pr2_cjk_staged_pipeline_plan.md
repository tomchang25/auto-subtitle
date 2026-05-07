# PR2 Plan: Move CJK Processing onto the Shared Staged Pipeline

## Goal

Move the existing CJK subtitle processing path onto a shared staged pipeline while preserving current CJK behavior and keeping English untouched.

## Context

PR1 extracted the shared transcript-first data models and kept the old CJK model names available through compatibility aliases. PR2 should now move the CJK orchestration itself onto the shared pipeline skeleton.

The current CJK path already has a clear staged shape: input, correction, sentence splitting, alignment, postprocess, and final cues. The problem is that these stages are still owned by the CJK strategy as a separate end-to-end flow. This makes later work harder to share across languages.

PR2 should reduce that ownership: the CJK strategy should become a compatibility wrapper, and the actual execution should happen through a shared staged runner driven by CJK-specific policy behavior.

This is an extraction/refactor PR. It should not attempt to improve subtitle quality.

## High-Level Design

After this PR, the CJK runtime shape should be:

```text
CjkPipelineStrategy
→ CJK policy
→ shared staged runner
→ CJK stage implementations
→ writer-compatible subtitle chunks
```

The shared runner owns the stage order and artifact lifecycle.

The CJK policy owns CJK-specific behavior, including transcript/timing separation, no-op correction by default, sentence splitting, alignment, benchmark mode, fallback, and display-width-aware postprocess.

The existing public CJK strategy entry point should remain available so current callers do not need to change.

## Requirements

### 1. Shared staged runner

Introduce a shared runner that executes the CJK flow through explicit stages:

1. input
2. correction
3. sentence splitting
4. alignment
5. postprocess / final cues

The runner should be generic in structure and should not contain CJK-specific algorithms.

### 2. CJK policy

Represent CJK-specific behavior as a policy or equivalent configuration object.

The policy should provide only the behavior needed for the current CJK extraction. Avoid designing speculative future hooks for English or other languages in this PR.

### 3. CJK strategy compatibility

Keep the existing CJK strategy import and constructor behavior.

The CJK strategy should become a thin wrapper that binds the CJK policy and delegates execution to the shared staged runner.

### 4. Artifact migration

Introduce a canonical shared artifact location for staged pipeline outputs.

For compatibility, continue writing the existing CJK artifact files to the legacy CJK artifact location as a mirror.

The canonical staged artifacts should be the only cache source. The legacy CJK artifacts should be write-only compatibility output.

### 5. Behavior preservation

Preserve current CJK observable behavior, including:

- generated writer chunks
- SenseVoice transcript plus Whisper timing behavior
- no-op correction by default
- benchmark mode
- alignment fallback
- display-width-aware postprocess
- final cue metadata shape expected by existing tests

### 6. Cache safety

Bump the stage schema version so stale pre-refactor artifacts are ignored.

Force reruns should clear both canonical artifacts and legacy mirrors for known stage files.

Repeated cached runs should avoid unnecessary legacy mirror rewrites when contents have not changed.

### 7. English isolation

Do not migrate English in this PR.

The English strategy should remain behaviorally unchanged.

## Non-Goals

1. Do not add English staged pipeline support.
2. Do not add LLM correction.
3. Do not add glossary correction.
4. Do not add rule-based or neural punctuation restoration.
5. Do not redesign CJK alignment or postprocess behavior.
6. Do not add translation timing projection.
7. Do not remove legacy CJK artifact output.
8. Do not use the legacy CJK artifact location as a cache source.
9. Do not introduce speculative framework abstractions that are not needed for the CJK extraction.

## Implementation Guidance

Keep the extraction as small as possible.

Prefer moving existing CJK behavior into the staged runner and CJK policy over rewriting algorithms. If a protocol, hook, result type, or abstraction is not needed to preserve current CJK behavior, defer it.

Avoid creating a new “god object” that simply moves all of the old CJK strategy complexity into a larger CJK policy. The goal is to separate orchestration from language-specific behavior, not to add more abstraction than the current PR needs.

Do not normalize or improve final metadata shape in this PR. Preserve the current metadata contract unless a change is unavoidable and explicitly tested.

The runner should read only canonical staged artifacts. Legacy CJK artifacts should be mirrors for compatibility and debugging.

## Testing Expectations

Existing CJK tests should continue to pass with minimal or no changes.

Add focused tests for:

1. CJK runs produce canonical staged artifacts.
2. CJK runs still produce legacy CJK artifact mirrors.
3. The runner does not read legacy artifacts as cache input.
4. Stale pre-refactor artifacts are ignored after the schema version bump.
5. Force reruns refresh both canonical artifacts and legacy mirrors.
6. Benchmark mode still bypasses normal stages and preserves expected metadata.
7. Split transcript/timing behavior remains compatible.
8. Alignment fallback still returns writer-compatible output.
9. The CJK strategy remains constructible and dispatch-compatible.
10. English behavior remains unchanged.

Byte-identical output may be checked for stable deterministic fixtures, but it should not become a broad brittle requirement if behavior is otherwise preserved.

## Acceptance Criteria

1. CJK processing runs through the shared staged runner.
2. The CJK strategy is a compatibility wrapper rather than the owner of the full orchestration.
3. Canonical staged artifacts are produced for CJK runs.
4. Legacy CJK artifacts are still produced as mirrors.
5. The runner reads only canonical staged artifacts.
6. Current CJK benchmark, fallback, split-source transcript/timing, and postprocess behavior remain compatible.
7. English remains untouched.
8. Full test suite passes.

## Guardrails

- Keep PR2 focused on CJK orchestration extraction.
- Do not improve subtitle quality in this PR.
- Do not add new correction, punctuation, translation, or force-alignment behavior.
- Do not design the full final pipeline framework yet.
- Do not let shared models become a dumping ground for stage logic.
- If a helper belongs to stage execution, keep it with the staged pipeline rather than adding more logic to the model layer.
- If a behavior change is unavoidable, document it clearly and add a regression test.
