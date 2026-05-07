# Pre-Plan: Move CJK Processing onto the Shared Staged Pipeline

## 1. Goal

Route the existing CJK subtitle processing path through a shared staged pipeline so future language features can reuse the same orchestration model without changing current CJK behavior.

## 2. Why This Is Needed

The CJK path currently owns its own end-to-end orchestration even though the project is moving toward a shared transcript-first pipeline. Keeping CJK behavior isolated makes later correction, punctuation restoration, alignment, translation projection, and diagnostics work harder to apply consistently across languages. This step should extract the CJK flow into the shared staged model while preserving behavior.

## 3. Requirements

1. Shared orchestration — support running the existing CJK flow through a shared staged pipeline with explicit input, correction, sentence, alignment, postprocess, and final-output stages.
2. CJK policy — represent CJK-specific behavior as a language policy or equivalent stage configuration rather than as a separate end-to-end pipeline.
3. Compatibility wrapper — keep the existing CJK strategy entry point available so current callers and strategy dispatch behavior continue to work.
4. Artifact compatibility — write canonical shared stage artifacts while preserving the existing CJK artifact outputs for compatibility and debugging.
5. Cache/source-of-truth discipline — treat shared stage artifacts as the canonical cache source and avoid reading from legacy CJK artifact mirrors.
6. Behavior preservation — preserve current CJK output behavior, SenseVoice transcript plus Whisper timing behavior, benchmark mode, fallback behavior, and postprocess behavior.

## 4. Non-Goals

1. Do not migrate English to the shared staged pipeline in this plan.
2. Do not add LLM correction, glossary correction, punctuation restoration, or rule-based boundary restoration.
3. Do not redesign CJK alignment, postprocessing, fallback behavior, or generated subtitle quality.
4. Do not change translation behavior or translation timing projection.
5. Do not remove existing CJK public imports or legacy artifact outputs.

## 5. Acceptance Criteria

1. CJK processing runs through the shared staged orchestration while preserving current observable behavior.
2. Existing CJK strategy dispatch and construction remain compatible.
3. Shared stage artifacts are produced for CJK runs, and legacy CJK artifacts remain available.
4. Legacy CJK artifacts are not used as the canonical cache source.
5. Existing English behavior remains unchanged.
6. Existing CJK benchmark, split transcript/timing, fallback, and postprocess behavior remain compatible.
