# CJK Pipeline Rewrite Pre-Plans

These pre-plans are high-level feature boundaries intended to be converted into concrete Plan-mode implementation prompts later. They describe what should change and why, without prescribing exact files, APIs, or implementation details.

---

## Pre-Plan 1: Separate the CJK Pipeline from the Existing Subtitle Pipeline

### 1. Goal

Create a dedicated CJK subtitle pipeline so Chinese/Japanese/Korean processing no longer depends on the English-oriented word-timestamp segmentation flow.

### 2. Requirements

1. Pipeline separation — support a distinct CJK flow that is selected by detected or explicit source language without changing the existing English pipeline behavior.

2. Transcript-first architecture — replace the current CJK dependency on ASR word timestamps with a transcript-first flow that treats text quality and timing as separate concerns.

3. Stage artifacts — preserve intermediate CJK artifacts for raw transcript, corrected transcript, sentence list, alignment result, and final subtitle output.

4. Graceful degradation — support fallback behavior when correction, alignment, or timing data is unavailable, rather than failing the entire subtitle run.

5. Compatibility — final CJK output must still conform to the existing subtitle writer and cache/resumability concepts.

6. Refactor boundary — keep existing English segmentation, punctuation restoration, and word-level alignment logic untouched unless shared abstractions are required.

### 3. Status

Implemented. The CJK path has been separated from the English strategy and now has a transcript-first architecture with CJK-specific artifacts and fallback behavior.

---

## Pre-Plan 2: Add SenseVoice Transcript Backend with Whisper Timing Provider

### 1. Goal

Use SenseVoice as the primary CJK transcript source while using Whisper as the timing provider so CJK subtitles can combine better text quality with usable subtitle timestamps.

### 2. Why This Is Needed

SenseVoice does not provide reliable subtitle timestamps, so it cannot independently produce normal timed subtitles. The CJK pipeline separates transcript text from timing anchors, which makes it possible to use SenseVoice for text and Whisper for timing instead of forcing SenseVoice output into a fake word-timestamp format.

### 3. Requirements

1. SenseVoice transcript source — support SenseVoice as a CJK transcript provider that returns text without requiring word-level, character-level, or segment-level timestamps.

2. Whisper timing source — support Whisper as the default CJK timing provider when SenseVoice is used, with timing represented separately from transcript text.

3. CJK transcript/timing merge — align SenseVoice-derived sentences against Whisper timing anchors so final subtitles use SenseVoice text with Whisper start/end times.

4. Default CJK routing — prefer SenseVoice transcript plus Whisper timing for CJK when SenseVoice is available, with Whisper-only fallback when SenseVoice fails or produces unusable text.

5. Timing degradation — when Whisper timing is unavailable or invalid, mark timing as missing or estimated rather than emitting fake normal timestamps.

6. Diagnostics — persist transcript source, timing source, timing status, backend names, transcript length, fallback usage, and alignment outcome in CJK artifacts.

### 4. Non-Goals

1. Do not implement local LLM correction or glossary correction.

2. Do not implement forced audio alignment.

3. Do not redesign English or non-CJK transcription behavior.

4. Do not require SenseVoice to emit word-level timestamps.

5. Do not expose full CJK mode selection in CLI or UI yet.

### 5. Acceptance Criteria

1. A CJK run can use SenseVoice text as the final subtitle display text while taking subtitle timing from Whisper.

2. SenseVoice-only output is treated as untimed transcript data unless an explicit estimated-timing fallback is enabled.

3. Whisper-only fallback can still produce a usable timed subtitle when SenseVoice is unavailable, empty, or suspiciously incomplete.

4. CJK artifacts clearly distinguish transcript source from timing source.

5. No fake SenseVoice word timestamps are produced to satisfy legacy pipeline assumptions.

6. Existing English subtitle behavior remains unchanged.

### 6. Status

Implemented. SenseVoice text-only output is now used as the CJK transcript source, Whisper remains the timing source, fallback to Whisper-only is recorded, and split transcript/timing metadata is persisted.

---

## Pre-Plan 3: Add CJK Subtitle Postprocessing and Readable Output

### 1. Goal

Replace hard-cut CJK subtitle output with language-aware postprocessing so SenseVoice transcript plus Whisper timing can produce readable subtitles before heavier correction stages are added.

### 2. Why This Is Needed

The CJK transcript/timing split now produces usable text and timing, but the final subtitle output is still limited by hard cutting. Before adding LLM correction or more aggressive fuzzy alignment, the pipeline needs a stable postprocess layer that improves cue readability, preserves timing safety, and records formatting decisions for diagnostics.

### 3. Requirements

1. CJK cue postprocess — support display-width-aware splitting and merging for CJK subtitles instead of simple hard cuts.

2. Sentence-aware boundaries — prefer aligned sentence boundaries, punctuation, and pauses when deciding where subtitle cues should break.

3. Readability constraints — enforce configurable limits for display width, minimum duration, maximum duration, short-cue merging, and long-cue splitting.

4. Timing safety — preserve monotonic, non-overlapping subtitle intervals and prevent invalid cues from being written as normal subtitles.

5. Postprocess diagnostics — record whether each cue was preserved, split, merged, shortened, expanded, or fallback-generated.

6. Regression safety — preserve existing English and non-CJK output behavior.

### 4. Non-Goals

1. Do not add local LLM correction in this plan.

2. Do not add glossary correction in this plan.

3. Do not redesign SenseVoice or Whisper backend orchestration.

4. Do not implement forced audio alignment.

5. Do not expose full CJK mode selection in CLI or UI yet.

6. Do not change translation behavior.

### 5. Acceptance Criteria

1. Normal CJK output no longer relies on simple hard cuts for final subtitle generation.

2. Long CJK cues are split at readable punctuation, pause, or display-width boundaries.

3. Very short CJK cues are merged when timing gap, duration, and display-width constraints allow.

4. Final CJK SRT has valid, monotonic, non-overlapping timestamps.

5. Postprocess decisions are visible in CJK artifacts or diagnostics.

6. Existing English subtitle behavior remains unchanged.

---

## Pre-Plan 4: Add CJK Subtitle Diagnostics and Benchmark Artifacts

### 1. Goal

Preserve enough diagnostics to compare transcript source, timing source, alignment quality, postprocess decisions, and fallback behavior across CJK runs.

### 2. Why This Is Needed

Once CJK output is no longer just raw hard-cut text, users need to distinguish backend quality, alignment quality, and formatting quality. Diagnostics should make it clear whether an issue came from SenseVoice text, Whisper timing, fuzzy alignment, postprocessing, or fallback behavior.

### 3. Requirements

1. Raw artifact export — save raw transcript, timing anchors, sentence list, alignment results, postprocess results, and final subtitle output for debugging.

2. Benchmark report — generate metadata that compares backend selection, transcript length, timing status, alignment confidence, postprocess actions, and fallback behavior.

3. Mode clarity — distinguish benchmark/raw output, aligned output, postprocessed output, corrected output, and final subtitle output.

4. Invalid timing handling — keep ASR text available even when timing is invalid, but prevent invalid cues from being written as normal subtitles.

5. Fallback visibility — record when the pipeline used Whisper-only output, fallback timing, hard-cut output, or estimated timing.

6. Regression safety — preserve existing output behavior for English and non-CJK flows.

### 4. Non-Goals

1. Do not add LLM correction in this plan.

2. Do not add new ASR backends in this plan.

3. Do not expose full UI controls in this plan.

4. Do not implement forced alignment.

### 5. Acceptance Criteria

1. CJK runs produce enough artifacts to inspect transcript, timing, alignment, postprocess, and final output independently.

2. Benchmark metadata makes backend fallback and low-confidence alignment visible.

3. Invalid or estimated timing is clearly marked and never confused with normal timing.

4. Final subtitle output can be compared against raw and postprocessed intermediate output.

5. Existing English output remains unchanged.

---

## Pre-Plan 5: Add Glossary and Conservative Local LLM Correction for CJK

### 1. Goal

Add conservative CJK transcript correction so ASR errors, proper nouns, punctuation, and sentence boundaries can be improved without rewriting the speaker’s meaning.

### 2. Why This Is Needed

SenseVoice plus Whisper timing already produces usable baseline subtitles, but CJK ASR can still misrecognize names, technical terms, homophones, and punctuation. Correction should improve transcript quality while preserving alignment safety and avoiding aggressive rewriting.

### 3. Requirements

1. Glossary correction — support deterministic correction for known names, terms, products, games, and domain-specific vocabulary.

2. Conservative LLM correction — support local LLM correction for ASR misrecognitions, homophones, punctuation, and sentence boundaries without changing speaker meaning.

3. Sentence list output — require correction to produce stable ordered sentence units rather than direct SRT output.

4. Dual transcript input — allow correction to compare SenseVoice transcript with Whisper transcript when both are available.

5. Guardrails — detect excessive insertions, deletions, or rewriting so risky corrected text can be rejected or downgraded.

6. Deterministic fallback — if correction fails or is disabled, continue with raw transcript or glossary-only correction.

### 4. Non-Goals

1. Do not implement external hosted LLM providers in this plan.

2. Do not implement forced audio alignment.

3. Do not change English correction behavior.

4. Do not allow correction to summarize, rewrite, or materially change speaker meaning.

5. Do not require correction for normal CJK subtitle generation.

### 5. Acceptance Criteria

1. Glossary-only correction can improve known terms without invoking an LLM.

2. Local LLM correction can be enabled separately and remains conservative.

3. Risky corrected text is rejected, downgraded, or clearly marked.

4. Corrected sentence units remain alignable to existing timing anchors.

5. Normal CJK output still works when correction is disabled.

---

## Pre-Plan 6: Strengthen Fuzzy Alignment from Corrected CJK Text to Timing Anchors

### 1. Goal

Improve alignment from corrected CJK sentence text to Whisper timing anchors so corrected subtitles can retain usable audio timing even when transcript text differs from the timing transcript.

### 2. Why This Is Needed

The pipeline already performs basic split-backend alignment, but correction and postprocessing increase the gap between display text and timing text. Stronger fuzzy alignment is needed before correction can become a default or heavily used path.

### 3. Requirements

1. Fuzzy text alignment — support approximate alignment between corrected CJK sentences and Whisper timed transcript text.

2. Edit-tolerant matching — handle corrected characters, punctuation changes, proper noun fixes, deletions, insertions, and minor text differences without requiring exact token equality.

3. Alignment confidence — produce confidence or diagnostic metadata for each aligned sentence or cue.

4. Low-confidence fallback — fall back to Whisper timed transcript, coarse segment timing, or hard-cut output when alignment confidence is too low.

5. Timing safety — never emit invalid subtitle intervals, even when alignment is partial or uncertain.

6. Future extensibility — keep the alignment design open for later audio-level forced alignment without requiring it in this plan.

### 4. Non-Goals

1. Do not implement audio-level forced alignment in this plan.

2. Do not add new transcript backends.

3. Do not change English alignment behavior.

4. Do not require LLM correction to be enabled.

### 5. Acceptance Criteria

1. Corrected CJK text can be aligned to timing anchors even with minor text differences.

2. Low-confidence alignment is visible and does not silently produce misleading timestamps.

3. Fallback behavior is deterministic and recorded.

4. Final subtitle intervals remain valid and monotonic.

5. Existing SenseVoice-plus-Whisper baseline behavior remains available.

---

## Pre-Plan 7: Expose CJK Modes in CLI, UI, and Configuration

### 1. Goal

Expose the new CJK flow through clear user-facing modes so users can choose speed, stability, or accuracy without understanding backend internals.

### 2. Why This Is Needed

As CJK processing gains separate transcript backends, timing providers, correction modes, postprocess behavior, and fallback policies, users need a small set of understandable modes rather than low-level backend controls.

### 3. Requirements

1. CJK mode selection — support named CJK modes such as fast, stable, accuracy, and benchmark.

2. Backend configurability — expose transcript backend, timing backend, correction mode, and Whisper model selection where appropriate.

3. Sensible defaults — default CJK to SenseVoice transcript plus Whisper timing when available, with Whisper-only fallback.

4. Benchmark mode — allow users to bypass correction and advanced postprocessing to inspect raw backend behavior.

5. UI/CLI consistency — keep CLI flags, GUI controls, and config defaults aligned so the same pipeline behavior is available from both interfaces.

6. User-visible diagnostics — show when fallback occurred, when SenseVoice lacked timing, when alignment confidence was low, and when postprocess changed cue boundaries.

### 4. Non-Goals

1. Do not expose every internal tuning parameter.

2. Do not remove lower-level configuration for advanced users.

3. Do not change default English behavior.

4. Do not make experimental backends default.

### 5. Acceptance Criteria

1. Users can select CJK behavior through named modes.

2. Defaults produce the recommended SenseVoice-plus-Whisper path.

3. Benchmark mode remains available for raw backend inspection.

4. CLI, UI, and config expose consistent behavior.

5. Fallbacks and low-confidence output are visible to users.

---

## Suggested Implementation Order

1. Separate CJK Pipeline — completed.
2. Normalize CJK Transcript/Timing Contract — completed as part of the pipeline separation and follow-up refactor.
3. SenseVoice Transcript with Whisper Timing — completed.
4. CJK Subtitle Postprocessing and Readable Output — next.
5. CJK Subtitle Diagnostics and Benchmark Artifacts.
6. Glossary and Conservative Local LLM Correction.
7. Strengthen Fuzzy Alignment.
8. CLI/UI/Config Modes.
