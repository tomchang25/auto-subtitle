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

---

## Pre-Plan 2: Add SenseVoice as the Primary CJK Transcript Backend

### 1. Goal

Use SenseVoice as the primary CJK transcript source because it is fast and produces usable text, while treating it as transcript-only unless reliable timestamps are available.

### 2. Requirements

1. SenseVoice backend — support SenseVoice as a selectable ASR backend for CJK transcript generation.

2. Transcript-only contract — allow ASR backends to return transcript-level or segment-level output without requiring fake word-level timestamps.

3. Timing status metadata — record whether a backend provides word timing, segment timing, no timing, or untrusted timing.

4. Default CJK routing — make SenseVoice the preferred CJK transcript backend when available, with Whisper as fallback.

5. No fake timestamps — do not synthesize word-level timestamps for SenseVoice output just to satisfy the existing word segment format.

6. Backend diagnostics — include backend name, model name, timing availability, transcript length, and fallback usage in logs or metadata.

---

## Pre-Plan 3: Use Whisper as CJK Timing Provider and Fallback Transcript Backend

### 1. Goal

Use Whisper as the reliable CJK timing anchor and fallback transcript source when SenseVoice output is missing, weak, or difficult to align.

### 2. Requirements

1. Timing backend role — support Whisper as a separate timing provider rather than only as a full ASR backend.

2. Fallback transcript role — use Whisper transcript output when SenseVoice fails, produces empty output, or produces suspiciously incomplete text.

3. Model configurability — expose Whisper large-v3-turbo and full large-v3 as distinct CJK timing/transcript candidates.

4. Timing validation — detect invalid, missing, non-monotonic, or suspiciously sparse timing before using Whisper output as alignment anchors.

5. CJK decoding options — support CJK-specific Whisper settings that prioritize stable Chinese output and usable timing.

6. Paraformer demotion — prevent Paraformer from being automatically selected for CJK and keep it only as an explicit experimental or legacy option.

---

## Pre-Plan 4: Add Local LLM Correction and Sentence Segmentation for CJK

### 1. Goal

Add a conservative local LLM correction stage that fixes CJK ASR errors and produces sentence-level units before alignment.

### 2. Requirements

1. Conservative correction — support local LLM correction for ASR misrecognitions, homophones, punctuation, proper nouns, and sentence boundaries without rewriting the speaker’s meaning.

2. Sentence list output — require the correction stage to produce a stable ordered sentence list rather than direct SRT output.

3. Dual transcript input — allow the corrector to compare SenseVoice transcript with Whisper transcript when both are available.

4. Guardrails — detect excessive insertions, deletions, or rewriting so risky corrected text can be rejected or downgraded.

5. Configurability — make correction selectable as off, glossary-only, local LLM, or future external LLM mode.

6. Deterministic fallback — if correction fails or is disabled, continue with raw transcript or glossary-only correction.

---

## Pre-Plan 5: Add Fuzzy Alignment from Corrected CJK Text to Whisper Timing Anchors

### 1. Goal

Align corrected CJK sentence text to Whisper timing anchors so CJK subtitles can use improved text while retaining usable audio timing.

### 2. Requirements

1. Fuzzy text alignment — support approximate alignment between corrected CJK sentences and Whisper timed transcript segments.

2. Edit-tolerant matching — handle corrected characters, punctuation changes, proper noun fixes, and minor text differences without requiring exact token equality.

3. Alignment confidence — produce confidence or diagnostic metadata for each aligned sentence.

4. Low-confidence fallback — fall back to Whisper timed transcript, coarse segment timing, or hard-cut output when alignment confidence is too low.

5. Timing safety — never emit invalid subtitle intervals, even when alignment is partial or uncertain.

6. Future extensibility — keep the alignment design open for later audio-level forced alignment without requiring it in the first implementation.

---

## Pre-Plan 6: Add CJK Subtitle Output, Diagnostics, and Benchmark Artifacts

### 1. Goal

Produce CJK subtitles from corrected and aligned sentences while preserving enough diagnostics to compare ASR, correction, and alignment quality.

### 2. Requirements

1. CJK subtitle formatting — output SRT from aligned CJK sentence units using display-width-aware hard splitting for long lines.

2. Raw artifact export — save raw transcript, corrected transcript, sentence list, timing anchors, alignment results, and final SRT for debugging.

3. Benchmark report — generate metadata that compares backend selection, transcript length, timing status, correction risk, alignment confidence, and fallback behavior.

4. Invalid timing handling — keep ASR text available even when timing is invalid, but prevent invalid cues from being written as normal subtitles.

5. Mode clarity — distinguish benchmark/raw output, corrected output, and final subtitle output so model evaluation is not confused with polishing quality.

6. Regression safety — preserve existing output behavior for English and non-CJK flows.

---

## Pre-Plan 7: Expose CJK Modes in CLI, UI, and Configuration

### 1. Goal

Expose the new CJK flow through clear user-facing modes so users can choose speed, stability, or accuracy without understanding backend internals.

### 2. Requirements

1. CJK mode selection — support named CJK modes such as fast, stable, accuracy, and benchmark.

2. Backend configurability — expose transcript backend, timing backend, correction mode, and Whisper model selection where appropriate.

3. Sensible defaults — default CJK to SenseVoice transcript plus Whisper timing when available, with Whisper-only fallback.

4. Benchmark mode — allow users to bypass correction and advanced segmentation to inspect raw backend behavior.

5. UI/CLI consistency — keep CLI flags, GUI controls, and config defaults aligned so the same pipeline behavior is available from both interfaces.

6. User-visible diagnostics — show when fallback occurred, when SenseVoice lacked timing, and when alignment confidence was low.

---

## Suggested Implementation Order

1. Separate CJK Pipeline
2. SenseVoice Transcript Backend
3. Whisper Timing/Fallback Backend
4. CJK Artifact + Benchmark Output
5. Local LLM Correction
6. Fuzzy Alignment
7. CLI/UI/Config Modes
