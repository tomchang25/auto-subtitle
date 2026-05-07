# Subforge Product Roadmap

This document separates the roadmap into three related but distinct areas:

1. **subforge** — archive-grade subtitle production for long-form VODs.
2. **subforge-live** — realtime livestream comprehension for personal viewing.
3. **subforge-core** — shared concepts and overlap boundaries, without requiring an immediate shared package.

The key principle is that `subforge-live` should not become a required ingest path for `subforge`. Formal archive subtitles should be regenerated from the original VOD or media file through the full `subforge` pipeline.

---

## 1. subforge

### Product Positioning

`subforge` is a long-form subtitle production pipeline for corrected and translated transcripts.

It is not intended to be another short-form auto-caption generator. Its value is in producing high-quality, inspectable, reproducible subtitles for long videos such as livestream archives, lectures, technical videos, podcasts, meetings, and CJK-heavy content.

### Strategic Direction

```text
Long-form VOD
→ clean audio preprocessing
→ transcript generation
→ transcript/timing separation
→ correction
→ alignment
→ readable subtitle postprocessing
→ translation timing projection
→ final SRT/VTT output
```

### Differentiation

#### 1. Long-form first

`subforge` should optimize for 2-hour-plus livestream archives and long videos, not short social clips.

Focus areas:

- resumable processing
- cached intermediate artifacts
- stable long-video segmentation
- reproducible outputs
- diagnostics for failed or low-quality sections

This is different from creator tools that optimize for fast, styled, burn-in captions.

#### 2. Transcript-first source of truth

Raw ASR output should not be treated as final subtitle text.

The pipeline should distinguish:

```text
raw transcript
corrected transcript
translated transcript
timing anchors
aligned cues
final subtitle cues
```

This allows corrected text or translated text to become the source of truth while timing remains a separate layer.

#### 3. CJK-first quality path

Chinese, Japanese, and Korean should be treated as first-class subtitle languages, not just supported languages.

The CJK path should preserve:

- CJK punctuation behavior
- display-width-aware splitting
- no-whitespace assumptions
- proper handling of mixed CJK and English text
- SenseVoice transcript quality
- Whisper timing stability

#### 4. Proper noun and glossary workflow

Long-form technical and livestream content often contains names, game titles, product names, VTuber names, memes, jargon, and recurring domain-specific terms.

`subforge` should support deterministic glossary correction before heavier LLM correction.

This should reduce repeated ASR and translation mistakes without requiring the model to rewrite the transcript.

#### 5. Translation-aware subtitle generation

Subtitle translation should not be forced into low-quality line-by-line literal translation.

The desired flow is:

```text
source transcript with timing
→ semantic source blocks
→ coherent block translation
→ target-language cue segmentation
→ timing projection back onto translated cues
```

This avoids two common failure modes:

```text
word-by-word translation
→ unnatural or incorrect translation

block translation
→ good text but no usable subtitle timing
```

#### 6. Artifact-first debuggability

Every major stage should produce inspectable artifacts.

A user should be able to determine whether a bad subtitle came from:

- ASR transcript quality
- timing source quality
- alignment failure
- correction risk
- postprocessing decisions
- translation quality
- fallback behavior

This makes `subforge` closer to a subtitle build system than a one-shot caption generator.

#### 7. Reproducible local/batch workflow

`subforge` should prioritize predictable batch processing over realtime latency.

It should be safe to rerun a long VOD with different settings, compare artifacts, preserve intermediate outputs, and produce stable final subtitles.

---

### Engineering Roadmap

#### 1. CJK pipeline separation

Status: completed.

Purpose:

- Separate the CJK path from the English-oriented word-timestamp subtitle flow.
- Keep English behavior unchanged.
- Make CJK processing transcript-first.

#### 2. CJK transcript/timing contract normalization

Status: completed.

Purpose:

- Treat transcript text and timing anchors as separate concepts.
- Avoid fake word timestamps for transcript-only backends.
- Preserve provenance and fallback metadata.

#### 3. SenseVoice transcript plus Whisper timing

Status: completed.

Purpose:

- Use SenseVoice as the CJK transcript source.
- Use Whisper as the timing provider.
- Fall back to Whisper-only when SenseVoice is unavailable or suspicious.
- Persist transcript/timing backend metadata.

#### 4. CJK subtitle postprocessing and readable output

Status: next.

Purpose:

Replace hard-cut output with readable CJK subtitle formatting.

Includes:

- display-width-aware splitting
- punctuation-aware cue boundaries
- short-cue merging
- long-cue splitting
- timing safety
- postprocess diagnostics

#### 5. CJK diagnostics and benchmark artifacts

Purpose:

Make backend choice, timing source, fallback behavior, alignment quality, and postprocess decisions inspectable.

Includes:

- raw transcript artifact
- timing anchor artifact
- alignment artifact
- postprocess artifact
- final cue artifact
- benchmark metadata

#### 6. Glossary correction

Purpose:

Provide deterministic correction for recurring names, terms, game titles, product names, technical vocabulary, and domain-specific phrases.

This should happen before LLM correction.

#### 7. Conservative local LLM correction

Purpose:

Fix CJK ASR errors, punctuation, sentence boundaries, and homophones without rewriting the speaker’s meaning.

Guardrails should prevent:

- summarization
- tone rewriting
- large insertions
- large deletions
- sentence reordering
- hallucinated content

#### 8. Stronger fuzzy alignment

Purpose:

Improve alignment between corrected CJK text and Whisper timing anchors when the corrected display text differs from the timing transcript.

Includes:

- edit-tolerant matching
- alignment confidence
- fallback behavior
- timing safety

#### 9. Translation timing projection

Purpose:

Support translated subtitles that are natural and readable without losing source timing.

The translation pipeline should translate coherent blocks and then project timing back onto target-language subtitle cues.

#### 10. Partial rerun and dirty-region workflow

Purpose:

For long videos, small transcript or glossary edits should not require rerunning the full pipeline.

The pipeline should eventually support marking affected regions as dirty and rerunning only the necessary stages.

#### 11. CLI, UI, and configuration modes

Purpose:

Expose clear CJK modes without forcing users to understand every backend detail.

Possible modes:

- fast
- stable
- accuracy
- benchmark

---

## 2. subforge-live

### Product Positioning

`subforge-live` is a realtime Japanese livestream comprehension tool.

It is not an archive subtitle production tool and should not be required input for `subforge`.

Its goal is simple:

```text
Help me understand a Japanese livestream while it is happening.
```

Formal VOD subtitles should still be produced by rerunning the archived video through `subforge`.

### Strategic Direction

```text
YouTube livestream URL
→ yt-dlp live audio input
→ ffmpeg PCM decode
→ VAD / live chunking
→ Japanese ASR
→ rolling-context translation
→ overlay display
→ optional local history
```

### Differentiation

#### 1. YouTube livestream first

For the target use case, direct livestream audio input is better than system audio capture.

Primary input should be:

```text
YouTube live URL
→ yt-dlp
→ ffmpeg
→ PCM stream
```

System audio capture should be a fallback, not the default.

#### 2. Low-latency comprehension

The live tool should optimize for “can follow the stream now,” not for final subtitle quality.

Latency matters more than perfect formatting.

#### 3. Rolling-context translation

Japanese live translation benefits from prior context because:

- subjects are often omitted
- sentence-final meaning matters
- negation may appear late
- streamer-specific terms repeat
- game and VTuber names need consistency

Translation should use recent transcript context instead of translating isolated 1–2 second chunks.

#### 4. Rough and stable dual captions

The live UI should support two layers:

```text
rough caption
→ fast, low latency, may be imperfect

stable caption
→ slightly delayed, translated with more context
```

This gives a better tradeoff between latency and translation quality.

#### 5. Streamer and domain glossary

The live tool should support a lightweight glossary or name table for:

- VTuber names
- game names
- character names
- locations
- memes
- recurring terms
- technical terms

This is more important than formal transcript correction for the live use case.

#### 6. Overlay-first UX

The primary UI should be an overlay or side panel while watching.

It should not become a full subtitle editor.

#### 7. Optional local history

Local history can be useful for:

- searching what was said
- reviewing a translation
- debugging ASR mistakes
- collecting glossary candidates

However, this history should not become a required input for `subforge`.

---

### Engineering Roadmap

#### 1. yt-dlp live audio input

Purpose:

Take a YouTube livestream URL directly and resolve the live audio stream.

This should be the primary input path.

#### 2. ffmpeg PCM decode

Purpose:

Decode the live audio stream into ASR-ready PCM chunks.

#### 3. VAD and live chunking

Purpose:

Split the live audio stream into speech chunks with acceptable latency and reasonable phrase boundaries.

Tuning goals:

- avoid cutting too early
- avoid overly long chunks
- preserve enough context for Japanese sentence structure
- keep latency acceptable for live viewing

#### 4. Japanese ASR

Purpose:

Produce usable Japanese transcript segments from live audio.

Candidate backends:

- SenseVoice
- faster-whisper
- whisper.cpp later if useful

The first version should use one backend cleanly before adding backend switching.

#### 5. Overlay caption display

Purpose:

Show Japanese transcript and/or translation while watching.

The overlay should be lightweight and focused on readability.

#### 6. Rolling-context translation

Purpose:

Translate with recent transcript context instead of isolated chunks.

The translator should be aware of:

- previous source text
- previous translations
- glossary entries
- current partial or stable segment

#### 7. Rough and stable dual subtitle mode

Purpose:

Show a fast rough translation first, then optionally update it with a more stable block translation.

#### 8. Glossary and name table

Purpose:

Keep names and recurring terms consistent during live translation.

The glossary can start as a simple local mapping.

#### 9. Optional session history

Purpose:

Preserve a local, searchable history for personal review and debugging.

This should not be treated as formal subtitle source material.

#### 10. Optional system audio capture fallback

Purpose:

Support non-YouTube sources or cases where yt-dlp cannot access the stream.

Possible use cases:

- unsupported platforms
- browser-only streams
- Discord
- Zoom
- local apps
- DRM or access-restricted content where direct stream input is not practical

---

## 3. subforge-core

### Purpose

`subforge-core` is not an immediate extraction target.

For now, it is a conceptual boundary describing the parts of `subforge` and `subforge-live` that may overlap.

A real shared package should only be created after both projects have stable duplicated concepts.

### Design Principle

Do not over-abstract too early.

The two projects have different runtime shapes:

```text
subforge
→ batch, offline, artifact-first, archive quality

subforge-live
→ streaming, low latency, overlay-first, live comprehension
```

Shared code should be extracted only when the shared boundary is stable.

---

### Potential Shared Concepts

#### 1. Language profiles

Possible shared responsibilities:

- language codes
- CJK vs non-CJK behavior
- punctuation sets
- display width policy
- sentence boundary hints

This is likely to become shared eventually.

#### 2. Glossary model

Possible shared responsibilities:

- glossary entry schema
- source term
- target term
- aliases
- language scope
- case sensitivity
- domain tag
- notes

`subforge-live` may use glossaries for live consistency.
`subforge` may use glossaries for deterministic correction and translation quality.

#### 3. Transcript segment schema

Possible shared responsibilities:

```text
start time
end time
source text
language
backend
confidence
is partial
is final
```

This may be useful across both live and batch workflows, but should not be extracted until both projects prove they need the same shape.

#### 4. Translation segment schema

Possible shared responsibilities:

```text
source text
translated text
source language
target language
translation backend
context window
glossary version
stability status
```

This may become important if both projects support rolling-context or block-based translation.

#### 5. Text normalization

Possible shared responsibilities:

- CJK punctuation normalization
- whitespace cleanup
- repeated punctuation cleanup
- full-width and half-width handling
- optional Simplified/Traditional normalization
- safe formatting cleanup

#### 6. Prompt templates

Possible shared responsibilities:

- conservative correction prompts
- glossary-aware translation prompts
- block translation prompts
- no-rewrite guardrails
- CJK-specific translation instructions

Live prompts and archive prompts should remain separate at first, even if they share smaller reusable fragments.

#### 7. Subtitle display-width utilities

Possible shared responsibilities:

- CJK display width calculation
- mixed English/CJK width handling
- line length measurement
- punctuation-aware break hints

This is more critical to `subforge`, but may later help `subforge-live` overlay wrapping.

#### 8. Artifact metadata vocabulary

Possible shared responsibilities:

- backend name
- model name
- timing source
- transcript source
- fallback reason
- confidence
- correction mode
- glossary version

This should remain conceptual until artifact schemas stabilize.

---

### What Should Not Be Shared Yet

#### 1. Audio capture

`subforge-live` needs streaming input and possibly system audio capture.

`subforge` needs offline media download, audio extraction, and preprocessing.

These should stay separate.

#### 2. Overlay UI

Overlay behavior belongs only to `subforge-live`.

`subforge` should not depend on overlay or live UI code.

#### 3. Streaming queues

Realtime queue management, partial segment updates, backpressure, and latency policies are live-only concerns.

#### 4. Demucs and VOD preprocessing

Archive-quality preprocessing belongs to `subforge`.

It should not be required for live captions.

#### 5. Full subtitle postprocessing

Final subtitle formatting belongs to `subforge`.

`subforge-live` may need simple overlay wrapping, but it should not own archive-grade subtitle layout.

#### 6. Formal alignment pipeline

Full alignment, correction, and translation timing projection belong to `subforge`.

`subforge-live` should not attempt to produce formal archive subtitles.

---

### When to Actually Extract subforge-core

Only extract a real shared package when at least three of the following are true:

1. Both projects maintain duplicated glossary schemas.
2. Both projects maintain duplicated language profiles.
3. Both projects maintain duplicated CJK text normalization.
4. Both projects maintain duplicated translation prompt fragments.
5. Both projects maintain duplicated transcript or translation segment schemas.
6. Both projects need compatible artifact metadata.
7. Both projects benefit from the same subtitle display-width utility.

Until then, keep the projects separate and allow small duplication.

---

## Final Strategic Split

```text
subforge
→ Make high-quality subtitles for long-form archives.

subforge-live
→ Help me understand Japanese livestreams while they are happening.

subforge-core
→ A future shared boundary for stable overlap, not an immediate abstraction.
```

## Required Integration

There is no required integration between `subforge` and `subforge-live`.

The formal archive subtitle path remains:

```text
YouTube VOD or local media
→ subforge full pipeline
→ final subtitles
```

The live viewing path remains:

```text
YouTube livestream
→ subforge-live
→ realtime comprehension
```

Optional future bridge only:

```text
subforge-live may export:
  - glossary candidates
  - bookmarks
  - manually marked terms
  - rough session notes

subforge may optionally consume:
  - glossary hints
  - manual notes
```

`subforge` should not depend on `subforge-live` transcript output for formal subtitle generation.
