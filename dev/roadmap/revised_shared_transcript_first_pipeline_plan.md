# Plan: Unified Subtitle Pipeline

## Current State (after PR1–PR3)

PR1–PR3 moved both English and CJK onto a single `StagedPipelineRunner`. The runner calls language-specific behavior through a `Policy` protocol. Both paths now share the same stage order:

```
build_inputs → correct → split_sentences → align → postprocess → final_cues
```

This is done and working. The strategies (`EnglishPipelineStrategy`, `CjkPipelineStrategy`) are thin wrappers.

### What still lives outside the staged pipeline

Several processing steps currently live in `processor.py`, before or after the staged runner:

| Step                        | Location                                | Notes                                                                                  |
| --------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------- |
| Punctuation restoration     | `processor.py`, before `strategy.run()` | Mutates `word_segments` in place; CJK skips it                                         |
| Translation                 | `processor.py`, after `strategy.run()`  | Runs on postprocessed writer chunks — already cut into short fragments with no context |
| SenseVoice transcript fetch | `processor.py`, before `strategy.run()` | Result passed via `StrategyContext` fields                                             |
| Chinese benchmark mode      | Removed in PR4b                         | `CjkPolicy.short_circuit()` and CLI/config benchmark hooks are gone                    |

The staged pipeline has no visibility into whether punctuation restoration ran or whether translation happened. These steps produce no stage artifacts and no metadata.

### Why current translation quality is poor

Translation currently runs **after postprocess**, which means it receives short, context-free subtitle fragments:

```
Input to translator (now):
  1. "Footage has emerged showing towering masts"
  2. "of the ship clipping the bridge"
  3. "as the sailing vessel was passing"
  4. "under the famous structure."

These are pieces of one sentence, but the translator sees them as independent lines.
```

Results: bad translations (no context), mismatched segment counts (LLM merges or splits freely), and a complex realignment step (`translation/aligner.py`) that often fails.

---

## PR4b — Metadata normalization and cleanup (done)

`final_cues.json` metadata has been normalized so English, CJK, and fallback paths emit the same required key set. PR4b also removed the Chinese benchmark shortcut and stale benchmark plumbing.

After PR4b, any consumer of `final_cues.json` can read required metadata without knowing which language produced it.

Implemented by commit `6f868bc88de2e00991016260b8d30f6c0460202b` / PR #21.

---

## Future direction: pull remaining steps into the staged pipeline

The goal is one linear flow where every step is a stage with artifacts and metadata:

```
transcription
  → punctuation restoration (if applicable)
  → correction (if applicable)
  → sentence splitting
  → translation (if applicable) ← after split, before align
  → alignment (current: English word-level, CJK char-level; future: force-aligned)
  → postprocess (source and translated, independently)
  → output
```

All steps run inside the staged pipeline. `processor.py` only handles I/O (download, audio preprocessing, model loading) and hands off to the runner.

### PR5 — Pull punctuation restoration into the staged pipeline

Move punctuation restoration from `processor.py` into the pipeline as a stage between `build_inputs` and `correct`.

Currently `processor.py` calls `restore_punctuation(word_segments, profile)` and mutates the word segments before the strategy ever sees them. This means:

- The staged pipeline doesn't know if punctuation was restored or not.
- There's no artifact recording the before/after.
- CJK skips it via `profile.skip_punctuation_model`, but this check lives in `processor.py`, not in the policy.

After PR5:

- Punctuation restoration becomes a policy-controlled stage.
- English policy enables it; CJK policy skips it.
- The runner records whether it ran/skipped in stage metadata.
- `processor.py` no longer touches `word_segments` after transcription.
- The stage cache hash includes the punctuation backend/config so changing punctuation settings invalidates downstream artifacts safely.

### PR6 — Pull translation into the staged pipeline (sentence-level)

Move translation from `processor.py` into the pipeline as a stage **after `split_sentences` and before `align`**.

#### Why after split, not after postprocess

The current flow translates postprocessed writer chunks — short fragments already cut by word count or display width. The translator has no sentence context and the output count frequently mismatches, requiring a fragile realignment step.

Moving translation to after `split_sentences` means the translator receives **complete sentences with full context**. The prompt enforces strict 1:1 sentence correspondence, so the output count is guaranteed to match.

#### Translation approach

Send all sentences in a single prompt with numbering. The LLM sees full context but outputs one translated line per source sentence:

```
以下是一段英文逐句轉錄，請逐句翻譯成繁體中文。
嚴格保持句數一致，第 N 句原文對應第 N 句譯文。
不要合併、不要拆分、不要省略。

1. Footage has emerged showing towering masts of the ship clipping the bridge as the sailing vessel was passing under the famous structure.
2. Parts of the masts reportedly fell onto the deck.

→
1. 畫面顯示，這艘帆船經過時，高聳的桅杆撞上了這座著名建築。
2. 據報導，部分桅杆掉落在甲板上。
```

If the returned count doesn't match, retry (not realign). This eliminates the `translation/aligner.py` complexity for the common case.

#### Bilingual cue pairing

After translation, source and translated sentences share 1:1 correspondence. Both go through alignment and postprocess independently:

- Source sentences → align with word timestamps → postprocess by source language rules → source cues
- Translated sentences → inherit the same sentence time range → postprocess by target language rules → translated cues

The source postprocess produces the primary cue track (it has real word-level timestamps). Translated text is then mapped onto source cues:

1. For each source sentence's time range, count how many source cues were produced.
2. Calculate proportional split points in the translated text based on source cue character ratios.
3. At each split point, look for nearby target-language punctuation (`，。、！？`).
4. If punctuation is found within range, snap to it. If not, cut at the proportional point.
5. Each translated fragment pairs with the corresponding source cue.

Example:

```
Source sentence: 5.36s — 9.92s
Source postprocess produced 4 cues:
  cue 1: "Footage has emerged showing towering masts"       5.36 - 7.52  (~33%)
  cue 2: "of the ship clipping the bridge"                  7.52 - 8.64  (~25%)
  cue 3: "as the sailing vessel was passing"                 8.64 - 9.20  (~25%)
  cue 4: "under the famous structure."                       9.20 - 9.92  (~17%)

Translated sentence: "畫面顯示，這艘帆船經過時，高聳的桅杆撞上了這座著名建築。"

Proportional split points in Chinese text: ~33%, ~58%, ~83%
  ~33% (pos ~7)  → 逗號在 pos 11 → snap → "畫面顯示，這艘帆船經過時，"
  ~58% (pos ~13) → no punctuation → cut at proportion → "高聳的桅杆"
  ~83% (pos ~18) → no punctuation → cut at proportion → "撞上了這座"
  remainder                                             → "著名建築。"

Final bilingual output:
  cue 1: "Footage has emerged showing towering masts"       5.36 - 7.52
         "畫面顯示，這艘帆船經過時，"

  cue 2: "of the ship clipping the bridge"                  7.52 - 8.64
         "高聳的桅杆"

  cue 3: "as the sailing vessel was passing"                8.64 - 9.20
         "撞上了這座"

  cue 4: "under the famous structure."                      9.20 - 9.92
         "著名建築。"
```

#### What this replaces

- `translation/aligner.py` (DP-based realignment) — no longer needed; 1:1 sentence mapping eliminates count mismatches.
- Ad-hoc `translation_cache/` in `processor.py` — cache moves to canonical stage directory.
- `_SENTENCE_DELIM` / `_BLOCK_DELIM` parsing in `gemini_translator.py` — replaced by numbered-line prompt/parse for the staged translation path.

### PR7 — Force alignment stage

Replace the current char-level / word-level alignment with a force alignment stage that can use audio-level alignment.

Currently:

- English: walks spaCy tokens onto ASR word timestamps (exact word matching).
- CJK: uses `difflib.SequenceMatcher` to fuzzy-match corrected text to timing-side text (char-level).

Both approaches break when the corrected/restored text diverges significantly from the ASR transcript. Force alignment against the audio would be more robust.

After PR7:

- Alignment stage can use audio-level forced alignment as a backend (e.g. `torchaudio` forced alignment API with wav2vec2 models).
- The current word-level and char-level aligners remain as fallbacks.
- Policy selects which aligner to use.

### PR8 — Move SenseVoice transcript fetch into the pipeline

Currently `processor.py` runs SenseVoice, caches the result to `sensevoice_text.txt`, and passes the text via `StrategyContext` fields. This is a side channel that the runner doesn't control.

After PR8:

- The runner owns the SenseVoice transcript fetch as part of `build_inputs`.
- The SenseVoice cache lives under the canonical stage directory.
- `StrategyContext` no longer carries transcript override fields.
- Fallback logic (SenseVoice unavailable / too short / empty) is handled inside the policy's `build_inputs`, with proper metadata.

---

## Target architecture

```
processor.py (I/O only):
  download audio
  preprocess audio (demucs, ffmpeg)
  load ASR model
  transcribe → word_segments
  pass to runner

staged pipeline runner:
  1. build_inputs        → Transcript + TimingAnchors
  2. punctuation_restore → Transcript (restored)           [policy-controlled]
  3. correct             → Transcript (corrected)           [policy-controlled]
  4. split_sentences     → list[Sentence]
  5. translate           → list[TranslatedSentence]         [policy-controlled, 1:1]
  6. align               → list[AlignedCue]                 [word/char/force]
  7. postprocess         → source cues + bilingual pairing
  8. final_cues          → final_cues.json with full metadata

translation pairing (inside step 7):
  - source sentences → align → postprocess → source cues (primary track)
  - per source sentence time range:
      count source cues in range
      proportional split translated text
      snap to target-language punctuation where possible
      pair each translated fragment with corresponding source cue

each stage:
  - reads explicit inputs from previous stage
  - writes canonical artifact
  - records metadata (ran/skipped, backend, config)
  - policy decides behavior (enable/disable, backend selection, params)
```

`processor.py` does not modify `word_segments`, does not run translation, does not fetch SenseVoice transcripts. It only does I/O and model loading, then delegates everything to the runner.

---

## Removed

- **Chinese benchmark mode** — removed in PR4b. It was a debug shortcut that bypassed all stages. If raw ASR output comparison is needed, it can be done by reading `raw_transcript.json` directly.
- **`translation/aligner.py` DP realignment** — replaced in PR6 by 1:1 sentence-level translation. No more count mismatches to fix after the fact.

---

## Implementation order

| PR   | Scope                                                         | Status |
| ---- | ------------------------------------------------------------- | ------ |
| PR1  | Shared models + compatibility re-exports                      | Done   |
| PR2  | CJK onto shared runner                                        | Done   |
| PR3  | English onto shared runner                                    | Done   |
| PR4b | Metadata normalization + remove benchmark + stale cleanup     | Done   |
| PR5  | Punctuation restoration into pipeline                         | Future |
| PR6  | Translation into pipeline (sentence-level, bilingual pairing) | Future |
| PR7  | Force alignment stage (wav2vec2 / torchaudio)                 | Future |
| PR8  | SenseVoice fetch into pipeline                                | Future |

PR5–PR8 are mostly independent and can be reordered based on priority. PR6 should land before removing the current translation realignment code from the runtime path. PR7 (force alignment) depends on external tooling (`torchaudio.pipelines` forced alignment API with wav2vec2).
