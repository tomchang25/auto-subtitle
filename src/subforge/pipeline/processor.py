import hashlib
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class PipelineCancelled(Exception):
    """Raised when the user cancels the pipeline between steps."""


from subforge.downloader.youtube import (
    get_video_title,
    download_audio,
    download_video,
)

from subforge.audio.preprocess import preprocess_audio

from subforge.subtitle.writer import write_srt

from subforge.config import (
    MODEL_TIER,
    WHISPER_TIER_MAP,
    SENSEVOICE_TIER_MAP,
    OUTPUT_DIR,
    USE_LLM_PUNCTUATION,
    TRANSLATE_TGT_LANG,
    TRANSLATE_SRC_LANG,
    TARGET_LANG_SHORT,
    ASR_BACKEND,
    ASR_SOURCE_LANGUAGE,
    CJK_USE_SENSEVOICE_TRANSCRIPT,
    CJK_SENSEVOICE_MIN_RATIO,
)
from subforge.nlp.lang_profile import get_profile
from subforge.pipeline.strategies import StrategyContext, get_strategy
from subforge.utils import get_bounds_and_text, save_word_segments

ProgressCallback = Callable[[str, str], None]


def _resolve_transcriber(backend: str, source_language: str):
    from subforge.transcription.factory import get_transcriber
    return get_transcriber(backend, source_language)  # (fn, actual_backend)


class SubtitlePipeline:
    def __init__(
        self,
        url: str = "",
        model_name: str | None = None,
        output_dir: Path = OUTPUT_DIR,
        use_demucs: bool = True,
        use_punctuation: bool = USE_LLM_PUNCTUATION,
        download_mp4: bool = False,
        video_quality: str = "1080p",
        translate_method: str | None = None,
        target_lang: str = TRANSLATE_TGT_LANG,
        progress_callback: ProgressCallback | None = None,
        force: bool = False,
        local_file: str | None = None,
        asr_backend: str = ASR_BACKEND,
        source_language: str = ASR_SOURCE_LANGUAGE,
        missing_backend_handler: Callable[[str, str, str], bool] | None = None,
    ):
        self.url = url
        self.local_file = Path(local_file) if local_file else None
        self.model_name = model_name or MODEL_TIER
        self.output_dir = output_dir
        self.use_demucs = use_demucs
        self.use_punctuation = use_punctuation
        self.download_mp4 = download_mp4
        self.video_quality = video_quality
        self.translate_method = translate_method
        self.target_lang = target_lang
        self.progress_callback = progress_callback
        self.force = force
        self.asr_backend = asr_backend
        self.source_language = source_language
        self.missing_backend_handler = missing_backend_handler
        self._cancelled = False
        self._fallback_from: str | None = None
        self.project_dir = None
        self.title = None
        self.audio_path = None
        self.video_path = None

    @property
    def fallback_from(self) -> str | None:
        """Backend that was requested but unavailable (``None`` = no fallback)."""
        return self._fallback_from

    _AUDIO_EXTENSIONS = {
        ".mp3",
        ".wav",
        ".flac",
        ".m4a",
        ".ogg",
        ".opus",
        ".wma",
        ".aac",
    }

    _VIDEO_EXTENSIONS = {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".flv",
        ".wmv",
        ".ts",
        ".m4v",
    }

    _MEDIA_EXTENSIONS = _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS

    def cancel(self):
        """Request cancellation. The pipeline will stop before the next step."""
        self._cancelled = True

    def _find_existing_audio(self, folder: Path) -> Path | None:
        """Return the first audio file found directly in *folder*, or None."""
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in self._AUDIO_EXTENSIONS:
                return f
        return None

    def _check_cancel(self):
        if self._cancelled:
            raise PipelineCancelled("Pipeline cancelled by user")

    def _emit(self, step: str, detail: str = ""):
        message = f"[{step}] {detail}" if detail else f"[{step}]"
        logger.info(message)
        if self.progress_callback is not None:
            try:
                self.progress_callback(step, detail)
            except Exception as exc:
                logger.error("progress_callback raised: %s", exc)

    def _extract_audio_from_video(self, video_path: Path, output_path: Path) -> Path:
        """Extract audio track from a video file using ffmpeg."""
        import subprocess

        from subforge.audio.preprocess import _find_ffmpeg
        ffmpeg = _find_ffmpeg()
        audio_out = output_path.with_suffix(".mp3")
        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-vn",             # no video
            "-acodec", "libmp3lame",
            "-q:a", "2",      # high quality mp3
            str(audio_out),
        ]
        logger.info("Extracting audio: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed:\n"
                f"{result.stderr.decode('utf-8', errors='replace')}"
            )
        return audio_out

    def _maybe_run_sensevoice_transcript(
        self,
        wav_path: Path,
        word_segments: list[dict],
    ) -> tuple[str | None, str | None, str | None]:
        """Return ``(text, transcript_source, fallback_reason)``.

        ``text is None`` means SenseVoice was skipped entirely; the strategy
        falls back to the Whisper-only path. ``fallback_reason`` is set when
        SenseVoice was attempted but its output was rejected, so callers can
        record why the fallback happened in CJK diagnostics.
        """
        from subforge.transcription.factory import is_backend_available

        if not is_backend_available("sensevoice"):
            self._emit(
                "Transcript",
                "SenseVoice unavailable — using Whisper-only transcript",
            )
            return None, None, "sensevoice_unavailable"

        cache_path = self.project_dir / "sensevoice_text.txt"
        if not self.force and cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
            self._emit("Transcript", f"SenseVoice transcript loaded ({len(text)} chars, cached)")
        else:
            self._emit("Transcript", "Running SenseVoice for transcript text…")
            try:
                from subforge.transcription.sensevoice_transcriber import (
                    transcribe_text_only,
                )

                model_name = SENSEVOICE_TIER_MAP.get(
                    self.model_name, SENSEVOICE_TIER_MAP["large"]
                )
                text, _detected = transcribe_text_only(
                    wav_path,
                    model_name,
                    progress_callback=self.progress_callback,
                )
            except Exception as exc:  # noqa: BLE001 — transcript backend boundary
                logger.warning("SenseVoice transcript failed: %s", exc)
                self._emit(
                    "Transcript",
                    f"SenseVoice failed ({type(exc).__name__}); using Whisper-only",
                )
                return None, None, f"sensevoice_error:{type(exc).__name__}"
            cache_path.write_text(text, encoding="utf-8")
            self._emit(
                "Transcript",
                f"SenseVoice transcript complete ({len(text)} chars, saved)",
            )

        if not text.strip():
            self._emit(
                "Transcript",
                "SenseVoice transcript empty — using Whisper-only",
            )
            return None, None, "sensevoice_empty"

        whisper_chars = sum(len(seg.get("word", "") or "") for seg in word_segments)
        if whisper_chars > 0:
            ratio = len(text) / whisper_chars
            if ratio < CJK_SENSEVOICE_MIN_RATIO:
                self._emit(
                    "Transcript",
                    "SenseVoice transcript suspiciously short "
                    f"({len(text)} vs Whisper {whisper_chars} chars, "
                    f"ratio={ratio:.2f}); using Whisper-only",
                )
                return None, None, "sensevoice_too_short"

        return text, "sensevoice", None

    def run(self) -> Path:
        self._emit("Pipeline", "=== Starting Subtitle Pipeline ===")

        if self.local_file:
            # --- Local file mode ---
            if not self.local_file.exists():
                raise FileNotFoundError(f"Local file not found: {self.local_file}")

            suffix = self.local_file.suffix.lower()
            if suffix not in self._MEDIA_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file format: {suffix}. "
                    f"Supported: {', '.join(sorted(self._MEDIA_EXTENSIONS))}"
                )

            self.title = self.local_file.stem
            self.project_dir = self.output_dir / self.title
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self._emit("Local", f"Using local file: {self.local_file.name}")
            self._emit("Local", f"Project directory: {self.project_dir}")
            self._check_cancel()

            if suffix in self._VIDEO_EXTENSIONS:
                # Extract audio from video (skip if cached mp3 exists)
                audio_dest = self.project_dir / self.local_file.stem
                cached_mp3 = audio_dest.with_suffix(".mp3")
                if not self.force and cached_mp3.exists():
                    self._emit("Extract", f"Skipped (found cached: {cached_mp3.name})")
                    self.audio_path = cached_mp3
                else:
                    self._emit("Extract", "Extracting audio from video file")
                    self.audio_path = self._extract_audio_from_video(
                        self.local_file, audio_dest
                    )
                self.video_path = self.local_file
            else:
                # Audio file — copy into project dir (skip if already there)
                dest = self.project_dir / self.local_file.name
                if not self.force and dest.exists():
                    self._emit("Local", f"Skipped copy (found cached: {dest.name})")
                else:
                    import shutil
                    shutil.copy2(self.local_file, dest)
                self.audio_path = dest
                self._emit("Local", f"Audio ready: {self.audio_path.name}")
            self._check_cancel()
        else:
            # --- YouTube URL mode (original behaviour) ---
            # Step 1: Get title and prepare directories
            self._emit("Title", "Fetching video title")
            self.title = get_video_title(self.url)
            self.project_dir = self.output_dir / self.title
            self.project_dir.mkdir(parents=True, exist_ok=True)
            self._emit("Title", f"Project directory: {self.project_dir}")
            self._check_cancel()

            # Step 2: Download audio (skip if audio file already exists in folder)
            output_base = self.project_dir / self.title
            existing_audio = self._find_existing_audio(self.project_dir)
            if existing_audio:
                self._emit("Download", f"Skipped (found existing: {existing_audio.name})")
                self.audio_path = existing_audio
            else:
                self._emit("Download", "Downloading audio")
                self.audio_path = download_audio(
                    self.url, output_base, format="mp3", force=False
                )
            self._check_cancel()

        # Step 3: Preprocess audio
        self._emit(
            "Preprocess",
            f"Preparing WAV (demucs={'on' if self.use_demucs else 'off'})",
        )
        processed_audio = preprocess_audio(
            self.audio_path,
            self.project_dir,
            use_demucs=self.use_demucs,
            force=self.force,
        )
        self._check_cancel()

        # Step 4: Transcribe (with checkpoint)
        word_segments_path = self.project_dir / "word_segments.json"
        lang_path = self.project_dir / "detected_lang.txt"
        if not self.force and word_segments_path.exists():
            import json

            self._emit("Transcribe", "Loading cached word segments")
            with open(word_segments_path, "r", encoding="utf-8") as f:
                word_segments = json.load(f)
            detected_lang = lang_path.read_text().strip() if lang_path.exists() else "en"
            self._emit("Transcribe", f"{len(word_segments)} words, lang={detected_lang} (from cache)")
        else:
            from subforge.transcription.factory import resolve_backend, resolve_model

            effective_language = self.source_language

            # Pre-transcription language detection when both backend and language are auto
            if self.asr_backend == "auto" and self.source_language == "auto":
                self._emit("Detect", "Running language detection…")
                from subforge.transcription.faster_whisper_transcriber import detect_language
                detect_model = WHISPER_TIER_MAP["large"]
                detected_hint, prob = detect_language(processed_audio, detect_model)
                self._emit("Detect", f"lang={detected_hint} (probability={prob:.2f})")
                effective_language = detected_hint

            requested_backend = resolve_backend(self.asr_backend, effective_language)

            # Give the caller (GUI / CLI) a chance to install a missing backend
            from subforge.transcription.factory import (
                is_backend_available, _BACKEND_EXTRA, _BACKEND_RUNTIME_PKG, DEFAULT_BACKEND,
            )
            if (
                not is_backend_available(requested_backend)
                and requested_backend != DEFAULT_BACKEND
            ):
                extra = _BACKEND_EXTRA.get(requested_backend, requested_backend)
                pkg = _BACKEND_RUNTIME_PKG.get(requested_backend, requested_backend)
                if self.missing_backend_handler:
                    self.missing_backend_handler(requested_backend, extra, pkg)
                    # Handler may have installed the package; get_transcriber
                    # will re-check and fallback if still unavailable.

            transcribe, concrete_backend = _resolve_transcriber(
                requested_backend, effective_language,
            )

            if concrete_backend != requested_backend:
                self._emit(
                    "Fallback",
                    f"{requested_backend} is not installed — falling back to {concrete_backend}",
                )
                self._fallback_from = requested_backend

            concrete_model = resolve_model(self.model_name, concrete_backend)

            self._emit(
                "Engine",
                f"backend={concrete_backend}, model={concrete_model}, lang={effective_language}",
            )

            word_segments, detected_lang = transcribe(
                processed_audio,
                concrete_model,
                progress_callback=self.progress_callback,
            )
            save_word_segments(word_segments, word_segments_path)
            lang_path.write_text(detected_lang, encoding="utf-8")
            self._emit("Transcribe", f"{len(word_segments)} words, lang={detected_lang} (saved checkpoint)")

        profile = get_profile(detected_lang)
        self._emit("NLP", f"Language profile: {profile.code} (join='{profile.join_token}', char_mode={profile.use_char_count})")
        self._check_cancel()

        # Step 4b: Punctuation restoration (optional, English-like languages only)
        if self.use_punctuation and not profile.skip_punctuation_model:
            self._emit("Punctuation", "Restoring punctuation")
            from subforge.nlp.punctuation import restore_punctuation

            word_segments = restore_punctuation(word_segments, profile)
            self._check_cancel()
        elif profile.skip_punctuation_model:
            self._emit("Punctuation", f"Skipped (not needed for {profile.code})")

        # Step 5–8: language-specific NLP pipeline (English vs CJK)
        strategy = get_strategy(profile)

        # Pre-Plan 2 — CJK transcript/timing split. For CJK languages we try
        # to use SenseVoice as the transcript backend while keeping Whisper
        # word_segments as the timing source. Whisper-only is the
        # documented fallback.
        sensevoice_text: str | None = None
        sensevoice_source: str | None = None
        sensevoice_backend: str | None = None
        sensevoice_model: str | None = None
        timing_backend: str | None = None
        timing_model: str | None = None
        transcript_fallback: str | None = None
        if not profile.use_spacy and CJK_USE_SENSEVOICE_TRANSCRIPT:
            sensevoice_text, sensevoice_source, transcript_fallback = (
                self._maybe_run_sensevoice_transcript(
                    processed_audio,
                    word_segments,
                )
            )
            if sensevoice_text is not None:
                sensevoice_backend = "sensevoice"
                sensevoice_model = SENSEVOICE_TIER_MAP.get(
                    self.model_name, SENSEVOICE_TIER_MAP["large"]
                )
                timing_backend = "whisper"
                timing_model = WHISPER_TIER_MAP.get(
                    self.model_name, WHISPER_TIER_MAP["large"]
                )

        ctx = StrategyContext(
            profile=profile,
            project_dir=self.project_dir,
            force=self.force,
            emit=self._emit,
            check_cancel=self._check_cancel,
            transcript_text=sensevoice_text,
            transcript_source=sensevoice_source,
            transcript_backend=sensevoice_backend,
            transcript_model=sensevoice_model,
            timing_backend=timing_backend,
            timing_model=timing_model,
            transcript_fallback=transcript_fallback,
        )
        refined = strategy.run(word_segments, ctx)
        self._check_cancel()

        # Step 9: Write SRT
        en_sentences = get_bounds_and_text(refined, profile=profile)

        if self.translate_method:
            self._emit("Translate", f"method={self.translate_method}, target={self.target_lang}")
            from subforge.translation.factory import create_translator

            translator = create_translator(
                self.translate_method,
                src_lang=TRANSLATE_SRC_LANG,
                tgt_lang=self.target_lang,
            )
            translation_cache = self.project_dir / "translation_cache"

            # Invalidate translation cache if segmentation changed
            seg_hash = hashlib.sha256(
                "\n".join(s["segment"] for s in en_sentences).encode()
            ).hexdigest()
            hash_file = self.project_dir / "translation_cache.hash"
            if hash_file.exists() and hash_file.read_text().strip() != seg_hash:
                logger.info("Segmentation changed, clearing translation cache")
                self._emit("Translate", "Segmentation changed, clearing cache")
                if translation_cache.exists():
                    shutil.rmtree(translation_cache)
            hash_file.write_text(seg_hash)

            try:
                en_sentences = translator.translate(
                    en_sentences,
                    cache_dir=translation_cache,
                    force=self.force,
                    progress_callback=self.progress_callback,
                )
            except Exception as exc:
                err_msg = str(exc)
                if "429" in err_msg or "quota" in err_msg.lower():
                    self._emit(
                        "Translate",
                        "FAILED: API quota exceeded. "
                        "Free tier daily limit reached. "
                        "Try again tomorrow or enable billing.",
                    )
                else:
                    self._emit("Translate", f"FAILED: {type(exc).__name__}: {exc}")
                logger.warning(
                    "Translation failed, outputting English-only SRT: %s", exc
                )
            self._check_cancel()

        srt_path = self.project_dir / "output.srt"
        has_translation = any(s.get("translation") for s in en_sentences)

        if has_translation:
            from subforge.subtitle.formatter import format_srt

            # Derive short language codes for filenames
            src_short = TARGET_LANG_SHORT.get(TRANSLATE_SRC_LANG, "src")
            tgt_short = TARGET_LANG_SHORT.get(self.target_lang, "tgt")

            # Bilingual (default)
            write_srt(en_sentences, srt_path)
            # Source language only
            src_path = self.project_dir / f"output_{src_short}.srt"
            src_path.write_text(
                format_srt(en_sentences, mode="source"), encoding="utf-8"
            )
            # Translation only
            tgt_path = self.project_dir / f"output_{tgt_short}.srt"
            tgt_path.write_text(
                format_srt(en_sentences, mode="translation"), encoding="utf-8"
            )
            self._emit(
                "Done",
                f"SRT written: {srt_path.name}, {src_path.name}, {tgt_path.name}",
            )
        else:
            write_srt(en_sentences, srt_path)
            self._emit("Done", f"SRT written to {srt_path}")

        # Step 10 (optional): Download MP4 with audio (skip for local files)
        if self.download_mp4 and not self.local_file:
            self._check_cancel()
            self._emit("Video", f"Downloading MP4 (quality={self.video_quality})")
            output_base = self.project_dir / self.title
            self.video_path = download_video(
                self.url, output_base, quality=self.video_quality, force=False
            )
            self._emit("Video", f"MP4 saved to {self.video_path}")

        return srt_path
