import hashlib
import logging
import shutil
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

from subforge.nlp.text_semantically import (
    split_to_sentences,
    split_word_segments_by_punctuation,
)
from subforge.nlp.alignment import (
    align_sentences_with_timestamps,
    refine_sentences_by_timing,
)
from subforge.nlp.segmentation import split_long_sentences_by_length, merge_short_segments

from subforge.subtitle.writer import write_srt

from subforge.config import (
    WHISPER_MODEL,
    OUTPUT_DIR,
    MAX_GAP,
    MIN_DURATION,
    BREATH_GAP,
    MIN_WORDS_FOR_BREATH_SPLIT,
    SEG_MIN_WORDS,
    SEG_SOFT_WORDS,
    SEG_HARD_WORDS,
    SEG_PAUSE_THRESHOLD,
    MERGE_MAX_WORDS,
    MERGE_MAX_DURATION,
    MERGE_MAX_GAP,
    USE_LLM_PUNCTUATION,
    TRANSLATE_TGT_LANG,
    TRANSLATE_SRC_LANG,
    TARGET_LANG_SHORT,
)
from subforge.nlp.lang_profile import LanguageProfile, get_profile, DEFAULT as DEFAULT_PROFILE
from subforge.utils import get_bounds_and_text, save_word_segments

ProgressCallback = Callable[[str, str], None]


def _resolve_transcriber():
    from subforge.transcription.faster_whisper_transcriber import (
        transcribe_audio_word_level,
    )

    return transcribe_audio_word_level


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
    ):
        self.url = url
        self.local_file = Path(local_file) if local_file else None
        self.model_name = model_name or WHISPER_MODEL
        self.output_dir = output_dir
        self.use_demucs = use_demucs
        self.use_punctuation = use_punctuation
        self.download_mp4 = download_mp4
        self.video_quality = video_quality
        self.translate_method = translate_method
        self.target_lang = target_lang
        self.progress_callback = progress_callback
        self.force = force
        self._cancelled = False
        self.project_dir = None
        self.title = None
        self.audio_path = None
        self.video_path = None

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
        import shutil
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
            self._emit(
                "Transcribe",
                f"model={self.model_name}",
            )
            transcribe = _resolve_transcriber()
            word_segments, detected_lang = transcribe(
                processed_audio,
                self.model_name,
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

        if profile.use_spacy:
            # --- English path: spaCy tokenise → align → refine ---
            # Step 5: NLP sentence splitting
            self._emit("NLP", "Splitting into sentences (spaCy)")
            full_text = profile.join_token.join([seg["word"] for seg in word_segments])
            sentence_chunks = split_to_sentences(full_text)
            self._check_cancel()

            # Step 6: Timestamp alignment
            self._emit("Align", "Aligning sentences with timestamps")
            aligned = align_sentences_with_timestamps(word_segments, sentence_chunks)
            self._check_cancel()

            # Step 7: Refinement
            self._emit("Refine", "Refining segment timing")
            refined = refine_sentences_by_timing(
                aligned,
                min_duration=MIN_DURATION,
                max_gap=MAX_GAP,
                breath_gap=BREATH_GAP,
                min_words_for_breath_split=MIN_WORDS_FOR_BREATH_SPLIT,
            )
            self._check_cancel()
        else:
            # --- CJK path: split word_segments directly by punctuation ---
            self._emit("NLP", "Splitting by punctuation (CJK)")
            refined = split_word_segments_by_punctuation(word_segments, profile)
            self._check_cancel()

            # Still apply timing refinement
            self._emit("Refine", "Refining segment timing")
            refined = refine_sentences_by_timing(
                refined,
                min_duration=MIN_DURATION,
                max_gap=MAX_GAP,
                breath_gap=BREATH_GAP,
                min_words_for_breath_split=MIN_WORDS_FOR_BREATH_SPLIT,
            )
            self._check_cancel()

        # Step 8: Split long segments
        self._emit("Split", "Splitting long segments")
        refined = split_long_sentences_by_length(
            refined,
            min_words=profile.seg_min,
            max_words=profile.seg_hard,
            soft_words=profile.seg_soft,
            pause_threshold=SEG_PAUSE_THRESHOLD,
            profile=profile,
        )

        # Step 8b: Merge short segments back together
        self._emit("Merge", "Merging short segments")
        refined = merge_short_segments(
            refined,
            max_words=profile.merge_max,
            max_duration=MERGE_MAX_DURATION,
            max_gap=MERGE_MAX_GAP,
            profile=profile,
        )
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
