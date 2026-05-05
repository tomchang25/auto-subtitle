import logging
from pathlib import Path
from typing import Callable, Optional

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
)
from subforge.nlp.alignment import (
    align_sentences_with_timestamps,
    refine_sentences_by_timing,
)
from subforge.nlp.segmentation import split_long_sentences_by_length

from subforge.subtitle.writer import write_srt

from subforge.config import (
    WHISPER_MODEL,
    OUTPUT_DIR,
    MAX_WORDS,
    SOFT_LIMIT,
)
from subforge.utils import get_bounds_and_text

ProgressCallback = Callable[[str, str], None]


def _resolve_transcriber():
    from subforge.transcription.faster_whisper_transcriber import (
        transcribe_audio_word_level,
    )

    return transcribe_audio_word_level


class SubtitlePipeline:
    def __init__(
        self,
        url: str,
        model_name: Optional[str] = None,
        output_dir: Path = OUTPUT_DIR,
        use_demucs: bool = True,
        download_mp4: bool = False,
        video_quality: str = "1080p",
        translate_method: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.url = url
        self.model_name = model_name or WHISPER_MODEL
        self.output_dir = output_dir
        self.use_demucs = use_demucs
        self.download_mp4 = download_mp4
        self.video_quality = video_quality
        self.translate_method = translate_method
        self.progress_callback = progress_callback
        self._cancelled = False
        self.project_dir = None
        self.title = None
        self.audio_path = None
        self.video_path = None

    def cancel(self):
        """Request cancellation. The pipeline will stop before the next step."""
        self._cancelled = True

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

    def run(self) -> Path:
        self._emit("Pipeline", "=== Starting Subtitle Pipeline ===")

        # Step 1: Get title and prepare directories
        self._emit("Title", "Fetching video title")
        self.title = get_video_title(self.url)
        self.project_dir = self.output_dir / self.title
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._emit("Title", f"Project directory: {self.project_dir}")
        self._check_cancel()

        # Step 2: Download audio
        self._emit("Download", "Downloading audio")
        output_base = self.project_dir / self.title
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
            self.audio_path, self.project_dir, use_demucs=self.use_demucs
        )
        self._check_cancel()

        # Step 4: Transcribe
        self._emit(
            "Transcribe",
            f"model={self.model_name}",
        )
        transcribe = _resolve_transcriber()
        word_segments = transcribe(processed_audio, self.model_name)
        self._emit("Transcribe", f"{len(word_segments)} words")
        self._check_cancel()

        # Step 5: NLP sentence splitting
        self._emit("NLP", "Splitting into sentences")
        full_text = " ".join([seg["word"] for seg in word_segments])
        sentence_chunks = split_to_sentences(full_text)
        self._check_cancel()

        # Step 6: Timestamp alignment
        self._emit("Align", "Aligning sentences with timestamps")
        aligned = align_sentences_with_timestamps(word_segments, sentence_chunks)
        self._check_cancel()

        # Step 7: Refinement
        self._emit("Refine", "Refining segment timing")
        refined = refine_sentences_by_timing(aligned)
        self._check_cancel()

        # Step 8: Split long segments
        self._emit("Split", "Splitting long segments by word count")
        refined = split_long_sentences_by_length(
            refined, min_words=SOFT_LIMIT, max_words=MAX_WORDS
        )
        self._check_cancel()

        # Step 9: Write SRT
        en_sentences = get_bounds_and_text(refined)

        if self.translate_method:
            self._emit("Translate", f"method={self.translate_method}")
            from subforge.translation.factory import create_translator

            translator = create_translator(self.translate_method)
            en_sentences = translator.translate(en_sentences)
            self._check_cancel()

        srt_path = self.project_dir / "output.srt"
        write_srt(en_sentences, srt_path)
        self._emit("Done", f"SRT written to {srt_path}")

        # Step 10 (optional): Download MP4 with audio
        if self.download_mp4:
            self._check_cancel()
            self._emit("Video", f"Downloading MP4 (quality={self.video_quality})")
            output_base = self.project_dir / self.title
            self.video_path = download_video(
                self.url, output_base, quality=self.video_quality, force=False
            )
            self._emit("Video", f"MP4 saved to {self.video_path}")

        return srt_path
