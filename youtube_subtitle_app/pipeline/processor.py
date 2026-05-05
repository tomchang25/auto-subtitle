from pathlib import Path
from typing import Callable, Optional

from youtube_subtitle_app.downloader.youtube import (
    get_video_title,
    download_audio,
    download_video,
)

from youtube_subtitle_app.audio.preprocess import preprocess_audio

from youtube_subtitle_app.nlp.text_semantically import (
    split_to_sentences,
)
from youtube_subtitle_app.nlp.alignment import (
    align_sentences_with_timestamps,
    refine_sentences_by_timing,
)

from youtube_subtitle_app.subtitle.writer import write_srt

from youtube_subtitle_app.config import (
    DEFAULT_MODEL,
    WHISPER_MODEL,
    ENGINE,
    OUTPUT_DIR,
)
from youtube_subtitle_app.utils import get_bounds_and_text


ProgressCallback = Callable[[str, str], None]


def _resolve_transcriber(engine: str):
    if engine == "parakeet":
        from youtube_subtitle_app.transcription.nemo_transcriber import (
            transcribe_audio_word_level,
        )
        return transcribe_audio_word_level
    if engine == "faster-whisper":
        from youtube_subtitle_app.transcription.faster_whisper_transcriber import (
            transcribe_audio_word_level,
        )
        return transcribe_audio_word_level
    raise ValueError(f"Unknown transcription engine: {engine}")


def _default_model_for(engine: str) -> str:
    if engine == "parakeet":
        return DEFAULT_MODEL
    if engine == "faster-whisper":
        return WHISPER_MODEL
    raise ValueError(f"Unknown transcription engine: {engine}")


class SubtitlePipeline:
    def __init__(
        self,
        url: str,
        model_name: Optional[str] = None,
        output_dir: Path = OUTPUT_DIR,
        engine: str = ENGINE,
        use_demucs: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.url = url
        self.engine = engine
        self.model_name = model_name or _default_model_for(engine)
        self.output_dir = output_dir
        self.use_demucs = use_demucs
        self.progress_callback = progress_callback
        self.project_dir = None
        self.title = None
        self.audio_path = None

    def _emit(self, step: str, detail: str = ""):
        message = f"[{step}] {detail}" if detail else f"[{step}]"
        print(message)
        if self.progress_callback is not None:
            try:
                self.progress_callback(step, detail)
            except Exception as exc:
                print(f"[Pipeline] progress_callback raised: {exc}")

    def run(self) -> Path:
        self._emit("Pipeline", "=== Starting Subtitle Pipeline ===")

        # Step 1: Get title and prepare directories
        self._emit("Title", "Fetching video title")
        self.title = get_video_title(self.url)
        self.project_dir = self.output_dir / self.title
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self._emit("Title", f"Project directory: {self.project_dir}")

        # Step 2: Download audio
        self._emit("Download", "Downloading audio")
        output_base = self.project_dir / self.title
        self.audio_path = download_audio(
            self.url, output_base, format="mp3", force=False
        )

        # Step 3: Preprocess audio
        self._emit(
            "Preprocess",
            f"Preparing WAV (demucs={'on' if self.use_demucs else 'off'})",
        )
        processed_audio = preprocess_audio(
            self.audio_path, self.project_dir, use_demucs=self.use_demucs
        )

        # Step 4: Transcribe
        self._emit(
            "Transcribe",
            f"engine={self.engine} model={self.model_name}",
        )
        transcribe = _resolve_transcriber(self.engine)
        word_segments = transcribe(processed_audio, self.model_name)
        self._emit("Transcribe", f"{len(word_segments)} words")

        # Step 5: NLP sentence splitting
        self._emit("NLP", "Splitting into sentences")
        full_text = " ".join([seg["word"] for seg in word_segments])
        sentence_chunks = split_to_sentences(full_text)

        # Step 6: Timestamp alignment
        self._emit("Align", "Aligning sentences with timestamps")
        aligned = align_sentences_with_timestamps(word_segments, sentence_chunks)

        # Step 7: Refinement
        self._emit("Refine", "Refining segment timing")
        refined = refine_sentences_by_timing(aligned)

        # Step 8: Write SRT
        en_sentences = get_bounds_and_text(refined)
        srt_path = self.project_dir / "output.srt"
        write_srt(en_sentences, srt_path)
        self._emit("Done", f"SRT written to {srt_path}")
        return srt_path
