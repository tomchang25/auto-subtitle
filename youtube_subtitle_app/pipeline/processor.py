from pathlib import Path
from youtube_subtitle_app.downloader.youtube import (
    get_video_title,
    download_audio,
)

from youtube_subtitle_app.audio.preprocess import preprocess_audio

from youtube_subtitle_app.transcription.nemo_transcriber import (
    transcribe_audio_word_level,
)

from youtube_subtitle_app.nlp.text_semantically import (
    split_to_sentences,
)
from youtube_subtitle_app.nlp.alignment import refine_chunks_by_time

from youtube_subtitle_app.subtitle.writer import write_srt

from youtube_subtitle_app.config import *


class SubtitlePipeline:
    def __init__(self, url: str, model_name=DEFAULT_MODEL, output_dir=OUTPUT_DIR):
        self.url = url
        self.model_name = model_name
        self.output_dir = output_dir
        self.project_dir = None
        self.title = None
        self.audio_path = None

    def run(self):
        print("=== Starting Subtitle Pipeline ===")

        # Step 1: Get title and prepare directories
        self.title = get_video_title(self.url)
        self.project_dir = self.output_dir / self.title
        self.project_dir.mkdir(parents=True, exist_ok=True)
        print(f"Project directory: {self.project_dir}")

        # Step 2: Download audio
        output_base = self.project_dir / self.title
        self.audio_path = download_audio(
            self.url, output_base, format="mp3", force=False
        )

        # Step 3: Preprocess audio
        processed_audio = preprocess_audio(self.audio_path, self.project_dir)

        # Step 4: Transcribe
        word_segments = transcribe_audio_word_level(processed_audio, self.model_name)

        # Step 5: NLP sentence splitting
        full_text = " ".join([seg["word"] for seg in word_segments])
        sentence_chunks = split_to_sentences(full_text)

        # Step 6: Refine by time
        refined_chunks = refine_chunks_by_time(word_segments, sentence_chunks)

        # 7. Write to SRT
        srt_path = self.project_dir / "output.srt"
        write_srt(refined_chunks, srt_path)
