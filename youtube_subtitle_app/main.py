import argparse

from youtube_subtitle_app.pipeline.processor import SubtitlePipeline
from youtube_subtitle_app.config import (
    DEFAULT_MODEL,
    WHISPER_MODEL,
    ENGINE,
    OUTPUT_DIR,
    DEFAULT_URL,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate subtitles from a YouTube URL.")
    parser.add_argument("--url", help="YouTube URL (prompted if omitted).")
    parser.add_argument(
        "--engine",
        choices=("parakeet", "faster-whisper"),
        default=ENGINE,
        help="Transcription engine.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name. Defaults to the engine's configured default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    url = args.url or input("Enter YouTube URL: ").strip() or DEFAULT_URL

    if args.model is not None:
        model_name = args.model
    elif args.engine == "parakeet":
        model_name = DEFAULT_MODEL
    else:
        model_name = WHISPER_MODEL

    pipeline = SubtitlePipeline(
        url=url,
        engine=args.engine,
        model_name=model_name,
        output_dir=OUTPUT_DIR,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
