import argparse

from subforge.pipeline.processor import SubtitlePipeline
from subforge.config import (
    WHISPER_MODEL,
    OUTPUT_DIR,
    DEFAULT_URL,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate subtitles from a YouTube URL.")
    parser.add_argument("--url", help="YouTube URL (prompted if omitted).")
    parser.add_argument(
        "--model",
        default=None,
        help="Whisper model name (default: %(default)s).",
    )
    parser.add_argument(
        "--download-video",
        action="store_true",
        default=False,
        help="Also download the MP4 video (with audio) for testing with SRT.",
    )
    parser.add_argument(
        "--video-quality",
        default="1080p",
        choices=("480p", "720p", "1080p", "1440p", "2160p"),
        help="Video quality when downloading MP4 (default: 1080p).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    url = args.url or input("Enter YouTube URL: ").strip() or DEFAULT_URL

    model_name = args.model or WHISPER_MODEL

    pipeline = SubtitlePipeline(
        url=url,
        model_name=model_name,
        output_dir=OUTPUT_DIR,
        download_mp4=args.download_video,
        video_quality=args.video_quality,
    )

    srt_path = pipeline.run()
    print(f"\nSRT file: {srt_path}")
    if pipeline.video_path:
        print(f"MP4 file: {pipeline.video_path}")


if __name__ == "__main__":
    main()
