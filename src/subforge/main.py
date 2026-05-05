import argparse
import logging


def parse_args():
    from subforge.translation.factory import BACKEND_NAMES

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
    parser.add_argument(
        "--translate",
        default=None,
        choices=BACKEND_NAMES,
        help="Enable translation and choose backend (default: disabled).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Ignore cached results and re-run all steps from scratch.",
    )
    return parser.parse_args()


def main():
    from subforge.utils import check_dependencies
    check_dependencies(gui=False)

    from subforge.pipeline.processor import SubtitlePipeline
    from subforge.config import WHISPER_MODEL, OUTPUT_DIR, DEFAULT_URL

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
    args = parse_args()
    url = args.url or input("Enter YouTube URL: ").strip() or DEFAULT_URL

    model_name = args.model or WHISPER_MODEL

    pipeline = SubtitlePipeline(
        url=url,
        model_name=model_name,
        output_dir=OUTPUT_DIR,
        download_mp4=args.download_video,
        video_quality=args.video_quality,
        translate_method=args.translate,
        force=args.force,
    )

    logger = logging.getLogger(__name__)
    srt_path = pipeline.run()
    logger.info("SRT file: %s", srt_path)
    if pipeline.video_path:
        logger.info("MP4 file: %s", pipeline.video_path)


if __name__ == "__main__":
    main()
