import argparse
import logging
import sys


def parse_args():
    from subforge.translation.factory import BACKEND_NAMES
    from subforge.transcription.factory import BACKEND_NAMES as ASR_BACKEND_NAMES
    from subforge.config import MODEL_TIER, MODEL_TIERS, TARGET_LANGUAGES, TRANSLATE_TGT_LANG, ASR_BACKEND, ASR_SOURCE_LANGUAGE, CHINESE_BENCHMARK_MODE

    parser = argparse.ArgumentParser(
        description="Generate subtitles from a YouTube URL or local media file."
    )
    parser.add_argument("--url", help="YouTube URL (prompted if omitted).")
    parser.add_argument(
        "--file",
        default=None,
        help="Path to a local audio/video file (overrides --url).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Abstract model tier ({', '.join(MODEL_TIERS)}) or an explicit backend model name. "
            f"Default tier: {MODEL_TIER}."
        ),
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
        "--target-lang",
        default=TRANSLATE_TGT_LANG,
        choices=list(TARGET_LANGUAGES.keys()),
        help=(
            f"Target language for translation (default: {TRANSLATE_TGT_LANG}). "
            f"Available: {', '.join(f'{k} ({v})' for k, v in TARGET_LANGUAGES.items())}"
        ),
    )
    parser.add_argument(
        "--asr-backend",
        default=None,
        choices=ASR_BACKEND_NAMES,
        help=(
            f"ASR backend to use (default: {ASR_BACKEND}). "
            "'auto' always selects whisper. "
            "Use 'funasr' (Paraformer) or 'sensevoice' for legacy CJK experiments. "
            "Use 'fireredasr' for FireRedASR-AED-L benchmarks."
        ),
    )
    parser.add_argument(
        "--source-language",
        default=None,
        help=(
            f"Source language ISO 639-1 code hint (default: {ASR_SOURCE_LANGUAGE}). "
            "Used for language-profile selection and (with --asr-backend auto) for "
            "backend routing. 'auto' detects from audio via Whisper."
        ),
    )
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        default=False,
        help="Disable punctuation restoration (local model).",
    )
    parser.add_argument(
        "--chinese-benchmark",
        action="store_true",
        default=CHINESE_BENCHMARK_MODE,
        help=(
            "Chinese ASR benchmark mode: bypass semantic splitting, timing refinement, "
            "soft segmentation, and merging. Applies a simple hard-cut strategy so the "
            "SRT reflects raw ASR output for clean benchmark comparison."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Ignore cached results and re-run all steps from scratch.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG logging (shows full prompts/responses).",
    )
    return parser.parse_args()


def main():
    from subforge.utils import check_dependencies
    check_dependencies(gui=False)

    from subforge.pipeline.processor import SubtitlePipeline
    from subforge.config import MODEL_TIER, OUTPUT_DIR, DEFAULT_URL, ASR_BACKEND, ASR_SOURCE_LANGUAGE

    from subforge.config import LOG_LEVEL
    args = parse_args()
    log_level = logging.DEBUG if args.debug else LOG_LEVEL
    logging.basicConfig(level=log_level, format="%(levelname)s - %(name)s - %(message)s")
    local_file = args.file
    if local_file:
        url = ""
    else:
        url = args.url or input("Enter YouTube URL: ").strip() or DEFAULT_URL

    model_name = args.model or MODEL_TIER
    asr_backend = args.asr_backend or ASR_BACKEND
    source_language = args.source_language or ASR_SOURCE_LANGUAGE

    def _cli_missing_backend(backend: str, extra: str, pip_pkg: str) -> bool:
        """Prompt the user in the terminal to install or skip."""
        answer = input(
            f"'{backend}' is not installed. Install now? "
            f"(pip install {pip_pkg}) [y/N] "
        ).strip().lower()
        if answer == "y":
            import subprocess
            print(f"Installing {pip_pkg}…")
            ret = subprocess.call(
                [sys.executable, "-m", "pip", "install", pip_pkg],
            )
            if ret != 0:
                print("Installation failed. Falling back to whisper.")
                return False
            print("Installation successful.")
            return True
        print("Falling back to whisper.")
        return False

    pipeline = SubtitlePipeline(
        url=url,
        model_name=model_name,
        output_dir=OUTPUT_DIR,
        download_mp4=args.download_video,
        use_punctuation=not args.no_punctuation,
        video_quality=args.video_quality,
        translate_method=args.translate,
        target_lang=args.target_lang,
        force=args.force,
        local_file=local_file,
        asr_backend=asr_backend,
        source_language=source_language,
        missing_backend_handler=_cli_missing_backend,
        chinese_benchmark=args.chinese_benchmark,
    )

    logger = logging.getLogger(__name__)
    srt_path = pipeline.run()
    logger.info("SRT file: %s", srt_path)
    if pipeline.video_path:
        logger.info("MP4 file: %s", pipeline.video_path)


if __name__ == "__main__":
    main()
