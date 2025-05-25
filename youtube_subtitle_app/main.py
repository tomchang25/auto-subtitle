from youtube_subtitle_app.pipeline.processor import SubtitlePipeline
from youtube_subtitle_app.config import DEFAULT_MODEL, OUTPUT_DIR, DEFAULT_URL


def main():
    url = input("Enter YouTube URL: ").strip() or DEFAULT_URL

    pipeline = SubtitlePipeline(
        url=url, model_name=DEFAULT_MODEL, output_dir=OUTPUT_DIR
    )

    pipeline.run()


if __name__ == "__main__":
    main()
