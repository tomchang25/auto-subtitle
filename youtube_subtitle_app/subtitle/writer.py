from pathlib import Path
from youtube_subtitle_app.subtitle.formatter import format_srt


def write_srt(segments, output_path: Path):
    srt_text = format_srt(segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    return output_path
