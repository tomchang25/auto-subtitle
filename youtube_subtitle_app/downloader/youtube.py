from pathlib import Path
import yt_dlp
import re


def sanitize_filename(title: str) -> str:
    return re.sub(r"[^\w\s-]", "", title).strip()


def get_video_title(url: str) -> str:
    try:
        with yt_dlp.YoutubeDL(
            {"noplaylist": True, "quiet": True, "no_warnings": True}
        ) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info or "title" not in info:
            raise Exception("Failed to retrieve video info.")
        return sanitize_filename(info["title"])
    except Exception as e:
        raise RuntimeError(f"Error retrieving video title: {e}")


def download_video(
    url: str, output_path: Path, quality: str = "1080p", force: bool = False
) -> Path:
    video_path = output_path.with_suffix(".mp4")

    if not force and video_path.exists():
        print(f"Video already exists: {video_path}")
        return video_path

    format_string = f"bestvideo[height<={quality.replace('p', '')}][ext=mp4]+bestaudio[ext=m4a]/best[height<={quality.replace('p', '')}][ext=mp4]"
    opts = {
        "outtmpl": str(output_path),
        "format": format_string,
        "noplaylist": True,
    }

    try:
        print(f"Downloading video to: {video_path}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return video_path
    except Exception as e:
        if video_path.exists():
            video_path.unlink()
        raise RuntimeError(f"Error downloading video: {e}")


def download_audio(
    url: str, output_path: Path, format: str = "mp3", force: bool = False
) -> Path:
    audio_path = output_path.with_suffix(f".{format}")

    if not force and audio_path.exists():
        print(f"Audio already exists: {audio_path}")
        return audio_path

    opts = {
        "outtmpl": str(output_path),
        "format": "bestaudio/best",
        "extract-audio": True,
        "audio-format": format,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": format,
                "preferredquality": "192",
            }
        ],
    }

    try:
        print(f"Downloading audio to: {audio_path}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return audio_path
    except Exception as e:
        if audio_path.exists():
            audio_path.unlink()
        raise RuntimeError(f"Error downloading audio: {e}")
