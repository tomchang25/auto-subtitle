from youtube_subtitle_app.downloader.youtube import (
    get_video_title,
    download_audio,
    download_video,
)

TEST_URL = "https://www.youtube.com/watch?v=392JUMCBSQY"  # Mexican Navy ship hits Brooklyn Bridge in New York City | BBC News


def test_get_video_title():
    title = get_video_title(TEST_URL)
    assert isinstance(title, str)
    assert len(title) > 0
    print(f"Retrieved video title: {title}")


def test_download_audio(tmp_path):
    output_path = tmp_path / "test_audio"
    audio_file = download_audio(TEST_URL, output_path, format="mp3", force=True)
    assert audio_file.exists()
    assert audio_file.suffix == ".mp3"
    print(f"Audio downloaded to: {audio_file}")


def test_download_video(tmp_path):
    output_path = tmp_path / "test_video"
    video_file = download_video(TEST_URL, output_path, quality="1080p", force=True)
    assert video_file.exists()
    assert video_file.suffix == ".mp4"
    print(f"Video downloaded to: {video_file}")
