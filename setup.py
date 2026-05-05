from setuptools import setup, find_packages

setup(
    name="youtube_subtitle_app",
    version="0.1.0",
    author="Greysuki",
    description="A tool to download YouTube audio and generate subtitles using ASR and NLP.",
    packages=find_packages(),
    install_requires=[
        "yt-dlp",
        "nemo_toolkit[asr]",
        "faster-whisper",
        "spacy",
        "ffmpeg-python",
        "torch",
        "demucs",
        "PySide6",
        "pytest",
    ],
    python_requires=">=3.11",
)
