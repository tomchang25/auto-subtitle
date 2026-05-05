from setuptools import setup, find_packages

setup(
    name="auto-subtitle",
    version="0.1.0",
    author="Greysuki",
    description="Generate clean, accurate subtitles from YouTube videos.",
    packages=find_packages(),
    install_requires=[
        "yt-dlp",
        "faster-whisper",
        "spacy",
        "ffmpeg-python",
        "demucs",
        "PySide6",
        # torch: installed separately via setup.bat/sh with CUDA support
    ],
    extras_require={
        "experimental": [
            "nemo_toolkit[asr]",
        ],
        "dev": [
            "pytest",
        ],
    },
    python_requires=">=3.11",
)
