from setuptools import setup, find_packages

# Keep install_requires in sync with requirements.txt (base tier).
# torch / torchaudio are deliberately excluded — they need a platform-specific
# wheel (CUDA, CPU, or MPS) and are installed by setup.bat / setup.sh.
setup(
    name="auto-subtitle",
    version="0.1.0",
    author="Greysuki",
    description="Generate clean, accurate subtitles from YouTube videos.",
    packages=find_packages(),
    install_requires=[
        "faster-whisper",
        "PySide6-Essentials",
        "spacy",
        "demucs",
        "pydub",
        "yt-dlp",
        "ffmpeg-python",
        "pysubs2",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    python_requires=">=3.11",
)
