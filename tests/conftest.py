import importlib
import importlib.util
from pathlib import Path

collect_ignore = []

# Skip tests that require heavy optional dependencies
_optional_test_deps = {
    "test_audio_preprocess.py": "demucs",
    "test_downloader.py": "yt_dlp",
}

_this_dir = Path(__file__).parent

for test_file, module_name in _optional_test_deps.items():
    if importlib.util.find_spec(module_name) is None:
        collect_ignore.append(str(_this_dir / test_file))
