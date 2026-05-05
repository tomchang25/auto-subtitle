import importlib
import importlib.util

collect_ignore = []

# Skip tests that require heavy optional dependencies
_optional_test_deps = {
    "tests/test_audio_preprocess.py": "demucs",
    "tests/test_downloader.py": "yt_dlp",
}

for test_file, module_name in _optional_test_deps.items():
    if importlib.util.find_spec(module_name) is None:
        collect_ignore.append(test_file)
