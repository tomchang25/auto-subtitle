from pathlib import Path
import nemo.collections.asr as nemo_asr

# You can also pass this in as a parameter instead
_loaded_models = {}


def load_model(model_name: str):
    if model_name not in _loaded_models:
        print(f"[ASR] Loading model: {model_name}")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        model.eval()
        _loaded_models[model_name] = model
    return _loaded_models[model_name]


def transcribe_audio_word_level(wav_path: Path, model_name: str) -> list:
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {wav_path}")

    model = load_model(model_name)

    print(f"[ASR] Transcribing: {wav_path}")
    results = model.transcribe([str(wav_path)], timestamps=True)

    segments = []
    if results and results[0].timestamp and "word" in results[0].timestamp:
        segments.extend(results[0].timestamp["word"])
        print(f"[ASR] Transcription complete: {len(segments)} words")
    else:
        raise ValueError("ASR model did not return word-level timestamps")

    return segments
