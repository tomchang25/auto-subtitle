from pathlib import Path
import demucs.separate
import os


def run_demucs(input_path: Path, output_dir: Path) -> Path:
    """
    Runs Demucs to separate vocals from a given audio file.
    Returns the path to the separated vocals (or the original file if separation fails).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--mp3",
        "--two-stems",
        "vocals",
        "-o",
        str(output_dir),
        "--filename",
        "{stem}.{ext}",
        str(input_path),
    ]

    print(f"[Demucs] Processing: {input_path}")
    try:
        demucs.separate.main(args)
        output_file = output_dir / "htdemucs" / input_path.stem / "vocals.mp3"
        if output_file.exists():
            print(f"[Demucs] Vocal track saved: {output_file}")
            return output_file
        else:
            print(f"[Demucs] Expected output not found, using original.")
            return input_path
    except Exception as e:
        print(f"[Demucs] Error: {e}")
        return input_path
