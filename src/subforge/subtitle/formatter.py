def _format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def format_srt(segments, mode: str = "bilingual"):
    """Format segments as SRT text.

    mode:
        "bilingual"  — original + translation (if available)
        "source"     — original language only
        "translation" — translation only (falls back to original if missing)
    """
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_time(seg["start"])
        end = _format_time(seg["end"])
        original = seg["segment"].strip()
        translation = (seg.get("translation") or "").strip()

        if mode == "bilingual":
            text = original
            if translation:
                text = text + "\n" + translation
        elif mode == "source":
            text = original
        elif mode == "translation":
            text = translation if translation else original
        else:
            raise ValueError(f"Unknown SRT format mode: {mode!r}")

        srt_lines.extend([str(i), f"{start} --> {end}", text, ""])

    return "\n".join(srt_lines)
