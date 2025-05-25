def format_srt(segments):
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = round((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = format_time(seg["start"])
        end = format_time(seg["end"])
        text = seg["segment"].strip()
        srt_lines.extend([str(i), f"{start} --> {end}", text, ""])

    return "\n".join(srt_lines)
