"""Cache and artifact helpers for the staged subtitle pipeline.

The runner reads only canonical staged artifacts under
``project_dir/stages/``. Legacy mirrors (e.g. ``project_dir/cjk/``) are
write-only — populated by these helpers and never used as a cache
source.

Cache hits on a canonical artifact opportunistically backfill the
legacy mirror when it is missing or stale, so deleting the legacy
directory and rerunning from cache still produces a complete mirror.
Repeated cached runs avoid bumping mtime when contents have not
changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Bumped whenever the staged-runner artifact contract changes so caches
# from earlier layouts (including the pre-runner CJK strategy) are
# ignored on the first post-bump run.
STAGE_SCHEMA_VERSION = "v4"

# Canonical artifact directory under each project. The runner reads only
# from this directory; legacy mirror directories declared by the policy
# are write-only.
CANONICAL_DIRNAME = "stages"

STAGE_FILES: tuple[str, ...] = (
    "raw_transcript.json",
    "timing_anchors.json",
    "corrected_transcript.json",
    "sentences.json",
    "alignment.json",
    "final_cues.json",
)


def hash_inputs(*parts: str) -> str:
    """Return a stable SHA-256 over the given string parts."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def write_text_if_changed(path: Path, content: str) -> None:
    """Write *content* to *path* unless the file already has it.

    Used for both canonical and mirror writes so repeated cached runs do
    not bump mtime when nothing changed.
    """
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing == content:
            return
    path.write_text(content, encoding="utf-8")


def delete_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def clear_stage_files(canonical_dir: Path, legacy_dir: Path | None) -> None:
    """Delete every known stage file under canonical and legacy dirs."""
    for name in STAGE_FILES:
        delete_if_exists(canonical_dir / name)
        if legacy_dir is not None:
            delete_if_exists(legacy_dir / name)


def serialize_stage(input_hash: str, data) -> str:
    """Wrap *data* with its ``input_hash`` and return JSON text."""
    return json.dumps(
        {"input_hash": input_hash, "data": data},
        ensure_ascii=False,
        indent=2,
    )


def cached_stage_with_mirror(
    canonical_path: Path,
    legacy_path: Path | None,
    expected_hash: str,
):
    """Load a cached stage and refresh the legacy mirror if it lags.

    Returns the deserialized ``data`` payload on hit, or ``None`` on
    miss. On hit, the canonical artifact's bytes are mirrored into
    *legacy_path* via :func:`write_text_if_changed` so deleting the
    legacy directory and rerunning from cache still leaves a complete
    mirror.
    """
    if not canonical_path.exists():
        return None
    try:
        text = canonical_path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Stage cache unreadable at %s: %s", canonical_path, exc)
        return None
    if not isinstance(data, dict):
        return None
    if data.get("input_hash") != expected_hash:
        return None
    if legacy_path is not None:
        write_text_if_changed(legacy_path, text)
    return data.get("data")


def save_stage_with_mirror(
    canonical_path: Path,
    legacy_path: Path | None,
    input_hash: str,
    data,
) -> None:
    """Persist *data* to canonical and mirror to *legacy_path* if given."""
    serialized = serialize_stage(input_hash, data)
    canonical_path.write_text(serialized, encoding="utf-8")
    if legacy_path is not None:
        write_text_if_changed(legacy_path, serialized)


def write_artifact_with_mirror(
    canonical_path: Path,
    legacy_path: Path | None,
    content: str,
) -> None:
    """Write a non-cache artifact (e.g. ``final_cues.json``) with mirror.

    Uses :func:`write_text_if_changed` for both paths so repeated runs
    that produce the same bytes do not bump mtime.
    """
    write_text_if_changed(canonical_path, content)
    if legacy_path is not None:
        write_text_if_changed(legacy_path, content)


__all__ = [
    "CANONICAL_DIRNAME",
    "STAGE_FILES",
    "STAGE_SCHEMA_VERSION",
    "cached_stage_with_mirror",
    "clear_stage_files",
    "delete_if_exists",
    "hash_inputs",
    "save_stage_with_mirror",
    "serialize_stage",
    "write_artifact_with_mirror",
    "write_text_if_changed",
]
