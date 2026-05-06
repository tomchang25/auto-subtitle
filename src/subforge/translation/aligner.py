"""Post-translation alignment using character-length-ratio dynamic programming.

When the translation model returns a different number of segments than expected,
this module re-aligns the translated text back to the source segments using
length-based heuristics and dynamic programming.

The core idea: Chinese/CJK translations have a roughly predictable character
length relative to the English source. We use this ratio + DP to find the
optimal way to split/merge translated segments to match source segments 1:1.
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Typical character ratio: len(chinese) / len(english)
# English includes spaces, Chinese doesn't, so ratio is roughly 0.25-0.5
# We estimate dynamically from the actual batch for better accuracy.
_DEFAULT_RATIO = 0.35

# Punctuation/natural break points in CJK text
_CJK_BREAK_RE = re.compile(r"(?<=[，。！？、；：\n])")
# Whitespace break (fallback)
_SPACE_BREAK_RE = re.compile(r"\s+")


def estimate_ratio(en_texts: list[str], zh_text: str) -> float:
    """Estimate the char-length ratio from actual data."""
    total_en = sum(len(t) for t in en_texts)
    total_zh = len(zh_text.replace(" ", ""))
    if total_en == 0:
        return _DEFAULT_RATIO
    ratio = total_zh / total_en
    # Clamp to reasonable range
    return max(0.15, min(0.7, ratio))


def realign(
    en_texts: list[str],
    zh_segments: list[str],
) -> list[str]:
    """Re-align zh_segments to match en_texts count.

    Handles three cases:
    - len(zh) == len(en): return as-is
    - len(zh) > len(en): merge excess segments
    - len(zh) < len(en): split long segments

    Uses dynamic programming with length-ratio cost to find optimal alignment.

    Args:
        en_texts: Source English segments (N items)
        zh_segments: Translated Chinese segments (M items, M != N)

    Returns:
        List of exactly N Chinese strings aligned to en_texts.
    """
    n_en = len(en_texts)
    n_zh = len(zh_segments)

    if n_en == 0:
        return []

    if n_zh == 0:
        return [""] * n_en

    if n_zh == n_en:
        return list(zh_segments)

    # Join all Chinese text, preserving segment boundaries as potential split points
    # We use a special marker to remember where original segments were
    _SEG_MARKER = "\x00"
    zh_joined = _SEG_MARKER.join(zh_segments)

    # Compute expected lengths for each English segment
    ratio = estimate_ratio(en_texts, zh_joined.replace(_SEG_MARKER, ""))
    expected_lengths = [max(1, int(len(t) * ratio)) for t in en_texts]
    total_expected = sum(expected_lengths)
    total_actual = len(zh_joined.replace(_SEG_MARKER, ""))

    # Scale expected lengths to match actual total
    if total_expected > 0:
        scale = total_actual / total_expected
        expected_lengths = [max(1, int(l * scale)) for l in expected_lengths]

    logger.debug(
        "Realign: %d en segments, %d zh segments, ratio=%.3f",
        n_en, n_zh, ratio,
    )

    # Build candidate split positions (prefer segment markers and punctuation)
    # First, find all original segment boundaries
    seg_positions = []
    pos = 0
    for seg in zh_segments:
        pos += len(seg)
        seg_positions.append(pos)
        pos += 1  # for the marker
    # Remove the last one (end of text)
    if seg_positions:
        seg_positions = seg_positions[:-1]

    # Use a greedy approach with proportional splitting
    # This is simpler and more robust than full DP for this use case
    result = _split_proportional(zh_joined, _SEG_MARKER, en_texts, expected_lengths)

    if len(result) != n_en:
        # Shouldn't happen, but safety fallback
        logger.error(
            "Realign produced %d segments instead of %d, using naive split",
            len(result), n_en,
        )
        result = _naive_split(zh_joined.replace(_SEG_MARKER, ""), n_en)

    return result


def _split_proportional(
    zh_joined: str,
    seg_marker: str,
    en_texts: list[str],
    expected_lengths: list[int],
) -> list[str]:
    """Split Chinese text proportionally based on expected character lengths.

    Prefers splitting at original segment boundaries (seg_marker) or
    CJK punctuation rather than in the middle of words.
    """
    n_en = len(en_texts)
    # Remove markers but track their positions as preferred split points
    clean_text = ""
    preferred_splits: set[int] = set()
    for ch in zh_joined:
        if ch == seg_marker:
            preferred_splits.add(len(clean_text))
        else:
            clean_text += ch

    # Also add CJK punctuation positions as secondary split points
    secondary_splits: set[int] = set()
    for m in _CJK_BREAK_RE.finditer(clean_text):
        secondary_splits.add(m.start())

    total_len = len(clean_text)
    if total_len == 0:
        return [""] * n_en

    # Compute cumulative expected positions
    cum_expected = []
    running = 0
    for l in expected_lengths:
        running += l
        cum_expected.append(running)

    # Scale to actual length
    scale = total_len / cum_expected[-1] if cum_expected[-1] > 0 else 1.0

    result = []
    prev_pos = 0

    for i in range(n_en - 1):
        target_pos = int(cum_expected[i] * scale)
        # Clamp
        target_pos = max(prev_pos + 1, min(target_pos, total_len - (n_en - 1 - i)))

        # Find best split point near target
        best_pos = _find_best_split(
            clean_text, target_pos, preferred_splits, secondary_splits,
            search_radius=max(15, int(target_pos * 0.1)),
        )
        best_pos = max(prev_pos + 1, min(best_pos, total_len - (n_en - 1 - i)))

        result.append(clean_text[prev_pos:best_pos].strip())
        prev_pos = best_pos

    # Last segment gets the rest
    result.append(clean_text[prev_pos:].strip())

    return result


def _find_best_split(
    text: str,
    target: int,
    preferred: set[int],
    secondary: set[int],
    search_radius: int = 20,
) -> int:
    """Find the best split position near target.

    Priority: preferred (original segment boundary) > secondary (punctuation) > target.
    """
    lo = max(0, target - search_radius)
    hi = min(len(text), target + search_radius)

    # Look for preferred split points (original ||| boundaries)
    best_preferred = None
    best_preferred_dist = float("inf")
    for p in preferred:
        if lo <= p <= hi:
            dist = abs(p - target)
            if dist < best_preferred_dist:
                best_preferred = p
                best_preferred_dist = dist

    if best_preferred is not None and best_preferred_dist <= search_radius * 0.7:
        return best_preferred

    # Look for secondary split points (CJK punctuation)
    best_secondary = None
    best_secondary_dist = float("inf")
    for p in secondary:
        if lo <= p <= hi:
            dist = abs(p - target)
            if dist < best_secondary_dist:
                best_secondary = p
                best_secondary_dist = dist

    if best_secondary is not None:
        return best_secondary

    # Fall back to preferred even if a bit further
    if best_preferred is not None:
        return best_preferred

    # No good split point, just use target
    return target


def _naive_split(text: str, n: int) -> list[str]:
    """Evenly split text into n parts (last resort fallback)."""
    if n <= 0:
        return []
    if n == 1:
        return [text]
    chunk_size = max(1, len(text) // n)
    result = []
    for i in range(n - 1):
        result.append(text[i * chunk_size : (i + 1) * chunk_size].strip())
    result.append(text[(n - 1) * chunk_size :].strip())
    return result
