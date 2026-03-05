"""Segmenting long waveforms into fixed-size windows."""

from __future__ import annotations

import math
from typing import List


def compute_segment_starts(
    num_samples: int,
    window_samples: int,
    hop_samples: int,
) -> List[int]:
    """Return segment start offsets that cover an audio clip.

    For very short clips, returns a single start at 0 (the caller can zero-pad).
    """
    if num_samples <= 0:
        return [0]

    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_samples and hop_samples must be positive")

    if num_samples <= window_samples:
        return [0]

    count = 1 + int(math.ceil((num_samples - window_samples) / float(hop_samples)))
    max_start = num_samples - window_samples

    starts: List[int] = []
    for index in range(count):
        start = min(index * hop_samples, max_start)
        starts.append(start)

    # Guard against accidental duplicates due to min clipping.
    deduped = []
    last = None
    for value in starts:
        if value != last:
            deduped.append(value)
        last = value
    return deduped
