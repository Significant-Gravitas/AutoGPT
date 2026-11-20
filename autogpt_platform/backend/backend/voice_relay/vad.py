"""Energy voice-activity detection over 16-bit mono PCM.

The relay hears the person through this rather than through the vendor, so
the idle cutoff works the same on every leg and whatever the vendor does, and
the post-call job cuts the recording into the person's speech with it. A
10 ms window counts as speech above -45 dBFS RMS, and a frame counts as heard
when at least three such windows (30 ms) are in it, which ignores clicks.
"""

import numpy as np

from backend.voice_relay.config import IN_RATE

WINDOW = IN_RATE // 100
THRESHOLD = 32768 * 10 ** (-45 / 20)
MIN_WINDOWS = 3


def window_rms(pcm: bytes) -> np.ndarray:
    """The RMS level of each whole 10 ms window in *pcm*."""
    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16)
    windows = len(samples) // WINDOW
    if windows == 0:
        return np.zeros(0, dtype=np.float32)
    framed = samples[: windows * WINDOW].astype(np.float32).reshape(windows, WINDOW)
    return np.sqrt(np.mean(framed**2, axis=1))


def heard(pcm: bytes) -> bool:
    """Whether *pcm* holds speech, not silence or a click."""
    return int(np.count_nonzero(window_rms(pcm) > THRESHOLD)) >= MIN_WINDOWS


def segments(
    pcm: bytes,
    *,
    merge_gap_s: float = 0.5,
    min_s: float = 0.15,
    pad_s: float = 0.2,
) -> list[tuple[int, int]]:
    """The speech in *pcm*, as ``(start, end)`` sample offsets.

    Speech windows closer than *merge_gap_s* join into one segment; a segment
    shorter than *min_s* is a click and is dropped; each is padded by *pad_s*
    on both sides, within the recording.
    """
    speech = np.flatnonzero(window_rms(pcm) > THRESHOLD)
    window_s = WINDOW / IN_RATE
    regions: list[list[int]] = []
    for index in speech.tolist():
        if regions and (index - regions[-1][1]) * window_s <= merge_gap_s:
            regions[-1][1] = index + 1
        else:
            regions.append([index, index + 1])
    total = len(pcm) // 2
    pad = int(pad_s * IN_RATE)
    return [
        (max(0, start * WINDOW - pad), min(total, end * WINDOW + pad))
        for start, end in regions
        if (end - start) * window_s >= min_s
    ]
