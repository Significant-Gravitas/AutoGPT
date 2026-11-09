"""Energy voice-activity detection over 16-bit mono PCM.

The relay hears the person through this rather than through the vendor, so
the idle cutoff works the same on every leg and whatever the vendor does. A
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
