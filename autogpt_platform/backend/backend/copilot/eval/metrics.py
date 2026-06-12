"""Pure stats functions for the dream-eval harness.

No I/O. Tests pin known-answer cases. Suites import these to roll up
per-run elapsed_seconds / per-pass token counts into the aggregate
fields written to ``results.json``.
"""

from __future__ import annotations

import math
from typing import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile, p in [0, 100].

    Returns 0.0 on empty input rather than raising. Used so a freshly
    seeded run that yields zero passes doesn't crash the report.
    """
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    sorted_vals = sorted(values)
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def latency_summary(elapsed_seconds: Sequence[float]) -> dict[str, float]:
    """Stat block matching the ``results.json`` schema in p0-spec §7."""
    return {
        "n": len(elapsed_seconds),
        "mean_seconds": mean(elapsed_seconds),
        "p50_seconds": percentile(elapsed_seconds, 50),
        "p95_seconds": percentile(elapsed_seconds, 95),
        "min_seconds": float(min(elapsed_seconds)) if elapsed_seconds else 0.0,
        "max_seconds": float(max(elapsed_seconds)) if elapsed_seconds else 0.0,
    }
