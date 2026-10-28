"""Loader for the historical-average preflight cost estimates.

The JSON file is a snapshot of average credits-per-execution per block_id,
populated by an admin pulling fresh aggregates from `/admin/blocks/cost-estimates`
and committing the result. Used by `block_usage_cost()` to give dynamic-cost
blocks (SECOND/ITEMS/COST_USD) a non-zero pre-flight charge so post-flight
reconciliation only settles a small delta — bounding the billing-leak surface.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

_ESTIMATES_PATH = Path(__file__).parent / "block_preflight_estimates.json"


class BlockPreflightEstimate(TypedDict):
    block_name: str
    cost_type: str
    samples: int
    mean_credits: int


_cache: dict[str, BlockPreflightEstimate] | None = None
_cache_mtime_ns: int | None = None


def _load() -> dict[str, BlockPreflightEstimate]:
    """Return the estimates dict, refreshing if the JSON file has changed.

    The cache is keyed on the file's `mtime_ns` so a hot-swapped JSON (mounted
    config, manual edit on a long-running pod) is picked up on the next call
    without requiring a process restart. The fast path — file unchanged — is a
    single `stat()`.
    """
    global _cache, _cache_mtime_ns
    try:
        current_mtime = _ESTIMATES_PATH.stat().st_mtime_ns
    except OSError:
        # File missing or inaccessible — keep whatever's cached; if nothing
        # has ever loaded, return empty.
        return _cache if _cache is not None else {}

    if _cache is not None and _cache_mtime_ns == current_mtime:
        return _cache

    try:
        raw = json.loads(_ESTIMATES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception(
            "Failed to load %s; preflight estimates disabled",
            _ESTIMATES_PATH.name,
        )
        # Don't bump _cache_mtime_ns on a failed parse — otherwise a same-mtime
        # in-place fix (low-resolution FS, atomic-rename to a buffer with the
        # same timestamp) leaves the empty cache poisoned forever.
        _cache = {}
        return _cache

    if not isinstance(raw, dict):
        logger.error(
            "Invalid estimates JSON root in %s (expected object); "
            "preflight estimates disabled",
            _ESTIMATES_PATH.name,
        )
        _cache = {}
        return _cache

    estimates = raw.get("estimates", {}) or {}
    loaded: dict[str, BlockPreflightEstimate] = (
        estimates if isinstance(estimates, dict) else {}
    )

    _cache = loaded
    _cache_mtime_ns = current_mtime
    return loaded


def get_preflight_estimate(block_id: str) -> int:
    """Return the historical-average preflight cost for a block in credits.

    Returns 0 when no estimate is available (unseen block, low sample count,
    file missing, or malformed entry) so the caller falls back to the
    existing 0-preflight behaviour. Negative or non-numeric `mean_credits`
    are clamped to 0 — billing must never go negative on a corrupt entry.
    """
    entry = _load().get(block_id)
    if not entry:
        return 0
    raw_mean = entry.get("mean_credits", 0)
    try:
        mean = float(raw_mean)
    except (TypeError, ValueError):
        return 0
    # Python's `json.loads` accepts non-spec `NaN`/`Infinity`, so a corrupt
    # or hot-swapped JSON could land us with a non-finite mean — `int(round)`
    # raises on those. Clamp to 0 to keep the docstring's promise that no
    # JSON content can crash a billing call site.
    if not math.isfinite(mean):
        return 0
    return max(0, int(round(mean)))


def reset_cache() -> None:
    """Test-only: drop the in-memory cache so a freshly-written JSON re-loads."""
    global _cache, _cache_mtime_ns
    _cache = None
    _cache_mtime_ns = None
