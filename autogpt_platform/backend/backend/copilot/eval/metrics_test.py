from __future__ import annotations

import pytest

from .metrics import latency_summary, mean, percentile


def test_percentile_empty_returns_zero():
    assert percentile([], 50) == 0.0


@pytest.mark.parametrize(
    "p,expected",
    [
        (0, 1.0),
        (50, 3.0),
        (100, 5.0),
        # 95th of [1,2,3,4,5]: rank = 0.95 * 4 = 3.8 → 4 + 0.8 * (5-4) = 4.8
        (95, 4.8),
    ],
)
def test_percentile_linear_interpolation(p: float, expected: float):
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], p) == pytest.approx(expected)


def test_percentile_unsorted_input_is_sorted_internally():
    """Caller can pass values in any order."""
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 50) == pytest.approx(3.0)


def test_mean_empty():
    assert mean([]) == 0.0


def test_mean_basic():
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_latency_summary_empty_yields_zeroed_block():
    """A freshly-seeded eval that produced zero passes shouldn't crash —
    every numeric field reads 0.0 / 0."""
    s = latency_summary([])
    assert s == {
        "n": 0,
        "mean_seconds": 0.0,
        "p50_seconds": 0.0,
        "p95_seconds": 0.0,
        "min_seconds": 0.0,
        "max_seconds": 0.0,
    }


def test_latency_summary_realistic_block():
    """Spot-check the aggregate fields against a known input."""
    s = latency_summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["n"] == 5
    assert s["mean_seconds"] == pytest.approx(22.0)
    assert s["p50_seconds"] == pytest.approx(3.0)
    # 95th of [1,2,3,4,100]: rank=3.8 → 4 + 0.8*(100-4) = 80.8
    assert s["p95_seconds"] == pytest.approx(80.8)
    assert s["min_seconds"] == 1.0
    assert s["max_seconds"] == 100.0
