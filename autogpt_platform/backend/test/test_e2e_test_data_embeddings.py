"""
Unit tests for the E2E seeder's embedding backfill phase.
Stubs the search-embedding functions to avoid needing a database or an LLM.
"""

from test import e2e_test_data
from unittest.mock import patch

import pytest


def _stats(missing: int, total: int = 5584):
    return {
        "totals": {
            "without_embeddings": missing,
            "total": total,
            "coverage_percent": round((total - missing) / total * 100, 1),
        }
    }


class _Recorder:
    """Serves a stats sequence and records the backfill calls it triggers."""

    def __init__(self, stats_sequence, success=1):
        self._stats = list(stats_sequence)
        self._success = success
        self.batch_sizes: list[int] = []

    async def get_stats(self):
        return self._stats.pop(0) if len(self._stats) > 1 else self._stats[0]

    async def backfill(self, batch_size):
        self.batch_sizes.append(batch_size)
        return {
            "totals": {
                "success": self._success,
                "message": f"Overall: {self._success} succeeded",
            }
        }


@pytest.fixture
def seeder():
    return e2e_test_data.TestDataCreator.__new__(e2e_test_data.TestDataCreator)


async def _run(seeder, recorder, *, client=object()):
    with (
        patch.object(e2e_test_data, "get_openai_client", return_value=client),
        patch.object(e2e_test_data, "get_embedding_stats", recorder.get_stats),
        patch.object(e2e_test_data, "backfill_all_content_types", recorder.backfill),
    ):
        await seeder.backfill_content_embeddings()


async def test_skips_when_no_embedding_backend(seeder):
    """A stack without an embedding backend must not attempt any API call."""
    recorder = _Recorder([_stats(5584)])
    await _run(seeder, recorder, client=None)
    assert recorder.batch_sizes == []


async def test_skips_when_already_covered(seeder):
    recorder = _Recorder([_stats(0)])
    await _run(seeder, recorder)
    assert recorder.batch_sizes == []


async def test_does_not_claim_coverage_when_stats_are_unreadable(seeder):
    """get_embedding_stats reports zero missing on failure; that is not 100%."""
    broken = dict(_stats(0, 5584), error="connection refused")
    recorder = _Recorder([broken])
    await _run(seeder, recorder)
    assert recorder.batch_sizes == []


async def test_backfills_until_coverage_is_complete(seeder):
    recorder = _Recorder([_stats(5584), _stats(2000), _stats(0)])
    await _run(seeder, recorder)
    assert recorder.batch_sizes == [
        e2e_test_data.EMBEDDING_BACKFILL_BATCH_SIZE,
        e2e_test_data.EMBEDDING_BACKFILL_BATCH_SIZE,
    ]


async def test_gives_up_when_the_backfill_stops_making_progress(seeder):
    """A backfill that only fails must not spin until the deadline."""
    recorder = _Recorder([_stats(500)], success=0)
    await _run(seeder, recorder)
    assert len(recorder.batch_sizes) == 1


async def test_returns_on_deadline_instead_of_blocking_the_dump(seeder, monkeypatch):
    monkeypatch.setattr(e2e_test_data, "EMBEDDING_BACKFILL_TIMEOUT_SECONDS", 0.0)
    recorder = _Recorder([_stats(500)])
    await _run(seeder, recorder)
    assert recorder.batch_sizes == []
