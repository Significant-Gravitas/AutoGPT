"""Gather-step tests — the Cypher boundary is mocked; what's under test
is that the fetched rows carry what the dream pass needs downstream.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from .fetch import _fetch_active_facts


def _driver_returning(rows: list[dict]) -> MagicMock:
    driver = MagicMock()
    driver.execute_query = AsyncMock(return_value=(rows, None, None))
    return driver


def _row(**overrides) -> dict:
    row = {
        "uuid": "f-1",
        "source": "Nick",
        "target": "ProjectX",
        "name": "works_on",
        "fact": "Nick works on ProjectX",
        "scope": "project:x",
        "confidence": 0.8,
        "status": "active",
        "created_at": "2026-05-10T00:00:00+00:00",
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_active_facts_carry_recall_stamps():
    """The dream pass can only be usage-aware if the gather step reads
    the props warm-context retrieval writes."""
    driver = _driver_returning(
        [_row(recall_count=4, last_recalled_at="2026-08-01T00:00:00+00:00")]
    )

    facts = await _fetch_active_facts(driver, "g-1", 500)

    assert facts[0].recall_count == 4
    assert facts[0].last_recalled_at == "2026-08-01T00:00:00+00:00"
    query = driver.execute_query.await_args.args[0]
    assert "e.recall_count AS recall_count" in query
    assert "e.last_recalled_at AS last_recalled_at" in query


@pytest.mark.asyncio
async def test_facts_without_recall_props_read_as_never_recalled():
    """Edges written before the hit hook shipped have no recall props at
    all. They must parse as None (never recalled) rather than blowing up
    the gather — that's what makes the no-backfill story hold."""
    facts = await _fetch_active_facts(_driver_returning([_row()]), "g-1", 500)

    assert facts[0].recall_count is None
    assert facts[0].last_recalled_at is None
