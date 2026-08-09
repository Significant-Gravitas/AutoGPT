"""Gather-step tests — the Cypher boundary is mocked; what's under test
is that the fetched rows carry what the dream pass needs downstream.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from . import fetch as fetch_mod
from .fetch import _fetch_active_facts, fetch_usage_rows


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
    assert "toString(e.last_recalled_at) AS last_recalled_at" in query
    assert "toString(e.prev_recalled_at) AS prev_recalled_at" in query


@pytest.mark.asyncio
async def test_facts_without_recall_props_read_as_never_recalled():
    """Edges written before the hit hook shipped have no recall props at
    all. They must parse as None (never recalled) rather than blowing up
    the gather — that's what makes the no-backfill story hold."""
    facts = await _fetch_active_facts(_driver_returning([_row()]), "g-1", 500)

    assert facts[0].recall_count is None
    assert facts[0].last_recalled_at is None


# ---------------------------------------------------------------------------
# fetch_usage_rows — the apply-time usage refresh
# ---------------------------------------------------------------------------


def _patch_driver(mocker, driver) -> None:
    driver.close = AsyncMock(return_value=None)
    mocker.patch.object(
        fetch_mod, "AutoGPTFalkorDriver", MagicMock(return_value=driver)
    )


@pytest.mark.asyncio
async def test_fetch_usage_rows_returns_current_stamps(mocker):
    """The apply-time refresh reads only usage props, keyed by uuid."""
    driver = _driver_returning(
        [
            {
                "uuid": "hot",
                "recall_count": 3,
                "last_recalled_at": "2026-08-05T00:00:00+00:00",
                "prev_recalled_at": "2026-08-03T00:00:00+00:00",
            },
            {"uuid": "cold", "recall_count": None, "last_recalled_at": None},
        ]
    )
    _patch_driver(mocker, driver)

    rows = await fetch_usage_rows("u-1234567890ab", ["hot", "cold"])

    assert rows is not None
    assert {(r.uuid, r.recall_count) for r in rows} == {("hot", 3), ("cold", None)}
    # prev_recalled_at is the field protection actually keys on — an
    # unmapped column here would silently unprotect every refreshed fact.
    by_uuid = {r.uuid: r for r in rows}
    assert by_uuid["hot"].prev_recalled_at == "2026-08-03T00:00:00+00:00"
    assert by_uuid["cold"].prev_recalled_at is None
    driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_usage_rows_returns_none_on_query_failure(mocker):
    """A failed refresh means "no fresh data", never an exception — the
    demotion guard falls back to the bundle snapshot."""
    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=RuntimeError("boom"))
    _patch_driver(mocker, driver)

    assert await fetch_usage_rows("u-1234567890ab", ["hot"]) is None
    driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_usage_rows_returns_none_for_an_invalid_user_id(mocker):
    """derive_group_id rejects malformed ids — fail open to the snapshot
    rather than raising into the apply path."""
    ctor = MagicMock()
    mocker.patch.object(fetch_mod, "AutoGPTFalkorDriver", ctor)
    mocker.patch.object(
        fetch_mod, "derive_group_id", side_effect=ValueError("bad user id")
    )

    assert await fetch_usage_rows("", ["hot"]) is None
    ctor.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_usage_rows_skips_the_driver_for_no_uuids(mocker):
    ctor = MagicMock()
    mocker.patch.object(fetch_mod, "AutoGPTFalkorDriver", ctor)

    assert await fetch_usage_rows("u-1234567890ab", []) == []
    ctor.assert_not_called()
