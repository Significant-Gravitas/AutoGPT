"""Tests for the backfill migration script — Cypher boundary mocked.

The migration walks every User row and runs one Cypher per group.
These tests pin its three branches: success, no-graph (FalkorDB raises
on a missing database), and invalid-group-id.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from . import backfill_edge_props as mig


@pytest.fixture(autouse=True)
def _stub_driver(mocker):
    """Replace AutoGPTFalkorDriver with a MagicMock that returns canned
    execute_query results."""
    driver = mocker.MagicMock()
    driver.close = AsyncMock(return_value=None)
    mocker.patch.object(
        mig, "AutoGPTFalkorDriver", mocker.MagicMock(return_value=driver)
    )
    return driver


@pytest.mark.asyncio
async def test_backfill_one_user_counts_updated(_stub_driver):
    """Happy path — Cypher returns one row with the updated count."""
    _stub_driver.execute_query = AsyncMock(return_value=([{"updated": 7}], None, None))
    updated = await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264")
    assert updated == 7
    _stub_driver.execute_query.assert_awaited_once_with(mig.BACKFILL_QUERY)
    _stub_driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_one_user_no_records_returns_zero(_stub_driver):
    """Empty result set (no edges needed backfilling)."""
    _stub_driver.execute_query = AsyncMock(return_value=([], None, None))
    assert await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264") == 0


@pytest.mark.asyncio
async def test_backfill_one_user_none_result_returns_zero(_stub_driver):
    """Driver returns None when the database doesn't exist yet.

    Pyright fix path — the migration handles this gracefully so it
    doesn't crash on freshly-signed-up users with no Graphiti graph.
    """
    _stub_driver.execute_query = AsyncMock(return_value=None)
    assert await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264") == 0


@pytest.mark.asyncio
async def test_backfill_one_user_swallows_driver_exception(_stub_driver, caplog):
    """Missing-graph errors are logged at debug and treated as no-op."""
    _stub_driver.execute_query = AsyncMock(side_effect=Exception("no such db"))
    assert await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264") == 0


@pytest.mark.asyncio
async def test_backfill_one_user_invalid_user_id_short_circuits(_stub_driver):
    """``derive_group_id`` raises ValueError on garbage input — migration
    must log and return 0 rather than attempting a query."""
    assert await mig.backfill_one_user("") == 0
    _stub_driver.execute_query.assert_not_called()
