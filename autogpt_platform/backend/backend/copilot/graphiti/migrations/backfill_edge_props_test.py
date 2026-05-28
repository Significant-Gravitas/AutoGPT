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
async def test_backfill_one_user_propagates_driver_exception(_stub_driver):
    """Unexpected driver failures must bubble up — silent suppression
    would let auth/config regressions hide as 'no edges to backfill',
    which the migration's success-print would then advertise."""
    _stub_driver.execute_query = AsyncMock(side_effect=RuntimeError("auth failed"))
    with pytest.raises(RuntimeError, match="auth failed"):
        await mig.backfill_one_user("9aa20a1c-805e-4128-8bb2-c27515140264")
    _stub_driver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_one_user_invalid_user_id_short_circuits(_stub_driver):
    """``derive_group_id`` raises ValueError on garbage input — migration
    must log and return 0 rather than attempting a query."""
    assert await mig.backfill_one_user("") == 0
    _stub_driver.execute_query.assert_not_called()


def _fake_user(user_id: str):
    """Minimal stand-in for a prisma User row — only ``id`` is read."""

    class _U:
        id = user_id

    return _U()


@pytest.mark.asyncio
async def test_backfill_all_users_pages_with_cursor(mocker, _stub_driver):
    """``backfill_all_users`` must paginate via keyset cursor on ``id``
    so memory stays bounded as the user table grows.

    We force a tiny batch size, return two full pages then a short
    final page, and assert that each ``find_many`` call carries the
    correct ``id > last_seen_id`` filter and ordering.
    """
    mocker.patch.object(mig, "USER_BATCH_SIZE", 2)

    # Cypher always reports 1 edge updated so we can count calls.
    _stub_driver.execute_query = AsyncMock(return_value=([{"updated": 1}], None, None))

    page_1 = [
        _fake_user("11111111-1111-1111-1111-111111111111"),
        _fake_user("22222222-2222-2222-2222-222222222222"),
    ]
    page_2 = [
        _fake_user("33333333-3333-3333-3333-333333333333"),
        _fake_user("44444444-4444-4444-4444-444444444444"),
    ]
    # Final page is shorter than the batch size — loop must terminate
    # without an extra empty fetch.
    page_3 = [_fake_user("55555555-5555-5555-5555-555555555555")]

    find_many = AsyncMock(side_effect=[page_1, page_2, page_3])
    fake_db = mocker.MagicMock()
    fake_db.connect = AsyncMock(return_value=None)
    fake_db.disconnect = AsyncMock(return_value=None)
    fake_db.user.find_many = find_many

    mocker.patch("prisma.Prisma", return_value=fake_db)

    users, edges = await mig.backfill_all_users()

    assert users == 5
    assert edges == 5
    assert find_many.await_count == 3

    # First call: no cursor.
    first_call_kwargs = find_many.await_args_list[0].kwargs
    assert first_call_kwargs["where"] == {}
    assert first_call_kwargs["order"] == {"id": "asc"}
    assert first_call_kwargs["take"] == 2

    # Subsequent calls: cursor advanced to last id of the previous page.
    assert find_many.await_args_list[1].kwargs["where"] == {
        "id": {"gt": "22222222-2222-2222-2222-222222222222"}
    }
    assert find_many.await_args_list[2].kwargs["where"] == {
        "id": {"gt": "44444444-4444-4444-4444-444444444444"}
    }


@pytest.mark.asyncio
async def test_backfill_all_users_empty_table_terminates(mocker, _stub_driver):
    """Empty User table — exactly one page-1 fetch, then exit."""
    find_many = AsyncMock(return_value=[])
    fake_db = mocker.MagicMock()
    fake_db.connect = AsyncMock(return_value=None)
    fake_db.disconnect = AsyncMock(return_value=None)
    fake_db.user.find_many = find_many

    mocker.patch("prisma.Prisma", return_value=fake_db)

    users, edges = await mig.backfill_all_users()

    assert (users, edges) == (0, 0)
    assert find_many.await_count == 1


@pytest.mark.asyncio
async def test_backfill_all_users_stops_on_short_page(mocker, _stub_driver):
    """If the first page is shorter than the batch size, the loop must
    exit after that page — no wasted empty fetch."""
    mocker.patch.object(mig, "USER_BATCH_SIZE", 100)

    _stub_driver.execute_query = AsyncMock(return_value=([{"updated": 0}], None, None))

    page = [
        _fake_user("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        _fake_user("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
    ]
    find_many = AsyncMock(return_value=page)
    fake_db = mocker.MagicMock()
    fake_db.connect = AsyncMock(return_value=None)
    fake_db.disconnect = AsyncMock(return_value=None)
    fake_db.user.find_many = find_many

    mocker.patch("prisma.Prisma", return_value=fake_db)

    users, edges = await mig.backfill_all_users()
    assert users == 2
    assert edges == 0
    assert find_many.await_count == 1
