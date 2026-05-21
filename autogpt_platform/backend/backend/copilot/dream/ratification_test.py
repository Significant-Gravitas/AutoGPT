"""Ratification pass tests — mock Graphiti driver + Redis at the
boundary, verify each tentative-edge outcome produces the right
Cypher / mark_edges_superseded call and the right counters.

These tests do NOT touch FalkorDB or a live Redis. ratification.py
opens an ``AutoGPTFalkorDriver`` for its own Cypher and calls
``mark_edges_superseded`` for demotions; both are patched at the
module surface so the dispatch logic can be exercised in isolation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from . import ratification as ratification_mod
from .ratification import (
    RATIFICATION_GRACE_PERIOD,
    RatificationResult,
    run_ratification_pass,
)


def _make_driver(records_for_list: list[dict], records_for_promote=None):
    """Build a MagicMock driver whose execute_query returns the rows we want.

    The ratification module issues two Cypher shapes:
      * the LIST query (status='tentative')   → return ``records_for_list``
      * the PROMOTE query (SET status=active) → return ``records_for_promote``

    We dispatch by inspecting the first arg of execute_query.
    """
    driver = MagicMock()
    driver.close = AsyncMock(return_value=None)

    async def fake_execute(query: str, **kwargs):
        if "status = 'tentative'" in query:
            return (list(records_for_list), None, None)
        if "ratified_at" in query:
            # Promote query — default to one-row response unless caller said otherwise.
            rows = (
                records_for_promote
                if records_for_promote is not None
                else [{"uuid": kwargs.get("uuid")}]
            )
            return (list(rows), None, None)
        return ([], None, None)

    driver.execute_query = AsyncMock(side_effect=fake_execute)
    return driver


@pytest.fixture(autouse=True)
def _patch_driver_constructor(mocker):
    """Default to a no-tentative-edges driver; individual tests override."""
    driver = _make_driver(records_for_list=[])
    mocker.patch.object(
        ratification_mod,
        "AutoGPTFalkorDriver",
        MagicMock(return_value=driver),
    )
    return driver


@pytest.fixture
def stub_mark_superseded(mocker):
    """Stub mark_edges_superseded so we can assert it was (or wasn't) called."""
    stub = AsyncMock(side_effect=lambda driver, uuids, **kw: (list(uuids), []))
    mocker.patch.object(ratification_mod, "mark_edges_superseded", stub)
    return stub


@pytest.fixture
def fake_redis(mocker):
    """Stub the Redis client used by the hit-count read path.

    Tests set ``fake_redis.hits[edge_uuid] = N`` to control the count
    a given edge sees. Default is zero.
    """

    class FakeRedis:
        def __init__(self):
            self.hits: dict[str, int] = {}

        async def get(self, key: str):
            # Keys are ``mem:hits:{user_id}:{edge_uuid}`` — split off the uuid.
            edge_uuid = key.rsplit(":", 1)[-1]
            value = self.hits.get(edge_uuid, 0)
            return str(value).encode() if value else None

        async def set(self, key: str, value, **kwargs):
            return True

        async def incr(self, key: str):
            edge_uuid = key.rsplit(":", 1)[-1]
            self.hits[edge_uuid] = self.hits.get(edge_uuid, 0) + 1
            return self.hits[edge_uuid]

    redis = FakeRedis()
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=redis),
    )
    return redis


@pytest.mark.asyncio
async def test_tentative_edge_with_hits_is_ratified_to_active(
    mocker, fake_redis, stub_mark_superseded
):
    """Hit count >= 1 → flip to active via the promote Cypher; no demotion."""
    edge = {"uuid": "edge-hot", "created_at": _hours_ago(2)}
    driver = _make_driver(records_for_list=[edge])
    mocker.patch.object(
        ratification_mod, "AutoGPTFalkorDriver", MagicMock(return_value=driver)
    )
    fake_redis.hits["edge-hot"] = 3

    result = await run_ratification_pass("u-ratified")

    assert isinstance(result, RatificationResult)
    assert result.ratified_count == 1
    assert result.superseded_count == 0
    assert result.examined_count == 1
    assert result.error is None
    # Promote query was issued
    promote_calls = [
        call
        for call in driver.execute_query.await_args_list
        if "ratified_at" in call.args[0]
    ]
    assert len(promote_calls) == 1
    assert promote_calls[0].kwargs == {"uuid": "edge-hot"}
    # And demotion was NOT called
    stub_mark_superseded.assert_not_awaited()


@pytest.mark.asyncio
async def test_tentative_edge_without_hits_past_grace_is_superseded_as_unratified(
    mocker, fake_redis, stub_mark_superseded
):
    """Zero hits, edge older than the grace window → mark_edges_superseded(reason='unratified')."""
    edge = {
        "uuid": "edge-stale",
        "created_at": _days_ago(RATIFICATION_GRACE_PERIOD.days + 2),
    }
    driver = _make_driver(records_for_list=[edge])
    mocker.patch.object(
        ratification_mod, "AutoGPTFalkorDriver", MagicMock(return_value=driver)
    )

    result = await run_ratification_pass("u-stale")

    assert result.superseded_count == 1
    assert result.ratified_count == 0
    stub_mark_superseded.assert_awaited_once()
    call = stub_mark_superseded.await_args
    assert call.args[1] == ["edge-stale"]
    assert call.kwargs["reason"] == "unratified"
    assert call.kwargs["new_status"] == "superseded"


@pytest.mark.asyncio
async def test_tentative_edge_within_grace_without_hits_is_untouched(
    mocker, fake_redis, stub_mark_superseded
):
    """Zero hits but still inside the grace window → no promote, no demote."""
    edge = {"uuid": "edge-young", "created_at": _hours_ago(6)}
    driver = _make_driver(records_for_list=[edge])
    mocker.patch.object(
        ratification_mod, "AutoGPTFalkorDriver", MagicMock(return_value=driver)
    )

    result = await run_ratification_pass("u-young")

    assert result.ratified_count == 0
    assert result.superseded_count == 0
    assert result.examined_count == 1
    stub_mark_superseded.assert_not_awaited()
    # No promote query was issued either
    promote_calls = [
        call
        for call in driver.execute_query.await_args_list
        if "ratified_at" in call.args[0]
    ]
    assert promote_calls == []


@pytest.mark.asyncio
async def test_already_active_edges_are_not_in_scope_of_the_listing_query(
    mocker, fake_redis, stub_mark_superseded
):
    """The list query filters ``status='tentative'`` so active edges are
    invisible to the pass — no examined, no ratified, no superseded."""
    # The default fixture returns zero tentative edges. An already-active
    # edge would not show up in the list query; we model that by leaving
    # the list empty here.
    driver = _make_driver(records_for_list=[])
    mocker.patch.object(
        ratification_mod, "AutoGPTFalkorDriver", MagicMock(return_value=driver)
    )

    result = await run_ratification_pass("u-only-active")

    assert result.examined_count == 0
    assert result.ratified_count == 0
    assert result.superseded_count == 0
    stub_mark_superseded.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_user_with_no_tentative_edges_returns_zero_counts_no_error(
    fake_redis, stub_mark_superseded
):
    """User with no tentative edges → counts all zero, no error, completed_at set."""
    result = await run_ratification_pass("u-empty")

    assert result.error is None
    assert result.examined_count == 0
    assert result.ratified_count == 0
    assert result.superseded_count == 0
    assert result.completed_at is not None
    stub_mark_superseded.assert_not_awaited()


@pytest.mark.asyncio
async def test_per_edge_failure_does_not_kill_the_rest_of_the_pass(
    mocker, fake_redis, stub_mark_superseded
):
    """A poison-pill edge raises mid-pass; the others still get processed,
    and the failure is captured in ``per_edge_errors`` rather than raised."""
    edges = [
        {"uuid": "edge-good-hot", "created_at": _hours_ago(2)},
        {"uuid": "edge-poison", "created_at": _hours_ago(2)},
        {"uuid": "edge-good-stale", "created_at": _days_ago(30)},
    ]
    driver = _make_driver(records_for_list=edges)
    mocker.patch.object(
        ratification_mod, "AutoGPTFalkorDriver", MagicMock(return_value=driver)
    )
    fake_redis.hits["edge-good-hot"] = 5
    # edge-poison: any hit-count read for this edge raises so the
    # per-edge try/except can catch it without breaking the others.
    original_get_hit_count = ratification_mod._get_hit_count

    async def hit_count_with_poison(user_id: str, edge_uuid: str) -> int:
        if edge_uuid == "edge-poison":
            raise RuntimeError("simulated redis explosion")
        return await original_get_hit_count(user_id, edge_uuid)

    mocker.patch.object(
        ratification_mod, "_get_hit_count", side_effect=hit_count_with_poison
    )

    result = await run_ratification_pass("u-mixed")

    assert result.examined_count == 3
    # The hot edge got ratified, the stale edge got superseded, the
    # poison edge contributed to per_edge_errors.
    assert result.ratified_count == 1
    assert result.superseded_count == 1
    assert len(result.per_edge_errors) == 1
    assert "edge-poison" in result.per_edge_errors[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hours_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=n)


def _days_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)
