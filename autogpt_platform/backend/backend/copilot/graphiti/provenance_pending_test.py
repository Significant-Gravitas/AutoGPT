"""Unit tests for ``provenance_pending``: the graphs noted as holding a dream
citation marker whose record failed, and the reaper's sweep over them, each
reconciled under its write lock and forgotten only once nothing is left to
do there. The live run is ``recall_provenance_integration_test.py``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import provenance_pending, recall_reconcile
from .recall_fake_redis import FakeRedis
from .scope import write_lock_key


def _opened() -> MagicMock:
    """``open_graph_driver``, handing out a driver that closes."""
    return MagicMock(return_value=MagicMock(close=AsyncMock()))


@pytest.fixture
def pending_redis(mocker) -> FakeRedis:
    redis = FakeRedis()
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=redis),
    )
    return redis


@pytest.mark.asyncio
async def test_notes_a_graph_and_never_raises(pending_redis: FakeRedis, mocker) -> None:
    await provenance_pending.note_pending("user_a")
    assert pending_redis.sets[provenance_pending.PENDING_KEY] == {"user_a"}

    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(side_effect=ConnectionError("redis down")),
    )
    await provenance_pending.note_pending("user_b")


@pytest.mark.asyncio
async def test_reconciles_each_graph_under_its_lock_and_forgets_the_done(
    pending_redis: FakeRedis, lock_redis: FakeRedis
) -> None:
    """A graph with a landed marker left, one that could not be settled, or
    a write that could still land stays noted."""
    pending_redis.sets[provenance_pending.PENDING_KEY] = {
        "user_done",
        "user_left",
        "user_unsettled",
        "user_waiting",
    }
    held: list[bool] = []
    outcomes = {
        "user_done": recall_reconcile.Reconciled(completed=2, expired=1),
        "user_left": recall_reconcile.Reconciled(completed=1, left=True),
        "user_unsettled": recall_reconcile.Reconciled(unsettled=1),
        "user_waiting": recall_reconcile.Reconciled(waiting=1),
    }

    async def reconciled(driver, group_id: str):
        held.append(write_lock_key(group_id) in lock_redis.values)
        return outcomes[group_id]

    with (
        patch.object(provenance_pending, "reconcile", reconciled),
        patch.object(provenance_pending, "open_graph_driver", _opened()),
    ):
        swept = await provenance_pending.sweep_pending()

    assert swept == provenance_pending.Swept(graphs=4, completed=3)
    assert held == [True, True, True, True]
    assert pending_redis.sets[provenance_pending.PENDING_KEY] == {
        "user_left",
        "user_unsettled",
        "user_waiting",
    }


@pytest.mark.asyncio
async def test_a_busy_or_failing_graph_is_kept_for_the_next_run(
    pending_redis: FakeRedis, lock_redis: FakeRedis
) -> None:
    pending_redis.sets[provenance_pending.PENDING_KEY] = {"user_busy", "user_bad"}
    lock_redis.values[write_lock_key("user_busy")] = "an ingestion's token"

    with (
        patch.object(provenance_pending, "SWEEP_LOCK_WAIT_SECONDS", 0),
        patch.object(
            provenance_pending,
            "reconcile",
            AsyncMock(side_effect=RuntimeError("down")),
        ),
        patch.object(provenance_pending, "open_graph_driver", _opened()),
    ):
        swept = await provenance_pending.sweep_pending()

    assert (swept.busy, swept.failed, swept.graphs) == (1, 1, 0)
    assert pending_redis.sets[provenance_pending.PENDING_KEY] == {
        "user_busy",
        "user_bad",
    }


@pytest.mark.asyncio
async def test_without_redis_it_sweeps_nothing(mocker) -> None:
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(side_effect=ConnectionError("redis down")),
    )

    assert await provenance_pending.sweep_pending() == provenance_pending.Swept()
