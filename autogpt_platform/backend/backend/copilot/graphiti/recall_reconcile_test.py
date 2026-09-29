"""Unit tests for ``recall_reconcile.reconcile``, which completes the dream
citation markers a graph still holds, and for the reaper's sweep over the
graphs noted as holding one (``provenance_pending.py``). The live run is
``recall_provenance_integration_test.py``.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import provenance_pending, recall_derivation, recall_reconcile
from .recall_fake_redis import FakeRedis
from .scope import write_lock_key

_NOW = datetime.now(timezone.utc)


def _marker(uuid: str, *, age: timedelta = timedelta(0)) -> dict:
    return {
        "uuid": uuid,
        "name": f"dream_p_{uuid}",
        "facts": ["f1"],
        "episodes": ["ep1"],
        "created_at": (_NOW - age).isoformat(),
    }


def _driver(*results) -> MagicMock:
    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=[(r, [], None) for r in results])
    return driver


def _queries(driver: MagicMock) -> list[str]:
    return [call.args[0] for call in driver.execute_query.await_args_list]


class TestReconcile:
    @pytest.mark.asyncio
    async def test_a_graph_with_no_marker_reads_once(self) -> None:
        driver = _driver([])

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled()
        assert _queries(driver) == [recall_reconcile.PENDING_MARKERS_QUERY]

    @pytest.mark.asyncio
    async def test_completes_a_marker_whose_write_landed(self) -> None:
        driver = _driver(
            [_marker("m1")],
            [{"uuid": "dream-ep"}],
            [{"uuid": "e1"}, {"uuid": "e2"}],
            [],
            [],
        )

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(completed=1)
        assert _queries(driver)[1:] == [
            recall_reconcile.RECORD_NAMED_EPISODES_QUERY,
            recall_reconcile.TOUCHED_FACTS_QUERY,
            recall_derivation.STAMP_FACTS_QUERY,
            recall_derivation.DROP_MARKER_QUERY,
        ]
        _, record, touched, stamp, drop = driver.execute_query.await_args_list
        assert record.kwargs == {
            "name": "dream_p_m1",
            "group_id": "user_a",
            "facts": ["f1"],
            "episodes": ["ep1"],
        }
        assert touched.kwargs == {"episodes": ["dream-ep"]}
        assert stamp.kwargs == {"uuids": ["e1", "e2"], "group_id": "user_a"}
        assert drop.kwargs == {"uuid": "m1"}

    @pytest.mark.asyncio
    async def test_a_write_that_touched_no_fact_is_completed_all_the_same(
        self,
    ) -> None:
        driver = _driver([_marker("m1")], [{"uuid": "dream-ep"}], [], [])

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done.completed == 1
        assert recall_derivation.STAMP_FACTS_QUERY not in _queries(driver)

    @pytest.mark.asyncio
    async def test_a_young_marker_whose_write_is_not_there_waits(self) -> None:
        driver = _driver([_marker("m1", age=timedelta(minutes=5))], [])

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(waiting=1)
        assert recall_derivation.DROP_MARKER_QUERY not in _queries(driver)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("created_at", ["long ago", None])
    async def test_an_old_marker_whose_write_never_landed_is_dropped(
        self, created_at: str | None
    ) -> None:
        old = _marker("m1", age=timedelta(hours=2))
        stale = {**old, "created_at": created_at} if created_at != "long ago" else old
        driver = _driver([stale], [], [])

        done = await recall_reconcile.reconcile(driver, "user_a")

        assert done == recall_reconcile.Reconciled(orphaned=1)
        assert _queries(driver)[-1] == recall_derivation.DROP_MARKER_QUERY

    @pytest.mark.asyncio
    async def test_more_markers_than_the_limit_are_left_for_the_next_call(
        self,
    ) -> None:
        driver = _driver([_marker("m1"), _marker("m2")], [{"uuid": "ep"}], [], [])

        done = await recall_reconcile.reconcile(driver, "user_a", limit=1)

        assert (done.completed, done.left) == (1, True)
        first = driver.execute_query.await_args_list[0]
        assert first.kwargs == {"limit": 2}


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


class TestTheSweep:
    @pytest.mark.asyncio
    async def test_notes_a_graph_and_never_raises(
        self, pending_redis: FakeRedis, mocker
    ) -> None:
        await provenance_pending.note_pending("user_a")
        assert pending_redis.sets[provenance_pending.PENDING_KEY] == {"user_a"}

        mocker.patch(
            "backend.data.redis_client.get_redis_async",
            AsyncMock(side_effect=ConnectionError("redis down")),
        )
        await provenance_pending.note_pending("user_b")

    @pytest.mark.asyncio
    async def test_reconciles_each_graph_under_its_lock_and_forgets_the_done(
        self, pending_redis: FakeRedis, lock_redis: FakeRedis
    ) -> None:
        pending_redis.sets[provenance_pending.PENDING_KEY] = {
            "user_done",
            "user_left",
            "user_waiting",
        }
        held: list[bool] = []
        outcomes = {
            "user_done": recall_reconcile.Reconciled(completed=2),
            "user_left": recall_reconcile.Reconciled(completed=1, left=True),
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

        assert swept == provenance_pending.Swept(graphs=3, completed=3)
        assert held == [True, True, True]
        assert pending_redis.sets[provenance_pending.PENDING_KEY] == {
            "user_left",
            "user_waiting",
        }

    @pytest.mark.asyncio
    async def test_a_busy_or_failing_graph_is_kept_for_the_next_run(
        self, pending_redis: FakeRedis, lock_redis: FakeRedis
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
    async def test_without_redis_it_sweeps_nothing(self, mocker) -> None:
        mocker.patch(
            "backend.data.redis_client.get_redis_async",
            AsyncMock(side_effect=ConnectionError("redis down")),
        )

        assert await provenance_pending.sweep_pending() == provenance_pending.Swept()
