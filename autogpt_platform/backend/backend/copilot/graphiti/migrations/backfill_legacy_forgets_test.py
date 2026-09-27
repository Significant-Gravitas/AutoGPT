"""Unit tests for the legacy-forget backfill: the Cypher it sends, in what
order, and under the graph's write lock. It is a one-off migration, so
nothing here runs it on a graph; the live run against an ingestion under way
is in ``recall_inflight_integration_test.py``.
"""

import argparse
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.graphiti import scope_lock
from backend.copilot.graphiti.recall import legacy_forget_predicate
from backend.copilot.graphiti.recall_fake_redis import FakeRedis
from backend.copilot.graphiti.recall_hide import REDACT_EPISODES_QUERY
from backend.copilot.graphiti.scope import write_lock_key

from . import backfill_legacy_forgets as backfill

_LOCK = write_lock_key("user_a")


def _driver(*results: list[dict]) -> AsyncMock:
    driver = AsyncMock()
    driver.graph_name = "user_a"
    driver.execute_query.side_effect = [(rows, [], None) for rows in results]
    return driver


class TestBackfillGraph:
    @pytest.mark.asyncio
    async def test_a_dry_run_only_counts(self) -> None:
        driver = _driver([{"edges": 3, "episodes": 2}])

        found = await backfill.backfill_graph(driver, apply=False)

        assert (found.edges, found.episodes) == (3, 2)
        assert driver.execute_query.await_count == 1
        assert "SET" not in driver.execute_query.await_args.args[0]

    @pytest.mark.asyncio
    async def test_apply_hides_like_a_forget_before_restamping_edges(self) -> None:
        """The forget's own scrub and redaction run on the edges the count
        found; the restamp goes last, since a restamped edge no longer has
        the legacy shape and a re-run could not find it again."""
        driver = _driver([{"uuids": ["e1", "e2"], "edges": 2, "episodes": 1}], [], [])
        scrub = AsyncMock()

        with patch.object(backfill, "scrub", scrub):
            await backfill.backfill_graph(driver, apply=True)

        scrub.assert_awaited_once_with(driver, ["e1", "e2"])
        count, redact, restamp = driver.execute_query.await_args_list
        assert count.args[0] == backfill.COUNT_QUERY
        assert redact.args[0] == REDACT_EPISODES_QUERY
        assert redact.kwargs["uuids"] == ["e1", "e2"]
        assert set(redact.kwargs) == {"uuids", "now"}
        assert restamp.args[0] == backfill.RETRACT_EDGES_QUERY
        assert restamp.kwargs == {
            "uuids": ["e1", "e2"],
            "status": "retracted",
            "reason": "user_signal",
        }

    def test_the_restamp_gives_a_legacy_forget_the_forget_marker(self) -> None:
        """``forgotten_at`` is the old forget's ``expired_at``, and a marker a
        forget already wrote is kept."""
        query = backfill.RETRACT_EDGES_QUERY
        assert "e.forgotten_at = coalesce(e.forgotten_at, e.expired_at)" in query
        assert "e.status = $status" in query
        assert "e.expiration_reason = $reason" in query

    @pytest.mark.asyncio
    async def test_apply_writes_nothing_where_there_is_nothing_to_restamp(
        self,
    ) -> None:
        driver = _driver([{"edges": 0, "episodes": 0}])

        await backfill.backfill_graph(driver, apply=True)

        assert driver.execute_query.await_count == 1

    def test_the_count_finds_edges_by_the_policys_legacy_clause(self) -> None:
        """The writes touch only the edges it found, and the restamp only
        those still in the legacy shape."""
        assert f"WHERE {legacy_forget_predicate('e')}" in backfill.COUNT_QUERY
        assert "legacy AS uuids" in backfill.COUNT_QUERY
        assert "ep.redacted_at IS NULL" in backfill.COUNT_QUERY
        assert "SET" not in backfill.COUNT_QUERY
        assert (
            f"WHERE e.uuid IN $uuids AND {legacy_forget_predicate('e')}"
            in backfill.RETRACT_EDGES_QUERY
        )


class TestBackfillAllGraphs:
    @pytest.mark.asyncio
    async def test_walks_every_memory_graph_and_skips_a_failing_one(self) -> None:
        lister = AsyncMock()
        lister.client.list_graphs = AsyncMock(
            return_value=["user_a", "expert_b", "user_broken", "other", "default_db"]
        )
        broken = AsyncMock()
        broken.execute_query.side_effect = RuntimeError("down")
        drivers = {
            "default_db": lister,
            "user_a": _driver([{"edges": 1, "episodes": 2}]),
            "expert_b": _driver([{"edges": 2, "episodes": 0}]),
            "user_broken": broken,
        }

        with patch.object(backfill, "_graph_driver", side_effect=drivers.__getitem__):
            totals = await backfill.backfill_all_graphs(apply=False)

        assert (totals.edges, totals.episodes, totals.failed) == (3, 2, 1)
        for driver in drivers.values():
            driver.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_one_named_graph_is_all_it_touches(self) -> None:
        driver = _driver([{"edges": 0, "episodes": 0}])
        opened: list[str] = []

        def open_graph(name: str) -> AsyncMock:
            opened.append(name)
            return driver

        with patch.object(backfill, "_graph_driver", side_effect=open_graph):
            await backfill.backfill_all_graphs(apply=True, graph="expert_x")

        assert opened == ["expert_x"]


class TestTheWriteLock:
    """Reproduced first by an independent validation
    (``r6-migration-concurrency.py``): without the lock, an ingestion paused
    at graphiti's save wrote its older copy over a finished backfill."""

    @pytest.mark.asyncio
    async def test_apply_holds_the_graphs_write_lock_from_count_to_restamp(
        self, lock_redis: FakeRedis
    ) -> None:
        held: list[bool] = []
        driver = _driver()
        answers = iter([[{"uuids": ["e1"], "edges": 1, "episodes": 1}], [], []])

        async def query(*args: object, **kwargs: object):
            held.append(_LOCK in lock_redis.values)
            return next(answers), [], None

        driver.execute_query.side_effect = query
        with patch.object(backfill, "scrub", AsyncMock()):
            found = await backfill.backfill_graph(driver, apply=True)

        assert (found.edges, found.busy) == (1, 0)
        assert held == [True, True, True], "count, redaction and restamp"
        assert _LOCK not in lock_redis.values, "released afterwards"

    @pytest.mark.asyncio
    async def test_a_graph_another_writer_keeps_locked_is_skipped_as_busy(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock_redis.values[_LOCK] = "an ingestion's token"
        monkeypatch.setattr(backfill, "BACKFILL_LOCK_WAIT_SECONDS", 0)
        driver = _driver()

        found = await backfill.backfill_graph(driver, apply=True)

        assert (found.edges, found.busy) == (0, 1)
        driver.execute_query.assert_not_awaited()
        assert lock_redis.values[_LOCK] == "an ingestion's token"

    @pytest.mark.asyncio
    async def test_without_redis_a_graph_is_skipped_not_written_unlocked(
        self,
    ) -> None:
        driver = _driver()
        down = AsyncMock(side_effect=ConnectionError("redis down"))

        with patch.object(scope_lock, "get_redis_async", down):
            found = await backfill.backfill_graph(driver, apply=True)

        assert found.busy == 1
        driver.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_dry_run_only_reads_so_takes_no_lock(
        self, lock_redis: FakeRedis
    ) -> None:
        lock_redis.values[_LOCK] = "an ingestion's token"

        found = await backfill.backfill_graph(
            _driver([{"edges": 2, "episodes": 1}]), apply=False
        )

        assert (found.edges, found.busy) == (2, 0)


class TestMain:
    @pytest.mark.parametrize(
        "totals, code",
        [
            (backfill.LegacyForgets(edges=2), 0),
            (backfill.LegacyForgets(edges=2, busy=1), 1),
            (backfill.LegacyForgets(failed=1), 1),
        ],
        ids=["all done", "a graph busy", "a graph failed"],
    )
    @pytest.mark.asyncio
    async def test_exits_non_zero_while_any_graph_is_left_to_run_again(
        self,
        totals: backfill.LegacyForgets,
        code: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        args = argparse.Namespace(apply=True, graph=None)
        with patch.object(
            backfill, "backfill_all_graphs", AsyncMock(return_value=totals)
        ):
            assert await backfill.main(args) == code

        assert ("run again" in capsys.readouterr().out) is bool(code)
