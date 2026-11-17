"""Unit tests for the derivation backfill: what it would write on a dry run
and writes with ``--apply``, under the graph's write lock, and its command
line. How it reads a description is in ``legacy_citations_test.py``, its
cascade in ``backfill_cascade_test.py``; the live run is in
``backfill_derivations_integration_test.py``.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.graphiti.recall_fake_redis import FakeRedis
from backend.copilot.graphiti.scope import write_lock_key

from . import backfill_derivations as backfill
from . import backfill_pages
from .backfill_fake import BackfillGraph, older_dreams

_LOCK = write_lock_key("user_a")


class TestBackfillGraph:
    @pytest.mark.asyncio
    async def test_a_dry_run_counts_and_writes_nothing(self) -> None:
        driver = older_dreams()

        found = await backfill.backfill_graph(
            driver, apply=False, cascade_forgets=False
        )

        assert (found.episodes, found.facts, found.unattributed) == (3, 3, 1)
        assert driver.writes == []

    @pytest.mark.asyncio
    async def test_apply_records_the_episodes_then_stamps_the_facts(self) -> None:
        """A fact a user's own turn also states is left unstamped; one whose
        dream episodes cite nothing is stamped empty and reported."""
        driver = older_dreams()

        await backfill.backfill_graph(driver, apply=True, cascade_forgets=False)

        assert driver.reconciled, "pending dream records are completed first"
        (records_query, records), (stamps_query, stamps) = driver.writes
        assert records_query == backfill.RECORD_EPISODES_QUERY
        assert records == [
            {"uuid": "d1", "facts": [], "episodes": ["e0", "e9"]},
            {"uuid": "d2", "facts": ["f1"], "episodes": []},
            {"uuid": "d4", "facts": [], "episodes": []},
        ]
        assert stamps_query == backfill.STAMP_FACTS_QUERY
        assert stamps == [
            {"uuid": "c1", "facts": [], "episodes": ["e0", "e9"]},
            {"uuid": "c2", "facts": ["f1", "f2"], "episodes": []},
            {"uuid": "c4", "facts": [], "episodes": []},
        ]

    @pytest.mark.asyncio
    async def test_reads_and_writes_in_batches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(backfill_pages, "BATCH_SIZE", 2)
        driver = older_dreams()

        await backfill.backfill_graph(driver, apply=True, cascade_forgets=False)

        assert [len(rows) for _, rows in driver.writes] == [2, 1, 2, 1]

    def test_a_record_already_written_is_never_replaced(self) -> None:
        assert "WHERE ep.derived_from_facts IS NULL" in backfill.RECORD_EPISODES_QUERY
        assert "WHERE e.derived_from_facts IS NULL" in backfill.STAMP_FACTS_QUERY
        assert "SET" not in backfill.DREAM_EPISODES_QUERY
        assert "SET" not in backfill.UNSTAMPED_FACTS_QUERY


class TestTheWriteLock:
    @pytest.mark.asyncio
    async def test_apply_holds_the_graphs_write_lock_throughout(
        self, lock_redis: FakeRedis
    ) -> None:
        driver = older_dreams()
        held: list[bool] = []
        query = driver.execute_query

        async def locked(cypher: str, **params: Any):
            held.append(_LOCK in lock_redis.values)
            return await query(cypher, **params)

        with patch.object(driver, "execute_query", locked):
            await backfill.backfill_graph(driver, apply=True, cascade_forgets=False)

        assert held and all(held)
        assert _LOCK not in lock_redis.values, "released afterwards"

    @pytest.mark.asyncio
    async def test_a_graph_another_writer_keeps_locked_is_skipped_as_busy(
        self, lock_redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock_redis.values[_LOCK] = "an ingestion's token"
        monkeypatch.setattr(backfill, "BACKFILL_LOCK_WAIT_SECONDS", 0)
        driver = older_dreams()

        found = await backfill.backfill_graph(driver, apply=True, cascade_forgets=True)

        assert (found.busy, found.episodes, driver.writes) == (1, 0, [])


class TestBackfillAllGraphs:
    @pytest.mark.asyncio
    async def test_walks_every_memory_graph_and_counts_a_failing_one(self) -> None:
        broken = BackfillGraph([], [])
        broken.answers = {}
        drivers = {
            "user_a": older_dreams(),
            "expert_b": older_dreams(),
            "user_broken": broken,
        }

        with (
            patch.object(
                backfill,
                "list_graph_names",
                AsyncMock(return_value=[*drivers, "other", "default_db"]),
            ),
            patch.object(
                backfill, "open_graph_driver", side_effect=drivers.__getitem__
            ),
        ):
            totals = await backfill.backfill_all_graphs(
                apply=False, cascade_forgets=False
            )

        assert (totals.episodes, totals.facts, totals.failed) == (6, 6, 1)
        for driver in drivers.values():
            driver.close.assert_awaited_once()


class TestMain:
    def test_the_command_line_is_a_dry_run_unless_told_otherwise(self) -> None:
        args = backfill.parser().parse_args([])
        assert (args.apply, args.graph, args.cascade_existing_forgets) == (
            False,
            None,
            False,
        )
        args = backfill.parser().parse_args(
            ["--apply", "--graph", "user_a", "--cascade-existing-forgets"]
        )
        assert (args.apply, args.graph, args.cascade_existing_forgets) == (
            True,
            "user_a",
            True,
        )

    @pytest.mark.parametrize(
        "totals, code",
        [
            (backfill.Derivations(facts=2), 0),
            (backfill.Derivations(facts=2, busy=1), 1),
            (backfill.Derivations(failed=1), 1),
        ],
        ids=["all done", "a graph busy", "a graph failed"],
    )
    @pytest.mark.asyncio
    async def test_exits_non_zero_while_any_graph_is_left_to_run_again(
        self,
        totals: backfill.Derivations,
        code: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        args = backfill.parser().parse_args(["--apply", "--cascade-existing-forgets"])
        with patch.object(
            backfill, "backfill_all_graphs", AsyncMock(return_value=totals)
        ):
            assert await backfill.main(args) == code

        out = capsys.readouterr().out
        assert ("run again" in out) is bool(code)
        assert "cascade from" in out
