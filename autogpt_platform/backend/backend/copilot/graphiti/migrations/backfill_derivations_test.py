"""Unit tests for the derivation backfill: how it reads a dream episode's
description, what it would write on a dry run and writes with ``--apply``,
under the graph's write lock, and its command line. The live run is in
``backfill_derivations_integration_test.py``.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.dream.citations import described_citations
from backend.copilot.graphiti.memory_model import MemoryForgetFailure
from backend.copilot.graphiti.recall_fake_redis import FakeRedis
from backend.copilot.graphiti.recall_reconcile import PENDING_MARKERS_QUERY
from backend.copilot.graphiti.scope import write_lock_key

from . import backfill_cascade
from . import backfill_derivations as backfill
from . import backfill_pages

_LOCK = write_lock_key("user_a")


class _Driver:
    """A graph as the backfill's reads see it, recording its writes."""

    graph_name = "user_a"

    def __init__(
        self,
        episodes: list[dict[str, Any]],
        facts: list[dict[str, Any]],
        forgotten: list[str] | None = None,
    ) -> None:
        self.answers = {
            backfill.DREAM_EPISODES_QUERY: episodes,
            backfill.UNSTAMPED_FACTS_QUERY: facts,
            backfill_cascade.FORGOTTEN_FACTS_QUERY: [
                {"uuid": u} for u in forgotten or []
            ],
            backfill_cascade.FACT_NAMES_QUERY: [],
            backfill_cascade.EPISODE_NAMES_QUERY: [],
        }
        self.writes: list[tuple[str, list[dict[str, Any]]]] = []
        # Whether it read the pending dream records (none here).
        self.reconciled = False
        self.close = AsyncMock()

    async def execute_query(self, query: str, **params: Any):
        if query == PENDING_MARKERS_QUERY:
            self.reconciled = True
            return [], [], None
        if query == backfill_cascade.MARKER_NAMES_QUERY:
            return [], [], None
        if query in self.answers:
            rows = [r for r in self.answers[query] if r["uuid"] > params["after"]]
            return rows[: params["limit"]], [], None
        self.writes.append((query, params["rows"]))
        return [], [], None


def _dream(uuid: str, description: str | None, **record: list[str]) -> dict:
    return {
        "uuid": uuid,
        "description": description,
        "facts": record.get("facts"),
        "episodes": record.get("episodes"),
    }


def _graph() -> _Driver:
    """Two dream episodes written before records (a consolidation listing
    episodes, a proposal listing facts), one recorded since, one listing
    nothing; and the facts they and a user's chat turn produced."""
    return _Driver(
        episodes=[
            _dream("d1", "dream-pass consolidation; src_episodes=e0,e9"),
            _dream("d2", "dream-pass proposal; rationale=r; src_facts=f1"),
            _dream("d3", "dream-pass consolidation", facts=["f2"], episodes=[]),
            _dream("d4", "dream-pass consolidation; src_episodes="),
        ],
        facts=[
            {"uuid": "c1", "episodes": ["d1"]},
            {"uuid": "c2", "episodes": ["d2", "d3"]},
            {"uuid": "c4", "episodes": ["d4"]},
            {"uuid": "merged", "episodes": ["d1", "chat"]},
            {"uuid": "user", "episodes": ["chat"]},
        ],
    )


class TestDescribedCitations:
    @pytest.mark.parametrize(
        "description, cited",
        [
            ("dream-pass consolidation; src_episodes=e1,e2", ([], ["e1", "e2"])),
            ("dream-pass proposal; rationale=r; src_facts=f1", (["f1"], [])),
            (
                "dream-pass proposal; src_episodes=e1; src_facts=f1,f2",
                (["f1", "f2"], ["e1"]),
            ),
            ("dream-pass consolidation; src_episodes=", ([], [])),
            ("dream-pass proposal", ([], [])),
            (None, ([], [])),
            (
                "dream-pass proposal; rationale=see src_facts=x; src_facts=f1",
                (["f1"], []),
            ),
        ],
        ids=[
            "consolidation",
            "proposal",
            "both kinds",
            "empty list",
            "nothing listed",
            "no description",
            "a rationale quoting a key",
        ],
    )
    def test_reads_the_uuids_each_kind_lists(
        self, description: str | None, cited: tuple[list[str], list[str]]
    ) -> None:
        assert described_citations(description) == cited


class TestBackfillGraph:
    @pytest.mark.asyncio
    async def test_a_dry_run_counts_and_writes_nothing(self) -> None:
        driver = _graph()

        found = await backfill.backfill_graph(
            driver, apply=False, cascade_forgets=False
        )

        assert (found.episodes, found.facts, found.unattributed) == (3, 3, 1)
        assert driver.writes == []

    @pytest.mark.asyncio
    async def test_apply_records_the_episodes_then_stamps_the_facts(self) -> None:
        """A fact a user's own turn also states is left unstamped; one whose
        dream episodes cite nothing is stamped empty and reported."""
        driver = _graph()

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
        driver = _graph()

        await backfill.backfill_graph(driver, apply=True, cascade_forgets=False)

        assert [len(rows) for _, rows in driver.writes] == [2, 1, 2, 1]

    def test_a_record_already_written_is_never_replaced(self) -> None:
        assert "WHERE ep.derived_from_facts IS NULL" in backfill.RECORD_EPISODES_QUERY
        assert "WHERE e.derived_from_facts IS NULL" in backfill.STAMP_FACTS_QUERY
        assert "SET" not in backfill.DREAM_EPISODES_QUERY
        assert "SET" not in backfill.UNSTAMPED_FACTS_QUERY

    @pytest.mark.asyncio
    async def test_a_dry_run_counts_the_forgets_it_would_cascade_from(self) -> None:
        driver = _Driver([], [], forgotten=["x1", "x2"])
        cascade = AsyncMock()

        with patch.object(backfill_cascade, "cascade", cascade):
            found = await backfill.backfill_graph(
                driver, apply=False, cascade_forgets=True
            )

        assert found.roots == 2
        cascade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_cascades_from_every_forget_after_stamping(self) -> None:
        driver = _Driver([], [], forgotten=["x1", "x2"])

        async def retract_two(
            driver, group_id, roots, now, result, *, erase, seeds
        ) -> None:
            assert (group_id, roots, erase, seeds) == (
                "user_a",
                ["x1", "x2"],
                False,
                {},
            )
            result.derived.extend(["d1", "d2"])

        with patch.object(
            backfill_cascade, "cascade", AsyncMock(side_effect=retract_two)
        ):
            found = await backfill.backfill_graph(
                driver, apply=True, cascade_forgets=True
            )

        assert (found.roots, found.derived, found.failed) == (2, 2, 0)

    @pytest.mark.asyncio
    async def test_a_cascade_that_stopped_short_fails_the_graph(self) -> None:
        driver = _Driver([], [], forgotten=["x1"])

        async def stop_short(driver, group_id, roots, now, result, **_: object):
            result.failures.append(MemoryForgetFailure.derived_left("x1"))

        with patch.object(
            backfill_cascade, "cascade", AsyncMock(side_effect=stop_short)
        ):
            found = await backfill.backfill_graph(
                driver, apply=True, cascade_forgets=True
            )

        assert found.failed == 1


class TestTheWriteLock:
    @pytest.mark.asyncio
    async def test_apply_holds_the_graphs_write_lock_throughout(
        self, lock_redis: FakeRedis
    ) -> None:
        driver = _graph()
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
        driver = _graph()

        found = await backfill.backfill_graph(driver, apply=True, cascade_forgets=True)

        assert (found.busy, found.episodes, driver.writes) == (1, 0, [])


class TestBackfillAllGraphs:
    @pytest.mark.asyncio
    async def test_walks_every_memory_graph_and_counts_a_failing_one(self) -> None:
        broken = _Driver([], [])
        broken.answers = {}
        drivers = {"user_a": _graph(), "expert_b": _graph(), "user_broken": broken}

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
