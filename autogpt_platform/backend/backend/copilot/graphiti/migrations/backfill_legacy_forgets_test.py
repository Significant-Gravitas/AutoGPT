"""Unit tests for the legacy-forget backfill: the Cypher it sends and in
what order. It is a one-off migration, so nothing here runs it on a graph.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.graphiti.recall import legacy_forget_predicate

from . import backfill_legacy_forgets as backfill


def _driver(*results: list[dict]) -> AsyncMock:
    driver = AsyncMock()
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
    async def test_apply_redacts_episodes_before_restamping_edges(self) -> None:
        """A restamped edge no longer has the legacy shape, so its episodes
        must already be redacted by then."""
        driver = _driver([{"edges": 1, "episodes": 1}], [], [])

        await backfill.backfill_graph(driver, apply=True)

        count, redact, restamp = driver.execute_query.await_args_list
        assert count.args[0] == backfill.COUNT_QUERY
        assert "SET ep.redacted_at = coalesce(ep.redacted_at, $now)" in redact.args[0]
        assert set(redact.kwargs) == {"now"}
        assert "SET e.status = $status, e.expiration_reason = $reason" in (
            restamp.args[0]
        )
        assert restamp.kwargs == {"status": "retracted", "reason": "user_signal"}

    @pytest.mark.asyncio
    async def test_apply_writes_nothing_where_there_is_nothing_to_restamp(
        self,
    ) -> None:
        driver = _driver([{"edges": 0, "episodes": 0}])

        await backfill.backfill_graph(driver, apply=True)

        assert driver.execute_query.await_count == 1

    def test_every_query_finds_edges_by_the_policys_legacy_clause(self) -> None:
        for query in (
            backfill.COUNT_QUERY,
            backfill.REDACT_EPISODES_QUERY,
            backfill.RETRACT_EDGES_QUERY,
        ):
            assert f"WHERE {legacy_forget_predicate('e')}" in query
        assert "ep.redacted_at IS NULL" in backfill.COUNT_QUERY
        assert "SET" not in backfill.COUNT_QUERY


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

        assert (totals.edges, totals.episodes) == (3, 2)
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
