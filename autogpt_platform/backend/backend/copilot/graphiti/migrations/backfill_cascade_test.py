"""Unit tests for the derivation backfill's cascade from the forgets already
made (``backfill_cascade.py``): the roots it finds (forgotten facts, facts a
hard forget purged that something still names, hidden episodes a record
names) and the cascades it runs from them, hard ones first and erasing. The
live run is ``graphiti/recall_cascade_resume_integration_test.py``.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.graphiti.memory_model import MemoryForgetFailure

from . import backfill_cascade
from . import backfill_derivations as backfill
from .backfill_fake import BackfillGraph

_PREFIX = "derived_from_forgotten:"


class _Graph:
    """What the root reads see: each query's rows, paged by uuid where the
    query pages."""

    graph_name = "user_a"

    def __init__(self) -> None:
        self.paged: dict[str, list[dict[str, Any]]] = {
            backfill_cascade.FORGOTTEN_FACTS_QUERY: [
                {"uuid": "d2"},
                {"uuid": "f-soft"},
            ],
            backfill_cascade.FACT_NAMES_QUERY: [
                _named("d1", facts=["f-soft", "f-gone"], episodes=["chat", "ep-d2"]),
                _named("d2", reason=f"{_PREFIX}f-purged"),
                _named("d3", reason="stale_fact"),
            ],
            backfill_cascade.EPISODE_NAMES_QUERY: [
                _named("ep-dream", facts=["f-gone"], episodes=["said"]),
                _named("said", hidden_for=["f-purged"]),
            ],
        }
        self.markers = [_named("m1", facts=["f-pending"])]
        self.existing = {"f-soft"}
        self.hidden = [
            {"uuid": "said", "hard": True, "hidden_for": ["f-purged"]},
            {"uuid": "chat", "hard": False, "hidden_for": []},
            {"uuid": "ep-d2", "hard": False, "hidden_for": ["d2"]},
        ]

    async def execute_query(self, query: str, **params: Any):
        if query in self.paged:
            rows = [r for r in self.paged[query] if r["uuid"] > params["after"]]
            return rows[: params["limit"]], [], None
        if query == backfill_cascade.MARKER_NAMES_QUERY:
            return self.markers, [], None
        if query == backfill_cascade.EXISTING_FACTS_QUERY:
            found = [{"uuid": u} for u in params["uuids"] if u in self.existing]
            return found, [], None
        assert query == backfill_cascade.HIDDEN_EPISODES_QUERY
        return [r for r in self.hidden if r["uuid"] in params["uuids"]], [], None


def _named(
    uuid: str,
    *,
    facts: list[str] | None = None,
    episodes: list[str] | None = None,
    hidden_for: list[str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "facts": facts or [],
        "episodes": episodes or [],
        "hidden_for": hidden_for or [],
        "reason": reason,
    }


class TestFindRoots:
    @pytest.mark.asyncio
    async def test_finds_every_root_the_graphs_forgets_left(self) -> None:
        """A fact a user forgot, still there, is soft (``d2``, which a cascade
        retracted for ``f-purged``, is picked up from that root); a gone
        fact a record, a marker, a reason or an episode's ``redacted_for``
        names was purged by a hard forget; a hidden episode a record names
        is a seed, erased from when a hard forget emptied it, naming the
        root of what it was hidden for, else itself."""
        roots = await backfill_cascade.find_roots(_Graph())

        assert roots.soft == ["f-soft"]
        assert sorted(roots.hard) == ["f-gone", "f-pending", "f-purged"]
        assert roots.hard_seeds == {"said": "f-purged"}
        assert roots.soft_seeds == {"chat": "chat", "ep-d2": "f-purged"}
        assert roots.count() == 7

    @pytest.mark.asyncio
    async def test_a_graph_no_forget_touched_has_no_root(self) -> None:
        graph = _Graph()
        graph.paged = {query: [] for query in graph.paged}
        graph.markers = []

        roots = await backfill_cascade.find_roots(graph)

        assert roots.count() == 0

    def test_the_reads_write_nothing(self) -> None:
        for query in (
            backfill_cascade.FORGOTTEN_FACTS_QUERY,
            backfill_cascade.FACT_NAMES_QUERY,
            backfill_cascade.EPISODE_NAMES_QUERY,
            backfill_cascade.MARKER_NAMES_QUERY,
            backfill_cascade.EXISTING_FACTS_QUERY,
            backfill_cascade.HIDDEN_EPISODES_QUERY,
        ):
            assert "SET" not in query and "DELETE" not in query


class TestCascadeExistingForgets:
    @pytest.mark.asyncio
    async def test_cascades_from_the_hard_roots_first_erasing(self) -> None:
        calls: list[tuple[list[str], dict[str, str], bool]] = []

        async def cascade(driver, group_id, roots, now, result, *, erase, seeds):
            calls.append((sorted(roots), seeds, erase))
            result.derived.append(f"d-{len(calls)}")

        with patch.object(backfill_cascade, "cascade", cascade):
            done = await backfill_cascade.cascade_existing_forgets(_Graph(), apply=True)

        assert calls == [
            (["f-gone", "f-pending", "f-purged"], {"said": "f-purged"}, True),
            (["f-soft"], {"chat": "chat", "ep-d2": "f-purged"}, False),
        ]
        assert done == backfill_cascade.Cascaded(roots=7, derived=2, failed=False)

    @pytest.mark.asyncio
    async def test_a_dry_run_counts_and_cascades_nothing(self) -> None:
        cascade = AsyncMock()

        with patch.object(backfill_cascade, "cascade", cascade):
            done = await backfill_cascade.cascade_existing_forgets(
                _Graph(), apply=False
            )

        cascade.assert_not_awaited()
        assert done.roots == 7


class TestThroughTheBackfill:
    @pytest.mark.asyncio
    async def test_a_dry_run_counts_the_forgets_it_would_cascade_from(self) -> None:
        driver = BackfillGraph([], [], forgotten=["x1", "x2"])
        cascade = AsyncMock()

        with patch.object(backfill_cascade, "cascade", cascade):
            found = await backfill.backfill_graph(
                driver, apply=False, cascade_forgets=True
            )

        assert found.roots == 2
        cascade.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_cascades_from_every_forget_after_stamping(self) -> None:
        driver = BackfillGraph([], [], forgotten=["x1", "x2"])

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
        driver = BackfillGraph([], [], forgotten=["x1"])

        async def stop_short(driver, group_id, roots, now, result, **_: object):
            result.failures.append(MemoryForgetFailure.derived_left("x1"))

        with patch.object(
            backfill_cascade, "cascade", AsyncMock(side_effect=stop_short)
        ):
            found = await backfill.backfill_graph(
                driver, apply=True, cascade_forgets=True
            )

        assert found.failed == 1
