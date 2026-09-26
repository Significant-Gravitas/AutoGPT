"""Tests for Graphiti warm context retrieval."""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER

from . import context
from .context import _format_context, fetch_warm_context
from .memory_model import MemoryEnvelope
from .scope import MemoryScope

_NOW = datetime(2025, 6, 1, tzinfo=timezone.utc)


def _edge(uuid: str = "edge-a", fact: str = "user likes python") -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id="user_abc",
        source_node_uuid="user",
        target_node_uuid="python",
        created_at=_NOW,
        name="preference",
        fact=fact,
        valid_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        attributes={"status": "active"},
    )


def _episode(content: str) -> EpisodicNode:
    return EpisodicNode(
        name="ep",
        group_id="user_abc",
        source=EpisodeType.text,
        source_description="chat",
        content=content,
        created_at=_NOW,
        valid_at=_NOW,
    )


class TestFetchWarmContextEmptyUserId:
    @pytest.mark.asyncio
    async def test_returns_none_for_empty_user_id(self) -> None:
        result = await fetch_warm_context("", "hello")
        assert result is None


class TestFetchWarmContextTimeout:
    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _slow_fetch(scope: MemoryScope, message: str) -> str:
            await asyncio.sleep(10)
            return "<temporal_context>data</temporal_context>"

        with patch.object(context, "_fetch", side_effect=_slow_fetch):
            # Set an extremely short timeout.
            monkeypatch.setattr(context.graphiti_config, "context_timeout", 0.01)
            result = await fetch_warm_context("valid-user-id", "hello")

        assert result is None


class TestFetchWarmContextGeneralError:
    @pytest.mark.asyncio
    async def test_returns_none_on_unexpected_error(self) -> None:
        with (
            patch.object(
                context,
                "search_facts",
                new_callable=AsyncMock,
                side_effect=RuntimeError("connection lost"),
            ),
            patch.object(context, "recent_episodes", new_callable=AsyncMock),
        ):
            result = await fetch_warm_context("abc", "hello")

        assert result is None


class TestFetchInternal:
    """``_fetch`` reads through the recall policy (``recall.py``).

    Both reads are mocked at their use site; the hit hook is replaced so no
    test here reaches Redis or FalkorDB.
    """

    @pytest.fixture(autouse=True)
    def spawn_hits(self):
        with patch.object(context, "_spawn_ratification_hits") as spawn:
            yield spawn

    @staticmethod
    def _reads(edges: list[EntityEdge], episodes: list[EpisodicNode]):
        return (
            patch.object(
                context, "search_facts", new_callable=AsyncMock, return_value=edges
            ),
            patch.object(
                context,
                "recent_episodes",
                new_callable=AsyncMock,
                return_value=episodes,
            ),
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_no_edges_or_episodes(self, spawn_hits) -> None:
        search, recent = self._reads([], [])
        with search, recent:
            result = await context._fetch(MemoryScope.for_user("abc"), "hello")

        assert result is None
        spawn_hits.assert_not_called()

    @pytest.mark.asyncio
    async def test_expert_scope_is_used_for_all_retrievals(self) -> None:
        search, recent = self._reads([], [])
        with search as search_mock, recent as recent_mock:
            await fetch_warm_context("user-1", "hello", expert_id="expert-1")

        expert_scope = MemoryScope.for_expert("user-1", "expert-1")
        assert search_mock.await_args.args[0] == expert_scope
        assert recent_mock.await_args.args[0] == expert_scope

    @pytest.mark.asyncio
    async def test_returns_context_with_edges(self, spawn_hits) -> None:
        edge = _edge()
        search, recent = self._reads([edge], [])
        with search, recent:
            result = await context._fetch(MemoryScope.for_user("abc"), "hello")

        assert result is not None
        assert "<temporal_context>" in result
        assert "user likes python" in result
        spawn_hits.assert_called_once_with(MemoryScope.for_user("abc"), [edge])

    @pytest.mark.asyncio
    async def test_returns_context_with_episodes(self) -> None:
        search, recent = self._reads([], [_episode("talked about coffee")])
        with search, recent:
            result = await context._fetch(MemoryScope.for_user("abc"), "hello")

        assert result is not None
        assert "talked about coffee" in result

    @pytest.mark.asyncio
    async def test_search_uses_cross_encoder_recipe(self) -> None:
        """P-1.4 contract: warm context must use the cross-encoder recipe,
        limited to ``context_max_facts``, and five recent episodes."""
        search, recent = self._reads([], [])
        with search as search_mock, recent as recent_mock:
            await context._fetch(MemoryScope.for_user("abc"), "hello world")

        search_mock.assert_awaited_once()
        assert search_mock.await_args.args == (
            MemoryScope.for_user("abc"),
            "hello world",
        )
        kwargs = search_mock.await_args.kwargs
        assert kwargs["recipe"] is EDGE_HYBRID_SEARCH_CROSS_ENCODER
        assert kwargs["limit"] == context.graphiti_config.context_max_facts
        recent_mock.assert_awaited_once_with(MemoryScope.for_user("abc"), 5)


class TestFormatContextWithContent:
    """Test _format_context with actual edges and episodes."""

    def test_with_edges_only(self) -> None:
        result = _format_context(edges=[_edge(fact="user likes coffee")], episodes=[])
        assert result is not None
        assert "<FACTS>" in result
        assert (
            "  - user likes coffee (valid: 2025-01-01 00:00:00+00:00 — present)"
            in result
        )
        assert "<temporal_context>" in result

    def test_with_episodes_only(self) -> None:
        result = _format_context(
            edges=[], episodes=[_episode("plain conversation text")]
        )
        assert result is not None
        assert "<RECENT_EPISODES>" in result
        assert "plain conversation text" in result

    def test_with_both_edges_and_episodes(self) -> None:
        result = _format_context(
            edges=[_edge(fact="user likes coffee")],
            episodes=[_episode("talked about coffee")],
        )
        assert result is not None
        assert "<FACTS>" in result
        assert "<RECENT_EPISODES>" in result

    def test_global_scope_episode_included(self) -> None:
        envelope = MemoryEnvelope(content="global note", scope="real:global")
        result = _format_context(
            edges=[], episodes=[_episode(envelope.model_dump_json())]
        )
        assert result is not None
        assert "<RECENT_EPISODES>" in result

    def test_non_global_scope_episode_excluded(self) -> None:
        envelope = MemoryEnvelope(content="project note", scope="project:crm")
        result = _format_context(
            edges=[], episodes=[_episode(envelope.model_dump_json())]
        )
        assert result is None


# ---------------------------------------------------------------------------
# Bug: empty <temporal_context> wrapper when all episodes are non-global
# ---------------------------------------------------------------------------


class TestFormatContextEmptyWrapper:
    """When all episodes are non-global and edges is empty, _format_context
    should return None (no useful content) instead of an empty XML wrapper.
    """

    def test_returns_none_when_all_episodes_filtered(self) -> None:
        envelope = MemoryEnvelope(
            content="project-only note",
            scope="project:crm",
        )
        result = _format_context(
            edges=[], episodes=[_episode(envelope.model_dump_json())]
        )
        assert result is None


# ---------------------------------------------------------------------------
# Ratification sync hit-hook spawned from warm-context retrieval
# ---------------------------------------------------------------------------


class TestRatificationHitHookFiresFireAndForget:
    """The hit-hook records warm-context hits + promotes tentative
    edges inline. It must NOT block the retrieval response — the
    chat turn cares about latency, the promotion can race the next
    retrieval to apply."""

    def test_spawn_helper_skips_empty_edge_list_no_task_created(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created_tasks: list[str] = []

        def fake_create_task(coro, name=None):
            created_tasks.append(name or "")
            coro.close()  # don't actually run the coroutine in test
            return AsyncMock()

        monkeypatch.setattr(context.asyncio, "create_task", fake_create_task)
        context._spawn_ratification_hits(MemoryScope.for_user("user-abc"), edges=[])
        assert created_tasks == []

    def test_spawn_helper_creates_task_with_retrieved_uuids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrieved edges → one fire-and-forget task carrying all of their
        uuids."""
        captured_calls: list[tuple[MemoryScope, list[str]]] = []

        async def fake_try_ratify(scope: MemoryScope, edge_uuids: list[str]):
            captured_calls.append((scope, edge_uuids))

        from backend.copilot.dream import ratification as ratification_mod

        monkeypatch.setattr(ratification_mod, "try_ratify_on_hit", fake_try_ratify)

        # asyncio.create_task needs an event loop — exercise via
        # run_until_complete instead of an actual task spawn.
        async def driver():
            edges = [_edge("edge-a"), _edge("edge-b")]
            context._spawn_ratification_hits(
                MemoryScope.for_expert("user-xyz", "expert-1"), edges=edges
            )
            # Yield once so the spawned task runs.
            await asyncio.sleep(0)

        asyncio.run(driver())
        assert captured_calls == [
            (MemoryScope.for_expert("user-xyz", "expert-1"), ["edge-a", "edge-b"])
        ]


class TestRatificationHitTaskRetention:
    """The event loop holds only weak references to tasks — the spawn
    helper must keep a strong reference until the task completes, or GC
    pressure can collect the hit-recording task mid-flight and silently
    drop hits the nightly sweep can never see."""

    def test_spawned_hit_task_retained_until_done_then_discarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from backend.copilot.dream import ratification as ratification_mod

        async def driver():
            context._pending_hit_tasks.clear()
            release = asyncio.Event()

            async def fake_try_ratify(scope: MemoryScope, edge_uuids: list[str]):
                await release.wait()

            monkeypatch.setattr(ratification_mod, "try_ratify_on_hit", fake_try_ratify)
            context._spawn_ratification_hits(
                MemoryScope.for_user("user-xyz"), edges=[_edge("edge-a")]
            )
            # Strong ref held while the task is in flight.
            assert len(context._pending_hit_tasks) == 1
            task = next(iter(context._pending_hit_tasks))

            release.set()
            await task
            # One more tick so the done-callback (call_soon) runs.
            await asyncio.sleep(0)
            assert context._pending_hit_tasks == set()

        asyncio.run(driver())

    def test_failed_hit_task_logs_exception_and_is_discarded(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The done-callback must observe the exception (no 'Task
        exception was never retrieved' noise) and log it at WARNING."""
        from backend.copilot.dream import ratification as ratification_mod

        async def driver():
            context._pending_hit_tasks.clear()

            async def fake_try_ratify(scope: MemoryScope, edge_uuids: list[str]):
                raise RuntimeError("falkordb down")

            monkeypatch.setattr(ratification_mod, "try_ratify_on_hit", fake_try_ratify)
            context._spawn_ratification_hits(
                MemoryScope.for_user("user-xyz"), edges=[_edge("edge-a")]
            )
            task = next(iter(context._pending_hit_tasks))
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)

        with caplog.at_level(
            logging.WARNING, logger="backend.copilot.graphiti.context"
        ):
            asyncio.run(driver())

        assert context._pending_hit_tasks == set()
        assert any(
            record.levelno == logging.WARNING and "failed" in record.getMessage()
            for record in caplog.records
        )
