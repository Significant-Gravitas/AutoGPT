"""Tests for Graphiti warm context retrieval."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from . import context
from ._format import extract_episode_body
from .context import _format_context, _is_non_global_scope, fetch_warm_context
from .memory_model import MemoryEnvelope, MemoryKind, SourceKind


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
        async def _slow_fetch(user_id: str, message: str) -> str:
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
        with patch.object(
            context,
            "get_graphiti_client",
            new_callable=AsyncMock,
            side_effect=RuntimeError("connection lost"),
        ):
            result = await fetch_warm_context("abc", "hello")

        assert result is None


# ---------------------------------------------------------------------------
# Bug: extract_episode_body() truncation breaks scope filtering
# ---------------------------------------------------------------------------


def _search_results(edges: list[object]) -> SimpleNamespace:
    """Stand-in for graphiti_core.search.search_config.SearchResults — only
    the ``.edges`` attribute is exercised by ``_fetch``."""
    return SimpleNamespace(edges=edges)


class TestFetchInternal:
    """Test the internal _fetch function with mocked graphiti client.

    After P-1.4, ``_fetch`` calls ``client.search_()`` (note trailing
    underscore) with the ``EDGE_HYBRID_SEARCH_CROSS_ENCODER`` recipe and
    expects a ``SearchResults`` object whose ``.edges`` attribute carries
    the candidate list. The mocks below reflect that shape.
    """

    @pytest.mark.asyncio
    async def test_returns_none_when_no_edges_or_episodes(self) -> None:
        mock_client = AsyncMock()
        mock_client.search_.return_value = _search_results([])
        mock_client.retrieve_episodes.return_value = []

        with patch.object(
            context,
            "get_graphiti_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await context._fetch("test-user", "hello")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_context_with_edges(self) -> None:
        edge = SimpleNamespace(
            fact="user likes python",
            name="preference",
            valid_at="2025-01-01",
            invalid_at=None,
        )
        mock_client = AsyncMock()
        mock_client.search_.return_value = _search_results([edge])
        mock_client.retrieve_episodes.return_value = []

        with patch.object(
            context,
            "get_graphiti_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await context._fetch("test-user", "hello")

        assert result is not None
        assert "<temporal_context>" in result
        assert "user likes python" in result

    @pytest.mark.asyncio
    async def test_returns_context_with_episodes(self) -> None:
        ep = SimpleNamespace(
            content="talked about coffee",
            created_at="2025-06-01T00:00:00Z",
        )
        mock_client = AsyncMock()
        mock_client.search_.return_value = _search_results([])
        mock_client.retrieve_episodes.return_value = [ep]

        with patch.object(
            context,
            "get_graphiti_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await context._fetch("test-user", "hello")

        assert result is not None
        assert "talked about coffee" in result

    @pytest.mark.asyncio
    async def test_search_call_uses_cross_encoder_recipe(self) -> None:
        """P-1.4 contract: warm context must use the cross-encoder recipe.

        Pins both the method (``search_`` not ``search``) and the recipe
        passed as ``config=``. If a future refactor swaps in a different
        recipe, this test fires.
        """
        mock_client = AsyncMock()
        mock_client.search_.return_value = _search_results([])
        mock_client.retrieve_episodes.return_value = []

        with patch.object(
            context,
            "get_graphiti_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            await context._fetch("test-user", "hello world")

        mock_client.search_.assert_awaited_once()
        kwargs = mock_client.search_.await_args.kwargs
        assert kwargs["query"] == "hello world"
        # Personal tier group derived directly from the user id ("test-user").
        assert kwargs["group_ids"] == ["user_test-user"]
        # The config is a copy of EDGE_HYBRID_SEARCH_CROSS_ENCODER with the
        # limit overridden to context_max_facts. Verify the edge-config
        # reranker is still ``cross_encoder`` so the contract is locked.
        from graphiti_core.search.search_config import EdgeReranker

        cfg = kwargs["config"]
        assert cfg.edge_config is not None
        assert cfg.edge_config.reranker == EdgeReranker.cross_encoder
        assert cfg.limit == context.graphiti_config.context_max_facts


class TestFormatContextWithContent:
    """Test _format_context with actual edges and episodes."""

    def test_with_edges_only(self) -> None:
        edge = SimpleNamespace(
            fact="user likes coffee",
            name="preference",
            valid_at="2025-01-01",
            invalid_at="present",
        )
        result = _format_context([(edge, None)], [])
        assert result is not None
        assert "<FACTS>" in result
        assert "user likes coffee" in result
        assert "<temporal_context>" in result

    def test_with_episodes_only(self) -> None:
        ep = SimpleNamespace(
            content="plain conversation text",
            created_at="2025-01-01T00:00:00Z",
        )
        result = _format_context([], [(ep, None)])
        assert result is not None
        assert "<RECENT_EPISODES>" in result
        assert "plain conversation text" in result

    def test_with_both_edges_and_episodes(self) -> None:
        edge = SimpleNamespace(
            fact="user likes coffee",
            valid_at="2025-01-01",
            invalid_at=None,
        )
        ep = SimpleNamespace(
            content="talked about coffee",
            created_at="2025-06-01T00:00:00Z",
        )
        result = _format_context([(edge, None)], [(ep, None)])
        assert result is not None
        assert "<FACTS>" in result
        assert "<RECENT_EPISODES>" in result

    def test_global_scope_episode_included(self) -> None:
        envelope = MemoryEnvelope(content="global note", scope="real:global")
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        result = _format_context([], [(ep, None)])
        assert result is not None
        assert "<RECENT_EPISODES>" in result

    def test_non_global_scope_episode_excluded(self) -> None:
        envelope = MemoryEnvelope(content="project note", scope="project:crm")
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        result = _format_context([], [(ep, None)])
        assert result is None


class TestIsNonGlobalScopeEdgeCases:
    """Verify _is_non_global_scope handles non-dict JSON without crashing."""

    def test_list_json_treated_as_global(self) -> None:
        assert _is_non_global_scope("[1, 2, 3]") is False

    def test_string_json_treated_as_global(self) -> None:
        assert _is_non_global_scope('"just a string"') is False

    def test_null_json_treated_as_global(self) -> None:
        assert _is_non_global_scope("null") is False

    def test_plain_text_treated_as_global(self) -> None:
        assert _is_non_global_scope("plain conversation text") is False


class TestIsNonGlobalScopeTruncation:
    """Verify _is_non_global_scope handles long MemoryEnvelope JSON.

    extract_episode_body() truncates to 500 chars.  A MemoryEnvelope with
    a long content field serializes to >500 chars, so the truncated string
    is invalid JSON.  The except clause falls through to return False,
    incorrectly treating a project-scoped episode as global.
    """

    def test_long_envelope_with_non_global_scope_detected(self) -> None:
        """Long MemoryEnvelope JSON should be parsed with raw (untruncated) body."""
        envelope = MemoryEnvelope(
            content="x" * 600,
            source_kind=SourceKind.user_asserted,
            scope="project:crm",
            memory_kind=MemoryKind.fact,
        )
        full_json = envelope.model_dump_json()
        assert len(full_json) > 500, "precondition: JSON must exceed truncation limit"

        # With the fix: _is_non_global_scope on the raw (untruncated) body
        # correctly detects the non-global scope.
        assert _is_non_global_scope(full_json) is True

        # Truncated body still fails — that's expected; callers must use raw body.
        ep = SimpleNamespace(content=full_json)
        truncated = extract_episode_body(ep)
        assert _is_non_global_scope(truncated) is False  # truncated JSON → parse fails


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
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        result = _format_context([], [(ep, None)])
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
        context._spawn_ratification_hits("user-abc", edges=[])
        assert created_tasks == []

    def test_spawn_helper_creates_task_with_retrieved_uuids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Edges with uuid attrs → fire-and-forget task scheduled with
        all of their uuids. Edges missing a uuid are filtered out so
        the hook never passes ``None`` to the ratification module."""
        captured_calls: list[tuple[str, list[str]]] = []

        async def fake_try_ratify(user_id: str, edge_uuids: list[str]):
            captured_calls.append((user_id, edge_uuids))

        from backend.copilot.dream import ratification as ratification_mod

        monkeypatch.setattr(ratification_mod, "try_ratify_on_hit", fake_try_ratify)

        # asyncio.create_task needs an event loop — exercise via
        # run_until_complete instead of an actual task spawn.
        async def driver():
            edges = [
                SimpleNamespace(uuid="edge-a"),
                SimpleNamespace(uuid="edge-b"),
                SimpleNamespace(uuid=None),  # filtered
                SimpleNamespace(),  # no uuid attr at all → filtered
            ]
            context._spawn_ratification_hits("user-xyz", edges=edges)
            # Yield once so the spawned task runs.
            await asyncio.sleep(0)

        asyncio.run(driver())
        assert len(captured_calls) == 1
        user_id, uuids = captured_calls[0]
        assert user_id == "user-xyz"
        assert uuids == ["edge-a", "edge-b"]


# ---------------------------------------------------------------------------
# Tiered fan-out: personal + org + session-team, provenance labels, budget
# ---------------------------------------------------------------------------


def _fact_edge(fact: str):
    return SimpleNamespace(
        fact=fact, name="rel", valid_at="2025-01-01", invalid_at=None
    )


def _tier_client(edges: list[object]) -> AsyncMock:
    client = AsyncMock()
    client.search_.return_value = _search_results(edges)
    client.retrieve_episodes.return_value = []
    return client


class TestWarmContextFanOut:
    @pytest.mark.asyncio
    async def test_fans_out_and_labels_shared_tiers(self) -> None:
        from .tiers import MemoryTier, TierTarget

        targets = [
            TierTarget("user_u1", MemoryTier.personal, None),
            TierTarget("org_org-1", MemoryTier.org, "org memory"),
            TierTarget(
                "team_team-1", MemoryTier.team, "team memory (Platform)", "team-1"
            ),
        ]
        clients = {
            "user_u1": _tier_client([_fact_edge("personal pref")]),
            "org_org-1": _tier_client([_fact_edge("org policy")]),
            "team_team-1": _tier_client([_fact_edge("team convention")]),
        }

        async def _fake_client(group_id: str):
            return clients[group_id]

        with (
            patch.object(
                context,
                "resolve_warm_targets",
                new_callable=AsyncMock,
                return_value=targets,
            ),
            patch.object(context, "get_graphiti_client", side_effect=_fake_client),
            patch.object(context, "_spawn_ratification_hits"),
        ):
            result = await context._fetch("u1", "query", "org-1", "team-1")

        assert result is not None
        # Personal fact is unlabelled; shared facts carry provenance labels.
        assert "personal pref" in result
        assert "[org memory] org policy" in result
        assert "[team memory (Platform)] team convention" in result
        # All three tier groups were queried.
        for client in clients.values():
            client.search_.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ratification_fires_only_on_personal_edges(self) -> None:
        from .tiers import MemoryTier, TierTarget

        personal_edge = _fact_edge("personal pref")
        org_edge = _fact_edge("org policy")
        targets = [
            TierTarget("user_u1", MemoryTier.personal, None),
            TierTarget("org_org-1", MemoryTier.org, "org memory"),
        ]
        clients = {
            "user_u1": _tier_client([personal_edge]),
            "org_org-1": _tier_client([org_edge]),
        }

        async def _fake_client(group_id: str):
            return clients[group_id]

        with (
            patch.object(
                context,
                "resolve_warm_targets",
                new_callable=AsyncMock,
                return_value=targets,
            ),
            patch.object(context, "get_graphiti_client", side_effect=_fake_client),
            patch.object(context, "_spawn_ratification_hits") as spawn,
        ):
            await context._fetch("u1", "query", "org-1", None)

        # Ratification-on-hit keys off the personal graph — it must receive
        # ONLY personal edges, never the org edge.
        spawn.assert_called_once()
        _, kwargs = spawn.call_args
        passed_edges = (
            kwargs.get("edges") if "edges" in kwargs else spawn.call_args[0][1]
        )
        assert org_edge not in passed_edges
        assert personal_edge in passed_edges

    @pytest.mark.asyncio
    async def test_flaky_shared_tier_does_not_sink_personal_context(self) -> None:
        from .tiers import MemoryTier, TierTarget

        targets = [
            TierTarget("user_u1", MemoryTier.personal, None),
            TierTarget("org_org-1", MemoryTier.org, "org memory"),
        ]
        personal_client = _tier_client([_fact_edge("personal pref")])

        async def _fake_client(group_id: str):
            if group_id == "org_org-1":
                raise RuntimeError("org graph down")
            return personal_client

        with (
            patch.object(
                context,
                "resolve_warm_targets",
                new_callable=AsyncMock,
                return_value=targets,
            ),
            patch.object(context, "get_graphiti_client", side_effect=_fake_client),
            patch.object(context, "_spawn_ratification_hits"),
        ):
            result = await context._fetch("u1", "query", "org-1", None)

        assert result is not None
        assert "personal pref" in result

    @pytest.mark.asyncio
    async def test_personal_keeps_at_least_half_the_fact_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from .tiers import MemoryTier, TierTarget

        monkeypatch.setattr(context.graphiti_config, "context_max_facts", 10)

        personal_edges = [_fact_edge(f"personal {i}") for i in range(10)]
        org_edges = [_fact_edge(f"org {i}") for i in range(10)]
        targets = [
            TierTarget("user_u1", MemoryTier.personal, None),
            TierTarget("org_org-1", MemoryTier.org, "org memory"),
        ]
        clients = {
            "user_u1": _tier_client(personal_edges),
            "org_org-1": _tier_client(org_edges),
        }

        async def _fake_client(group_id: str):
            return clients[group_id]

        with (
            patch.object(
                context,
                "resolve_warm_targets",
                new_callable=AsyncMock,
                return_value=targets,
            ),
            patch.object(context, "get_graphiti_client", side_effect=_fake_client),
            patch.object(context, "_spawn_ratification_hits"),
        ):
            result = await context._fetch("u1", "query", "org-1", None)

        assert result is not None
        personal_shown = sum(1 for i in range(10) if f"personal {i}" in result)
        # Budget 10, floor is 5 — personal must keep at least half.
        assert personal_shown >= 5
