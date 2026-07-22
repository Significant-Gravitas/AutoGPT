"""Tests for graphiti_search helper functions and tiered fan-out."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.graphiti.memory_model import MemoryEnvelope, MemoryKind, SourceKind
from backend.copilot.graphiti.tiers import MemoryTier, TierTarget
from backend.copilot.model import ChatSession
from backend.copilot.tools import graphiti_search as search_mod
from backend.copilot.tools.graphiti_search import (
    MemorySearchTool,
    _filter_episodes_by_scope,
    _format_episodes,
)
from backend.copilot.tools.models import ErrorResponse, MemorySearchResponse


class TestFilterEpisodesByScopeTruncation:
    """extract_episode_body() truncates to 500 chars.  A MemoryEnvelope
    with a long content field exceeds that limit, producing invalid JSON.
    _filter_episodes_by_scope then treats it as a plain-text episode
    (real:global), leaking project-scoped data into global results.
    """

    def test_long_envelope_filtered_by_scope(self) -> None:
        envelope = MemoryEnvelope(
            content="x" * 600,
            source_kind=SourceKind.user_asserted,
            scope="project:crm",
            memory_kind=MemoryKind.fact,
        )
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        # Requesting real:global scope — this project:crm episode should be excluded
        results = _filter_episodes_by_scope([ep], "real:global")
        assert (
            results == []
        ), f"project-scoped episode leaked into global results: {results}"

    def test_short_envelope_filtered_correctly(self) -> None:
        """Short envelopes (under 500 chars) are parsed correctly."""
        envelope = MemoryEnvelope(
            content="short note",
            scope="project:crm",
        )
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        results = _filter_episodes_by_scope([ep], "real:global")
        assert results == []


class TestRedundantFormatting:
    """_format_episodes is called even when scope filter will overwrite it.
    Not a correctness bug, but verify the scope path doesn't depend on it.
    """

    def test_scope_filter_independent_of_format_episodes(self) -> None:
        envelope = MemoryEnvelope(content="note", scope="real:global")
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        from_format = _format_episodes([ep])
        from_scope = _filter_episodes_by_scope([ep], "real:global")
        assert len(from_format) == 1
        assert len(from_scope) == 1


# ---------------------------------------------------------------------------
# Tiered search fan-out
# ---------------------------------------------------------------------------


def _fact_edge(fact: str):
    return SimpleNamespace(fact=fact, valid_at="2025-01-01", invalid_at=None)


def _tier_client(edges: list[object]) -> AsyncMock:
    client = AsyncMock()
    client.search.return_value = edges
    client.retrieve_episodes.return_value = []
    return client


def _org_session() -> ChatSession:
    return ChatSession(
        session_id="s",
        user_id="user-1",
        title=None,
        messages=[],
        usage=[],
        credentials={},
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        organization_id="org-1",
    )


class TestMemorySearchTiers:
    @pytest.mark.asyncio
    async def test_all_tiers_label_shared_results(self) -> None:
        tool = MemorySearchTool()
        targets = [
            TierTarget("user_user-1", MemoryTier.personal, None),
            TierTarget("org_org-1", MemoryTier.org, "org memory"),
            TierTarget(
                "team_team-1", MemoryTier.team, "team memory (Platform)", "team-1"
            ),
        ]
        clients = {
            "user_user-1": _tier_client([_fact_edge("personal fact")]),
            "org_org-1": _tier_client([_fact_edge("org fact")]),
            "team_team-1": _tier_client([_fact_edge("team fact")]),
        }

        async def _fake_client(group_id: str):
            return clients[group_id]

        with (
            patch(
                "backend.copilot.tools.graphiti_search.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                search_mod,
                "resolve_search_targets",
                new_callable=AsyncMock,
                return_value=targets,
            ),
            patch.object(search_mod, "get_graphiti_client", side_effect=_fake_client),
        ):
            result = await tool._execute(
                user_id="user-1", session=_org_session(), query="q", tier="all"
            )

        assert isinstance(result, MemorySearchResponse)
        assert any("[org memory] org fact" in f for f in result.facts)
        assert any("[team memory (Platform)] team fact" in f for f in result.facts)
        # Personal fact is unlabelled.
        assert any(f.startswith("personal fact") for f in result.facts)
        # Every resolved (active-membership) tier group was queried, and only those.
        for client in clients.values():
            client.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tier_all_queries_only_active_membership_targets(self) -> None:
        """The tool queries exactly the groups resolve_search_targets returns
        — which is the active-membership source — never an arbitrary team."""
        tool = MemorySearchTool()
        # resolve_search_targets returns ONLY personal (simulating a user with
        # no active team memberships in the org).
        targets = [TierTarget("user_user-1", MemoryTier.personal, None)]
        personal_client = _tier_client([_fact_edge("personal fact")])

        async def _fake_client(group_id: str):
            assert group_id == "user_user-1"
            return personal_client

        with (
            patch(
                "backend.copilot.tools.graphiti_search.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                search_mod,
                "resolve_search_targets",
                new_callable=AsyncMock,
                return_value=targets,
            ) as resolve,
            patch.object(search_mod, "get_graphiti_client", side_effect=_fake_client),
        ):
            result = await tool._execute(
                user_id="user-1", session=_org_session(), query="q", tier="all"
            )

        resolve.assert_awaited_once_with("user-1", "org-1", "all")
        assert isinstance(result, MemorySearchResponse)
        personal_client.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_team_tier_with_no_active_memberships_returns_no_memories(
        self,
    ) -> None:
        tool = MemorySearchTool()
        with (
            patch(
                "backend.copilot.tools.graphiti_search.is_enabled_for_user",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                search_mod,
                "resolve_search_targets",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await tool._execute(
                user_id="user-1", session=_org_session(), query="q", tier="team"
            )

        assert isinstance(result, MemorySearchResponse)
        assert result.facts == []
        assert "No memories found" in result.message

    @pytest.mark.asyncio
    async def test_org_tier_without_org_returns_polite_error(self) -> None:
        tool = MemorySearchTool()
        session = ChatSession(
            session_id="s",
            user_id="user-1",
            title=None,
            messages=[],
            usage=[],
            credentials={},
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        with patch(
            "backend.copilot.tools.graphiti_search.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await tool._execute(
                user_id="user-1", session=session, query="q", tier="org"
            )

        assert isinstance(result, ErrorResponse)
        assert "organization" in result.message.lower()
