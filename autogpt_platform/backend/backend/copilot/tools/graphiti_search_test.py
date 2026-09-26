"""Tests for the memory_search tool."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode

from backend.copilot.graphiti.memory_model import MemoryEnvelope, MemoryKind, SourceKind
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_search import MemorySearchTool
from backend.copilot.tools.models import MemorySearchResponse

_MODULE = "backend.copilot.tools.graphiti_search"
_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _edge(uuid: str, fact: str) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id="user_user-1",
        source_node_uuid="a",
        target_node_uuid="b",
        created_at=_NOW,
        name="relates",
        fact=fact,
        valid_at=_NOW,
        attributes={"status": "active"},
    )


def _episode(content: str) -> EpisodicNode:
    return EpisodicNode(
        name="ep",
        group_id="user_user-1",
        source=EpisodeType.text,
        source_description="chat",
        content=content,
        created_at=_NOW,
        valid_at=_NOW,
    )


async def _search(
    edges: list[EntityEdge],
    episodes: list[EpisodicNode],
    *,
    expert_id: str | None = None,
    **tool_kwargs,
):
    """Run the tool with both recall reads and the hit recorder mocked."""
    session = ChatSession.new("user-1", dry_run=False, expert_id=expert_id)
    search_facts = AsyncMock(return_value=edges)
    recent_episodes = AsyncMock(return_value=episodes)
    record_hit = MagicMock(return_value="hit-coroutine")
    spawn = MagicMock()
    with (
        patch(f"{_MODULE}.is_enabled_for_user", AsyncMock(return_value=True)),
        patch(f"{_MODULE}.search_facts", search_facts),
        patch(f"{_MODULE}.recent_episodes", recent_episodes),
        patch(f"{_MODULE}.record_hit", record_hit),
        patch(f"{_MODULE}.spawn_background_task", spawn),
    ):
        result = await MemorySearchTool()._execute(
            "user-1", session, query="private fact", **tool_kwargs
        )
    return result, search_facts, recent_episodes, record_hit, spawn


@pytest.mark.asyncio
async def test_expert_session_searches_only_expert_memory_group() -> None:
    result, search_facts, recent_episodes, _, _ = await _search(
        [], [], expert_id="expert-1"
    )

    assert isinstance(result, MemorySearchResponse)
    expert_scope = MemoryScope.for_expert("user-1", "expert-1")
    search_facts.assert_awaited_once_with(expert_scope, "private fact", limit=15)
    recent_episodes.assert_awaited_once_with(expert_scope, 5)


@pytest.mark.asyncio
async def test_renders_facts_and_episodes() -> None:
    result, *_ = await _search(
        [_edge("e1", "Alice works on Atlas")], [_episode("talked about Atlas")]
    )

    assert isinstance(result, MemorySearchResponse)
    assert result.facts == [
        "Alice works on Atlas (valid: 2025-01-01 00:00:00+00:00 — present)"
    ]
    assert result.recent_episodes == ["[2025-01-01 00:00:00+00:00] talked about Atlas"]


@pytest.mark.asyncio
async def test_returned_facts_are_counted_as_hits() -> None:
    """memory_search feeds the ratification sweep: a tentative fact the
    model retrieves here has been used, like one warm context surfaced."""
    edges = [_edge("e1", "fact one"), _edge("e2", "fact two")]

    _, _, _, record_hit, spawn = await _search(edges, [])

    record_hit.assert_called_once_with(MemoryScope.for_user("user-1"), ["e1", "e2"])
    spawn.assert_called_once()
    assert spawn.call_args.args == ("hit-coroutine",)


@pytest.mark.asyncio
async def test_no_facts_no_hit_task() -> None:
    _, _, _, record_hit, spawn = await _search([], [_episode("just chatting")])

    record_hit.assert_not_called()
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_scope_filter_drops_other_scopes_even_for_long_envelopes() -> None:
    """A MemoryEnvelope longer than the 500-char display cut is still scoped
    by its full body, so a project memory cannot leak into a global search."""
    long_project = MemoryEnvelope(
        content="x" * 600,
        source_kind=SourceKind.user_asserted,
        scope="project:crm",
        memory_kind=MemoryKind.fact,
    ).model_dump_json()
    episodes = [_episode(long_project), _episode("plain conversation")]

    result, *_ = await _search([], episodes, scope="real:global")

    assert isinstance(result, MemorySearchResponse)
    assert result.recent_episodes == ["[2025-01-01 00:00:00+00:00] plain conversation"]


@pytest.mark.asyncio
async def test_search_failure_is_reported_not_raised() -> None:
    session = ChatSession.new("user-1", dry_run=False)
    with (
        patch(f"{_MODULE}.is_enabled_for_user", AsyncMock(return_value=True)),
        patch(f"{_MODULE}.search_facts", AsyncMock(side_effect=RuntimeError("down"))),
        patch(f"{_MODULE}.recent_episodes", AsyncMock(return_value=[])),
    ):
        result = await MemorySearchTool()._execute("user-1", session, query="x")

    assert "temporarily unavailable" in result.message
