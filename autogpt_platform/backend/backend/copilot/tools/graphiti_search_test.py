"""Tests for graphiti_search helper functions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.graphiti.memory_model import MemoryEnvelope, MemoryKind, SourceKind
from backend.copilot.model import ChatSession
from backend.copilot.tools.graphiti_search import (
    MemorySearchTool,
    _filter_episodes_by_scope,
    _format_episodes,
)
from backend.copilot.tools.models import MemorySearchResponse


@pytest.mark.asyncio
async def test_expert_session_searches_only_expert_memory_group() -> None:
    session = ChatSession.new(
        "user-1",
        dry_run=False,
        expert_id="expert-1",
    )
    client = SimpleNamespace(
        search=AsyncMock(return_value=[]),
        retrieve_episodes=AsyncMock(return_value=[]),
    )

    with (
        patch(
            "backend.copilot.tools.graphiti_search.is_enabled_for_user",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "backend.copilot.tools.graphiti_search.derive_memory_group_id",
            return_value="expert_private_group",
        ) as derive_mock,
        patch(
            "backend.copilot.tools.graphiti_search.get_graphiti_client",
            new_callable=AsyncMock,
            return_value=client,
        ) as get_client_mock,
    ):
        result = await MemorySearchTool()._execute(
            "user-1",
            session,
            query="private fact",
        )

    assert isinstance(result, MemorySearchResponse)
    derive_mock.assert_called_once_with("user-1", "expert-1")
    get_client_mock.assert_awaited_once_with("expert_private_group")
    client.search.assert_awaited_once_with(
        query="private fact",
        group_ids=["expert_private_group"],
        num_results=15,
    )
    assert client.retrieve_episodes.await_args.kwargs["group_ids"] == [
        "expert_private_group"
    ]


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
