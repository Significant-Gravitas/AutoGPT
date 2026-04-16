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
        with (
            patch.object(
                context,
                "derive_group_id",
                return_value="user_abc",
            ),
            patch.object(
                context,
                "get_graphiti_client",
                new_callable=AsyncMock,
                side_effect=RuntimeError("connection lost"),
            ),
        ):
            result = await fetch_warm_context("abc", "hello")

        assert result is None


# ---------------------------------------------------------------------------
# Bug: extract_episode_body() truncation breaks scope filtering
# ---------------------------------------------------------------------------


class TestFetchInternal:
    """Test the internal _fetch function with mocked graphiti client."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_edges_or_episodes(self) -> None:
        mock_client = AsyncMock()
        mock_client.search.return_value = []
        mock_client.retrieve_episodes.return_value = []

        with (
            patch.object(context, "derive_group_id", return_value="user_abc"),
            patch.object(
                context,
                "get_graphiti_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
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
        mock_client.search.return_value = [edge]
        mock_client.retrieve_episodes.return_value = []

        with (
            patch.object(context, "derive_group_id", return_value="user_abc"),
            patch.object(
                context,
                "get_graphiti_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
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
        mock_client.search.return_value = []
        mock_client.retrieve_episodes.return_value = [ep]

        with (
            patch.object(context, "derive_group_id", return_value="user_abc"),
            patch.object(
                context,
                "get_graphiti_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
        ):
            result = await context._fetch("test-user", "hello")

        assert result is not None
        assert "talked about coffee" in result


class TestFormatContextWithContent:
    """Test _format_context with actual edges and episodes."""

    def test_with_edges_only(self) -> None:
        edge = SimpleNamespace(
            fact="user likes coffee",
            name="preference",
            valid_at="2025-01-01",
            invalid_at="present",
        )
        result = _format_context(edges=[edge], episodes=[])
        assert result is not None
        assert "<FACTS>" in result
        assert "user likes coffee" in result
        assert "<temporal_context>" in result

    def test_with_episodes_only(self) -> None:
        ep = SimpleNamespace(
            content="plain conversation text",
            created_at="2025-01-01T00:00:00Z",
        )
        result = _format_context(edges=[], episodes=[ep])
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
        result = _format_context(edges=[edge], episodes=[ep])
        assert result is not None
        assert "<FACTS>" in result
        assert "<RECENT_EPISODES>" in result

    def test_global_scope_episode_included(self) -> None:
        envelope = MemoryEnvelope(content="global note", scope="real:global")
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        result = _format_context(edges=[], episodes=[ep])
        assert result is not None
        assert "<RECENT_EPISODES>" in result

    def test_non_global_scope_episode_excluded(self) -> None:
        envelope = MemoryEnvelope(content="project note", scope="project:crm")
        ep = SimpleNamespace(
            content=envelope.model_dump_json(),
            created_at="2025-01-01T00:00:00Z",
        )
        result = _format_context(edges=[], episodes=[ep])
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
        result = _format_context(edges=[], episodes=[ep])
        assert result is None
