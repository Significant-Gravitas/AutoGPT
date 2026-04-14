"""Tests for Graphiti warm context retrieval."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from . import context
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
        from backend.copilot.graphiti._format import extract_episode_body

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
