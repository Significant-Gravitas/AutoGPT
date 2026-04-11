"""Tests for Graphiti warm context retrieval."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from . import context
from .context import fetch_warm_context


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
