"""Tests for the ask_question choice-button token store."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot import choices


@pytest.mark.asyncio
async def test_store_choice_writes_json_with_ttl():
    redis = MagicMock()
    redis.set = AsyncMock()
    with patch(
        "backend.copilot.bot.choices.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        token = await choices.store_choice("discord", ["US", "EU"])

    assert len(token) == 12
    key, value = redis.set.await_args.args
    assert key == f"copilot-bot:choice:discord:{token}"
    assert json.loads(value) == ["US", "EU"]
    assert redis.set.await_args.kwargs["ex"] == choices.CHOICE_TTL


@pytest.mark.asyncio
async def test_resolve_choice_returns_option_by_index():
    redis = MagicMock()
    redis.getdel = AsyncMock(return_value=json.dumps(["US", "EU"]))
    with patch(
        "backend.copilot.bot.choices.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        assert await choices.resolve_choice("discord", "tok", 1) == "EU"
    redis.getdel.assert_awaited_once_with("copilot-bot:choice:discord:tok")


@pytest.mark.asyncio
async def test_resolve_choice_missing_token_returns_none():
    redis = MagicMock()
    redis.getdel = AsyncMock(return_value=None)
    with patch(
        "backend.copilot.bot.choices.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        assert await choices.resolve_choice("discord", "gone", 0) is None


@pytest.mark.asyncio
async def test_resolve_choice_out_of_range_index_returns_none():
    """A forged or stale payload must not index-error into a 500."""
    redis = MagicMock()
    redis.getdel = AsyncMock(return_value=json.dumps(["US", "EU"]))
    with patch(
        "backend.copilot.bot.choices.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        assert await choices.resolve_choice("discord", "tok", 5) is None


@pytest.mark.asyncio
async def test_resolve_choice_is_single_use():
    """GETDEL makes the fetch atomic: a double-click or a platform-level
    delivery retry racing the same click twice must not both resolve, since
    each resolve continues the paused AutoPilot turn (a billable action)."""
    store: dict[str, str] = {"copilot-bot:choice:discord:tok": json.dumps(["US"])}

    async def fake_getdel(key: str) -> str | None:
        return store.pop(key, None)

    redis = MagicMock()
    redis.getdel = AsyncMock(side_effect=fake_getdel)
    with patch(
        "backend.copilot.bot.choices.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        assert await choices.resolve_choice("discord", "tok", 0) == "US"
        assert await choices.resolve_choice("discord", "tok", 0) is None


@pytest.mark.asyncio
async def test_clear_choice_deletes_key():
    redis = MagicMock()
    redis.delete = AsyncMock()
    with patch(
        "backend.copilot.bot.choices.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        await choices.clear_choice("discord", "tok")

    redis.delete.assert_awaited_once_with("copilot-bot:choice:discord:tok")
