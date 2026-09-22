"""Tests for bot_analytics writer functions (prisma mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import bot_analytics
from backend.platform_linking.models import BotEventInput, BotGuildInput, Platform


@pytest.mark.asyncio
async def test_record_bot_event_creates_row():
    prisma_mock = MagicMock()
    prisma_mock.create = AsyncMock()
    with patch.object(bot_analytics.BotEvent, "prisma", return_value=prisma_mock):
        await bot_analytics.record_bot_event(
            BotEventInput(
                platform=Platform.DISCORD,
                event_type="reply_sent",
                server_id="9",
                char_count=100,
                duration_ms=2500,
            )
        )

    data = prisma_mock.create.await_args.kwargs["data"]
    assert data["platform"] == "DISCORD"
    assert data["eventType"] == "reply_sent"
    assert data["serverId"] == "9"
    assert data["charCount"] == 100
    assert data["durationMs"] == 2500


@pytest.mark.asyncio
async def test_record_guild_joined_upserts():
    prisma_mock = MagicMock()
    prisma_mock.upsert = AsyncMock()
    with patch.object(bot_analytics.BotGuild, "prisma", return_value=prisma_mock):
        await bot_analytics.record_guild_joined(
            BotGuildInput(platform=Platform.DISCORD, server_id="5", name="Srv")
        )

    kwargs = prisma_mock.upsert.await_args.kwargs
    assert kwargs["where"]["platform_serverId"] == {
        "platform": "DISCORD",
        "serverId": "5",
    }
    assert kwargs["data"]["create"]["serverId"] == "5"
    assert kwargs["data"]["update"]["leftAt"] is None


@pytest.mark.asyncio
async def test_mark_guild_left_updates_active_rows():
    prisma_mock = MagicMock()
    prisma_mock.update_many = AsyncMock()
    with patch.object(bot_analytics.BotGuild, "prisma", return_value=prisma_mock):
        await bot_analytics.mark_guild_left(Platform.DISCORD, "5")

    where = prisma_mock.update_many.await_args.kwargs["where"]
    assert where == {"platform": "DISCORD", "serverId": "5", "leftAt": None}


@pytest.mark.asyncio
async def test_sync_guild_presence_marks_absent_as_left():
    prisma_mock = MagicMock()
    prisma_mock.upsert = AsyncMock()
    prisma_mock.update_many = AsyncMock()
    # Two rows currently joined; only "1" is still present.
    joined = [
        MagicMock(id="row1", serverId="1"),
        MagicMock(id="row2", serverId="2"),
    ]
    prisma_mock.find_many = AsyncMock(return_value=joined)

    with patch.object(bot_analytics.BotGuild, "prisma", return_value=prisma_mock):
        await bot_analytics.sync_guild_presence(
            Platform.DISCORD,
            [BotGuildInput(platform=Platform.DISCORD, server_id="1", name="A")],
        )

    prisma_mock.upsert.assert_awaited_once()
    prisma_mock.update_many.assert_awaited_once()
    where = prisma_mock.update_many.await_args.kwargs["where"]
    assert where == {"id": {"in": ["row2"]}}


@pytest.mark.asyncio
async def test_sync_guild_presence_noop_when_nothing_stale():
    prisma_mock = MagicMock()
    prisma_mock.upsert = AsyncMock()
    prisma_mock.update_many = AsyncMock()
    prisma_mock.find_many = AsyncMock(return_value=[MagicMock(id="row1", serverId="1")])

    with patch.object(bot_analytics.BotGuild, "prisma", return_value=prisma_mock):
        await bot_analytics.sync_guild_presence(
            Platform.DISCORD,
            [BotGuildInput(platform=Platform.DISCORD, server_id="1", name="A")],
        )

    prisma_mock.update_many.assert_not_awaited()
