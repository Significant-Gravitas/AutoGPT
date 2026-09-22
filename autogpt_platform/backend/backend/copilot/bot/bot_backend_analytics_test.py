"""Tests for BotBackend fire-and-forget analytics methods."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.copilot.bot.bot_backend import BotBackend
from backend.platform_linking.models import BotEventInput, BotGuildInput, Platform


def _backend() -> tuple[BotBackend, AsyncMock]:
    backend = BotBackend.__new__(BotBackend)
    client = AsyncMock()
    backend._client = client
    backend._analytics_tasks = set()
    return backend, client


async def _drain(backend: BotBackend) -> None:
    if backend._analytics_tasks:
        await asyncio.gather(*backend._analytics_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_track_event_records_bot_event():
    backend, client = _backend()
    backend.track_event(
        platform="discord",
        event_type="message_received",
        server_id="42",
        channel_type="thread",
        char_count=12,
    )
    await _drain(backend)

    client.record_bot_event.assert_awaited_once()
    event = client.record_bot_event.await_args.kwargs["event"]
    assert isinstance(event, BotEventInput)
    assert event.platform == Platform.DISCORD
    assert event.event_type == "message_received"
    assert event.server_id == "42"
    assert event.channel_type == "thread"
    assert event.char_count == 12


@pytest.mark.asyncio
async def test_track_guild_joined_and_left():
    backend, client = _backend()
    backend.track_guild_joined("discord", "7", "Cool Server")
    backend.track_guild_left("discord", "7")
    await _drain(backend)

    guild = client.record_guild_joined.await_args.kwargs["guild"]
    assert isinstance(guild, BotGuildInput)
    assert guild.server_id == "7"
    assert guild.name == "Cool Server"
    client.mark_guild_left.assert_awaited_once_with(
        platform=Platform.DISCORD, server_id="7"
    )


@pytest.mark.asyncio
async def test_sync_guilds_maps_all_servers():
    backend, client = _backend()
    backend.sync_guilds("discord", [("1", "A"), ("2", None)])
    await _drain(backend)

    kwargs = client.sync_guild_presence.await_args.kwargs
    assert kwargs["platform"] == Platform.DISCORD
    guilds = kwargs["guilds"]
    assert [g.server_id for g in guilds] == ["1", "2"]
    assert guilds[1].name is None


@pytest.mark.asyncio
async def test_analytics_failure_is_swallowed():
    backend, client = _backend()
    client.record_bot_event.side_effect = RuntimeError("rpc down")
    backend.track_event(platform="discord", event_type="message_received")
    # A failing write must never propagate into the caller's reply path.
    await _drain(backend)
    assert backend._analytics_tasks == set()


@pytest.mark.asyncio
async def test_bad_platform_string_is_swallowed():
    # Platform(...) raises ValueError on an unknown string. That must fail the
    # background task — never the synchronous caller in the reply path.
    backend, client = _backend()
    backend.track_event(platform="myspace", event_type="message_received")
    await _drain(backend)
    assert backend._analytics_tasks == set()
    # The bad platform must short-circuit before the RPC; the client must not
    # have been called.
    client.record_bot_event.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_analytics_task_is_silent():
    # Cancelled tasks during shutdown must not raise CancelledError from the
    # done-callback (which would print as an unhandled exception by asyncio).
    backend, client = _backend()

    async def hang() -> None:
        await asyncio.sleep(3600)

    client.record_bot_event.side_effect = lambda **_: hang()
    backend.track_event(platform="discord", event_type="message_received")
    task = next(iter(backend._analytics_tasks))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend._analytics_tasks == set()
