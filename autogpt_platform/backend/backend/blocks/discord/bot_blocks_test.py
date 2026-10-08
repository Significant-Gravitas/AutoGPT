"""Regression tests for #15301 (send hangs when channel.send raises) and
#15302 (empty content reported as 'Message sent')."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from backend.blocks.discord.bot_blocks import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    SendDiscordMessageBlock,
)


class _FakeClient:
    """Mimics discord.Client: start() dispatches on_ready as a task whose
    exceptions are only logged, then runs until close() is called."""

    instances: list["_FakeClient"] = []

    def __init__(self, *, channel, **kwargs):
        self._channel = channel
        self._on_ready = None
        self._closed = asyncio.Event()
        self.close = AsyncMock(side_effect=self._closed.set)
        self.user = "bot"
        self.guilds = []
        _FakeClient.instances.append(self)

    def event(self, fn):
        self._on_ready = fn
        return fn

    def get_channel(self, _channel_id):
        return self._channel

    async def start(self, _token):
        assert self._on_ready is not None
        task = asyncio.create_task(self._on_ready())
        await asyncio.wait([task])
        # Like discord.py: a handler exception goes to on_error, not here.
        if not task.cancelled():
            task.exception()
        # Real clients run until closed; bound it so a hang fails the test.
        await asyncio.wait_for(self._closed.wait(), timeout=2)


def _patch_client(mocker, channel):
    _FakeClient.instances.clear()
    mocker.patch(
        "backend.blocks.discord.bot_blocks.discord.Client",
        side_effect=lambda **kw: _FakeClient(channel=channel, **kw),
    )


async def _run(block, content: str):
    input_data = block.Input(
        channel_name="987654321098765432",
        message_content=content,
        credentials=TEST_CREDENTIALS_INPUT,  # type: ignore[arg-type]
    )
    return [out async for out in block.run(input_data, credentials=TEST_CREDENTIALS)]


@pytest.mark.asyncio
async def test_send_returns_error_and_closes_when_send_is_forbidden(mocker):
    channel = MagicMock()
    channel.id = 987654321098765432
    channel.send = AsyncMock(
        side_effect=discord.Forbidden(
            MagicMock(status=403, reason="Forbidden"), "Missing Permissions"
        )
    )
    _patch_client(mocker, channel)

    out = await _run(SendDiscordMessageBlock(), "hello")

    status = dict(out)["status"]
    assert status.startswith("Error")
    assert "Missing Permissions" in status
    _FakeClient.instances[0].close.assert_awaited()


@pytest.mark.asyncio
async def test_send_success_reports_message_id(mocker):
    channel = MagicMock()
    channel.id = 987654321098765432
    channel.send = AsyncMock(return_value=MagicMock(id=123456789012345678))
    _patch_client(mocker, channel)

    out = await _run(SendDiscordMessageBlock(), "hello")

    assert out == [
        ("status", "Message sent"),
        ("message_id", "123456789012345678"),
        ("channel_id", "987654321098765432"),
    ]
    _FakeClient.instances[0].close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   \n\t"])
async def test_send_rejects_empty_content_without_logging_in(mocker, content):
    client_cls = mocker.patch("backend.blocks.discord.bot_blocks.discord.Client")

    with pytest.raises(ValueError, match="Message content is empty"):
        await _run(SendDiscordMessageBlock(), content)

    client_cls.assert_not_called()
