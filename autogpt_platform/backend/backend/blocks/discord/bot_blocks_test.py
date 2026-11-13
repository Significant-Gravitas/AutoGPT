"""Regression tests for #15301 (send hangs when channel.send raises) and
#15302 (empty content reported as 'Message sent')."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from pydantic import SecretStr

from backend.blocks.discord.bot_blocks import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    CreateDiscordThreadBlock,
    DiscordChannelInfoBlock,
    DiscordUserInfoBlock,
    ReadDiscordMessagesBlock,
    ReplyToDiscordMessageBlock,
    SendDiscordDMBlock,
    SendDiscordEmbedBlock,
    SendDiscordFileBlock,
    SendDiscordMessageBlock,
    ThreadArchiveDuration,
)
from backend.data.execution import ExecutionContext
from backend.util.type import MediaFileType


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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        # Like discord.py: exiting closes the client unless a handler already did.
        if not self._closed.is_set():
            await self.close()

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


@pytest.mark.asyncio
async def test_partial_send_reports_the_chunks_already_delivered(mocker):
    channel = MagicMock()
    channel.id = 987654321098765432
    channel.send = AsyncMock(
        side_effect=[
            MagicMock(id=111),
            discord.HTTPException(MagicMock(status=500, reason="Server Error"), "boom"),
        ]
    )
    _patch_client(mocker, channel)

    out = dict(await _run(SendDiscordMessageBlock(), "x" * 2500))

    assert out["status"].startswith("Error sending message")
    assert "earlier chunks were already sent" in out["status"]
    assert out["message_id"] == "111"
    assert out["channel_id"] == "987654321098765432"
    assert channel.send.await_count == 2
    _FakeClient.instances[0].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_later_chunk_still_closes_the_client(mocker):
    channel = MagicMock()
    channel.id = 987654321098765432
    channel.send = AsyncMock(side_effect=[MagicMock(id=111), asyncio.CancelledError()])
    _patch_client(mocker, channel)
    block = SendDiscordMessageBlock()

    result = await block.send_message(
        token="t",
        channel_name="987654321098765432",
        server_name=None,
        message_content="x" * 2500,
    )

    _FakeClient.instances[0].close.assert_awaited_once()
    assert result["message_id"] == "111"
    assert "earlier chunks were already sent" in result["status"]


_TOKEN = "refused-token"

# Every discord.Client the module creates, reached through the method that starts it.
_LOGINS = [
    pytest.param(
        lambda: ReadDiscordMessagesBlock().run_bot(SecretStr(_TOKEN)), id="read"
    ),
    pytest.param(
        lambda: SendDiscordMessageBlock().send_message(_TOKEN, "1", None, "hi"),
        id="send_message",
    ),
    pytest.param(lambda: SendDiscordDMBlock().send_dm(_TOKEN, "1", "hi"), id="dm"),
    pytest.param(
        lambda: SendDiscordEmbedBlock().send_embed(_TOKEN, "1", None, {}), id="embed"
    ),
    pytest.param(
        lambda: SendDiscordFileBlock().send_file(
            _TOKEN, "1", None, MediaFileType("a.txt"), "a.txt", "", ExecutionContext()
        ),
        id="file",
    ),
    pytest.param(
        lambda: ReplyToDiscordMessageBlock().send_reply(_TOKEN, "1", "2", "hi", False),
        id="reply",
    ),
    pytest.param(
        lambda: DiscordUserInfoBlock().get_user_info(_TOKEN, "1"), id="user_info"
    ),
    pytest.param(
        lambda: DiscordChannelInfoBlock().get_channel_info(_TOKEN, "1", None),
        id="channel_info",
    ),
    pytest.param(
        lambda: CreateDiscordThreadBlock().create_thread(
            _TOKEN, "1", None, "t", False, ThreadArchiveDuration.ONE_HOUR, "hi"
        ),
        id="thread",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("login", _LOGINS)
async def test_a_refused_token_closes_the_client(mocker, login):
    # discord.py opens its aiohttp session before the login request, and
    # LoginFailure leaves it open unless the caller closes the client.
    real_client = discord.Client
    clients: list[discord.Client] = []

    def record(**kwargs):
        clients.append(real_client(**kwargs))
        return clients[-1]

    mocker.patch("backend.blocks.discord.bot_blocks.discord.Client", side_effect=record)
    request = mocker.patch.object(
        discord.http.HTTPClient,
        "request",
        side_effect=discord.HTTPException(
            MagicMock(status=401, reason="Unauthorized"), "401: Unauthorized"
        ),
    )

    with pytest.raises(discord.LoginFailure):
        await login()

    request.assert_awaited_once()
    assert len(clients) == 1
    assert clients[0].is_closed()
