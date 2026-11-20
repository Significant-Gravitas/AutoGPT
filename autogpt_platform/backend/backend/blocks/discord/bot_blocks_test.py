"""Regression tests for #15301 (send hangs when channel.send raises) and
#15302 (empty content reported as 'Message sent'), and for how much of a text
attachment ReadDiscordMessagesBlock adds to the message."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from aiohttp import web

from backend.blocks.discord.bot_blocks import (
    MAX_TEXT_ATTACHMENT_BYTES,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ReadDiscordMessagesBlock,
    SendDiscordMessageBlock,
)
from backend.util.request import Requests


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


class _FakeReadClient:
    """Mimics discord.Client once logged in: start() delivers one message to
    the on_message handler."""

    def __init__(self, *, message, **kwargs):
        self._message = message
        self._handlers = {}
        self.user = "bot"
        self.close = AsyncMock()

    def event(self, fn):
        self._handlers[fn.__name__] = fn
        return fn

    async def start(self, _token):
        await self._handlers["on_message"](self._message)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [b"the notes", b"a" * MAX_TEXT_ATTACHMENT_BYTES],
    ids=["short", "at-the-cap"],
)
async def test_read_adds_a_text_attachment_within_the_cap_whole(mocker, body: bytes):
    async def serve(_request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="text/plain", charset="utf-8")

    content = await _read_message_with_attachment(mocker, serve)

    assert content == _with_file(body.decode())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_limit",
    [None, MAX_TEXT_ATTACHMENT_BYTES],
    ids=["default-file-size-limit", "file-size-limit-at-the-cap"],
)
async def test_read_cuts_a_longer_text_attachment_at_the_cap(
    mocker, configured_limit: int | None
):
    async def serve_long_file(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={"Content-Type": "text/plain; charset=utf-8"}
        )
        await response.prepare(request)
        await response.write(b"a" * MAX_TEXT_ATTACHMENT_BYTES)
        try:
            while True:
                await response.write(b"b" * 65536)
        except ConnectionResetError:
            return response

    content = await _read_message_with_attachment(
        mocker, serve_long_file, configured_limit
    )

    assert content == _with_file(
        "a" * MAX_TEXT_ATTACHMENT_BYTES
        + f"\n[File truncated to its first {MAX_TEXT_ATTACHMENT_BYTES} bytes.]"
    )


def _with_file(file_content: str) -> str:
    return f"see the file\n\nFile from user: notes.txt\nContent: {file_content}"


async def _read_message_with_attachment(
    mocker, handler, configured_limit: int | None = None
) -> str:
    """Runs the block on one message whose notes.txt is served by ``handler``,
    through the real HTTP client, and returns its message_content.

    ``configured_limit`` stands in for a ``max_file_size_mb`` setting, which a
    limit the block passes itself overrides."""
    app = web.Application()
    app.router.add_get("/notes.txt", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    port = runner.addresses[0][1]

    message = SimpleNamespace(
        content="see the file",
        id=1,
        channel=SimpleNamespace(id=2, name="general"),
        author=SimpleNamespace(id=3, name="member"),
        attachments=[
            SimpleNamespace(
                filename="notes.txt", url=f"http://127.0.0.1:{port}/notes.txt"
            )
        ],
    )
    mocker.patch(
        "backend.blocks.discord.bot_blocks.discord.Client",
        side_effect=lambda **kw: _FakeReadClient(message=message, **kw),
    )
    # Only the loopback host is let through; the read itself is the real one.
    configured = (
        {} if configured_limit is None else {"max_response_bytes": configured_limit}
    )
    mocker.patch(
        "backend.blocks.discord.bot_blocks.Requests",
        side_effect=lambda **kw: Requests(
            trusted_origins=["127.0.0.1"], **(configured | kw)
        ),
    )

    block = ReadDiscordMessagesBlock()
    input_data = block.Input.model_validate({"credentials": TEST_CREDENTIALS_INPUT})
    try:
        async with asyncio.timeout(60):
            outputs = [
                output
                async for output in block.run(input_data, credentials=TEST_CREDENTIALS)
            ]
    finally:
        await runner.cleanup()
    return dict(outputs)["message_content"]
