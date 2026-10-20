from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.blocks.discord._auth import (
    TEST_BOT_CREDENTIALS,
    TEST_BOT_CREDENTIALS_INPUT,
)
from backend.blocks.discord.bot_blocks import SendDiscordMessageBlock


class _FakeChannel:
    def __init__(self, channel_id: int):
        self.id = channel_id
        self.send = AsyncMock(return_value=SimpleNamespace(id=123456789012345678))


class _FakeDiscordClient:
    def __init__(self, channel: _FakeChannel):
        self.user = "test-bot"
        self.guilds = []
        self.closed = False
        self._channel = channel
        self._on_ready = None

    def event(self, func):
        if func.__name__ == "on_ready":
            self._on_ready = func
        return func

    def get_channel(self, channel_id: int):
        return self._channel if channel_id == self._channel.id else None

    async def start(self, token: str) -> None:
        assert self._on_ready is not None
        await self._on_ready()

    async def close(self) -> None:
        self.closed = True


def _make_input(
    message_content: str, channel_name: str = "general"
) -> SendDiscordMessageBlock.Input:
    return SendDiscordMessageBlock.Input.model_validate(
        {
            "credentials": TEST_BOT_CREDENTIALS_INPUT,
            "channel_name": channel_name,
            "message_content": message_content,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("message_content", ["", "   ", "\n\t "])
async def test_blank_message_fails_before_connecting(mocker, message_content: str):
    client_cls = mocker.patch("backend.blocks.discord.bot_blocks.discord.Client")
    block = SendDiscordMessageBlock()
    outputs = []

    with pytest.raises(ValueError, match="cannot be empty"):
        async for output in block.run(
            _make_input(message_content), credentials=TEST_BOT_CREDENTIALS
        ):
            outputs.append(output)

    assert outputs == []
    client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_normal_message_sends_and_reports_success(mocker):
    channel = _FakeChannel(channel_id=987654321098765432)
    client = _FakeDiscordClient(channel)
    mocker.patch(
        "backend.blocks.discord.bot_blocks.discord.Client", return_value=client
    )
    block = SendDiscordMessageBlock()

    outputs = {
        name: value
        async for name, value in block.run(
            _make_input("Hello, Discord!", channel_name="987654321098765432"),
            credentials=TEST_BOT_CREDENTIALS,
        )
    }

    assert outputs == {
        "status": "Message sent",
        "message_id": "123456789012345678",
        "channel_id": "987654321098765432",
    }
    channel.send.assert_awaited_once_with("Hello, Discord!")
    assert client.closed
