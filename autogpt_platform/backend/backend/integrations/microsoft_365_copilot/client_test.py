import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import aiohttp
import pytest


async def _chunks() -> AsyncIterator[bytes]:
    yield b'data: {"id":"conversation","messages":[{"id":"answer","text":"Hello"}]}\r\n'
    yield b'id:1\r\n\r\ndata: {"id":"conversation","messages":[]}\n\n'
    yield (
        b'data: {"id":"conversation","messages":['
        b'{"id":"question","text":"Hi"},'
        b'{"id":"answer","text":"Hello world"}]}\n\n'
    )
    yield b"data: [DONE]\n\n"


async def _split_utf8_chunks() -> AsyncIterator[bytes]:
    payload = (
        'data: {"id":"conversation","messages":[{"id":"answer","text":"Hello 👋"}]}\n\n'
    ).encode()
    emoji_start = payload.index("👋".encode())
    yield payload[: emoji_start + 1]
    yield payload[emoji_start + 1 :]


@pytest.mark.asyncio
async def test_stream_parser_emits_only_new_text_from_full_snapshots() -> None:
    from backend.integrations.microsoft_365_copilot.client import (
        iter_copilot_text_deltas,
    )

    deltas = [delta async for delta in iter_copilot_text_deltas(_chunks())]

    assert deltas == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_parser_handles_utf8_characters_split_between_chunks() -> None:
    from backend.integrations.microsoft_365_copilot.client import (
        iter_copilot_text_deltas,
    )

    deltas = [delta async for delta in iter_copilot_text_deltas(_split_utf8_chunks())]

    assert deltas == ["Hello 👋"]


def test_chat_request_includes_timezone_context_and_web_grounding() -> None:
    from backend.integrations.microsoft_365_copilot.client import build_chat_request

    request = build_chat_request(
        "Summarize the launch discussion",
        timezone="America/Chicago",
        additional_context=["Use the application launch brief."],
        web_enabled=False,
    )

    assert request == {
        "message": {"text": "Summarize the launch discussion"},
        "locationHint": {"timeZone": "America/Chicago"},
        "additionalContext": [{"text": "Use the application launch brief."}],
        "contextualResources": {"webContext": {"isWebEnabled": False}},
    }


def test_client_timeout_does_not_cap_total_stream_lifetime() -> None:
    from backend.integrations.microsoft_365_copilot.client import (
        Microsoft365CopilotClient,
    )

    client = Microsoft365CopilotClient("token", timeout_seconds=45)

    assert client._timeout.total is None
    assert client._timeout.connect == 30
    assert client._timeout.sock_connect == 30
    assert client._timeout.sock_read == 45


@pytest.mark.asyncio
async def test_create_conversation_normalizes_network_errors() -> None:
    from backend.integrations.microsoft_365_copilot.client import (
        Microsoft365CopilotClient,
        Microsoft365CopilotError,
    )

    client = Microsoft365CopilotClient("token")
    client._session = MagicMock()
    client._session.post.side_effect = aiohttp.ClientConnectionError("private detail")

    with pytest.raises(Microsoft365CopilotError, match="could not create") as raised:
        await client.create_conversation()

    assert "private detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_stream_chat_normalizes_idle_timeout() -> None:
    from backend.integrations.microsoft_365_copilot.client import (
        Microsoft365CopilotClient,
        Microsoft365CopilotError,
    )

    client = Microsoft365CopilotClient("token")
    client._session = MagicMock()
    client._session.post.side_effect = asyncio.TimeoutError("private detail")

    with pytest.raises(Microsoft365CopilotError, match="could not continue") as raised:
        async for _ in client.stream_chat(
            "conversation",
            "hello",
            timezone="UTC",
        ):
            pass

    assert "private detail" not in str(raised.value)
