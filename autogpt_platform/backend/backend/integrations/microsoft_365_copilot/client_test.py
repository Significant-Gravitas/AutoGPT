from collections.abc import AsyncIterator

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
