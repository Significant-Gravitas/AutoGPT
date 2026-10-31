"""The backend sends the Claude Code CLI a user turn and nothing else."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.copilot.sdk.cli_io import _write_line, send_user_blocks, send_user_text


class _Transport:
    def __init__(self) -> None:
        self.lines: list[str] = []

    async def write(self, data: str) -> None:
        self.lines.append(data)


class _Client:
    def __init__(self) -> None:
        self._transport = _Transport()
        self.queried: list[tuple[Any, str]] = []

    async def query(self, prompt: Any, session_id: str = "default") -> None:
        self.queried.append((prompt, session_id))


@pytest.mark.asyncio
async def test_a_line_that_is_not_a_user_message_is_refused() -> None:
    client = _Client()

    with pytest.raises(ValueError, match="not a user message"):
        await _write_line(client, {"type": "not_a_user_turn"})  # type: ignore[arg-type]

    assert client._transport.lines == []


@pytest.mark.asyncio
async def test_a_turn_with_images_is_written_as_one_user_message() -> None:
    client = _Client()

    await send_user_blocks(
        client,  # type: ignore[arg-type]
        [{"type": "image", "source": {}}, {"type": "text", "text": "what is this?"}],
        "sess-1",
    )

    (line,) = client._transport.lines
    assert line.endswith("\n")
    sent = json.loads(line)
    assert sent["type"] == "user"
    assert sent["message"]["role"] == "user"
    assert sent["session_id"] == "sess-1"
    assert sent["client_composed"] is True


@pytest.mark.asyncio
async def test_text_goes_to_the_model_rather_than_the_clis_own_commands() -> None:
    """A message that opens with "/" would otherwise name a CLI command."""
    client = _Client()

    await send_user_text(client, "/compact", "sess-1")  # type: ignore[arg-type]

    assert client.queried == [(" /compact", "sess-1")]
    assert client._transport.lines == []


@pytest.mark.asyncio
async def test_an_unconnected_client_raises_rather_than_dropping_the_turn() -> None:
    client = MagicMock()
    client._transport = None

    with pytest.raises(RuntimeError, match="transport is not initialized"):
        await send_user_blocks(client, [{"type": "text", "text": "hi"}], "sess-1")
