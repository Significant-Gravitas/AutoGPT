"""The only place the backend puts a line on the Claude Code CLI's stdin.

That stream carries the SDK's control protocol as well as the conversation, so
everything sent here is one user turn. `architecture_test.py` fails any other
send to the CLI in a backend module that imports `claude_agent_sdk`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from backend.copilot.sdk.env import cli_prompt_text, cli_user_message

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeSDKClient

DEFAULT_SESSION_ID = "default"


async def send_user_text(
    client: ClaudeSDKClient, text: str, session_id: str = DEFAULT_SESSION_ID
) -> None:
    """Send *text* to the model as one user turn."""
    await client.query(cli_prompt_text(text), session_id=session_id)


async def send_user_blocks(
    client: ClaudeSDKClient,
    blocks: list[dict[str, Any]],
    session_id: str = DEFAULT_SESSION_ID,
) -> None:
    """Send *blocks* — text and image content blocks — as one user turn.

    The SDK wraps only a string prompt in a user message, so a turn carrying
    images is built here.
    """
    await _write_line(client, cli_user_message(blocks, session_id))


async def _write_line(client: ClaudeSDKClient, line: dict[str, Any]) -> None:
    if line.get("type") != "user" or not isinstance(line.get("message"), dict):
        raise ValueError(
            "Refusing to write a line to the Claude Code CLI that is not a user "
            f"message (type={line.get('type')!r})."
        )
    transport = client._transport
    if transport is None:
        raise RuntimeError("ClaudeSDKClient transport is not initialized")
    await transport.write(json.dumps(line) + "\n")
