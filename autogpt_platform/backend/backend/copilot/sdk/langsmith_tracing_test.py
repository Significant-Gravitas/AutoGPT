"""langsmith's Claude Agent SDK tracing must survive ``_iter_sdk_messages``.

The tracing wrapper around ``receive_response()`` keeps the
``claude.conversation`` run in a ContextVar that it sets while the first
message is fetched, and parents every ``claude.assistant.turn`` run on it.
``_iter_sdk_messages`` fetches each message in its own task, so unless every
fetch runs in the same context, main-agent replies after the first message
find no parent and are silently dropped from the trace.
"""

from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import patch

import claude_agent_sdk
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from langsmith import tracing_context
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk
from langsmith.run_trees import RunTree

from .service import _iter_sdk_messages


class _ScriptedClient:
    """Stands in for ``ClaudeSDKClient`` and replays one main-agent turn."""

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self.options = options

    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self) -> AsyncGenerator[Any, None]:
        yield SystemMessage(subtype="init", data={})
        yield AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="bash_exec", input={})],
            model="claude-test",
            message_id="msg_1",
        )
        yield UserMessage(content=[ToolResultBlock(tool_use_id="toolu_1")])
        yield AssistantMessage(
            content=[TextBlock(text="done")], model="claude-test", message_id="msg_2"
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=2,
            session_id="sess-1",
        )


@pytest.fixture
def traced_client() -> type[_ScriptedClient]:
    """Wrap the scripted client exactly as production wraps ``ClaudeSDKClient``.

    ``configure_claude_agent_sdk`` patches whichever class
    ``claude_agent_sdk.ClaudeSDKClient`` names, so pointing that name at the
    scripted client for the call wraps it through the public API.  The real
    tool class and the global trace config are left untouched.
    """
    with (
        patch.object(claude_agent_sdk, "ClaudeSDKClient", _ScriptedClient),
        patch.object(claude_agent_sdk, "SdkMcpTool", None),
        patch("langsmith.integrations.claude_agent_sdk.set_tracing_config"),
    ):
        assert configure_claude_agent_sdk()
    return _ScriptedClient


@pytest.fixture
def posted_runs(monkeypatch: pytest.MonkeyPatch) -> list[RunTree]:
    """Record every run the tracing wrapper posts instead of exporting it."""
    posted: list[RunTree] = []
    monkeypatch.setattr(RunTree, "post", lambda self, *_, **__: posted.append(self))
    monkeypatch.setattr(RunTree, "patch", lambda self, *_, **__: None)
    return posted


@pytest.mark.asyncio
async def test_main_agent_replies_get_assistant_turn_runs(
    traced_client: type[_ScriptedClient],
    posted_runs: list[RunTree],
) -> None:
    client = traced_client(options=ClaudeAgentOptions())
    await client.query("list the files")

    with tracing_context(enabled=True):
        received = [
            msg
            async for msg in _iter_sdk_messages(cast(ClaudeSDKClient, client))
            if msg is not None
        ]

    assert [type(msg).__name__ for msg in received] == [
        "SystemMessage",
        "AssistantMessage",
        "UserMessage",
        "AssistantMessage",
        "ResultMessage",
    ]
    conversation = next(run for run in posted_runs if run.name == "claude.conversation")
    turns = [run for run in posted_runs if run.name == "claude.assistant.turn"]
    assert [turn.parent_run_id for turn in turns] == [
        conversation.id,
        conversation.id,
    ], "each main-agent reply should get a turn run under the conversation"
