"""langsmith's Claude Agent SDK tracing must survive the orchestrator's SDK mode.

The tracing wrapper around ``receive_response()`` keeps the
``claude.conversation`` run in a ContextVar that it sets while the first
message is fetched, and parents every ``claude.assistant.turn`` run on it.
``_execute_tools_sdk_mode`` fetches each message in its own task, so unless
every fetch runs in the same context, replies after the first message find no
parent and are silently dropped from the trace.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import claude_agent_sdk
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
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
from pydantic import SecretStr

from backend.blocks.orchestrator import OrchestratorBlock
from backend.data.model import APIKeyCredentials


class _ScriptedClient:
    """Stands in for ``ClaudeSDKClient`` and replays one agent turn."""

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self.options = options

    async def __aenter__(self) -> "_ScriptedClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self) -> AsyncGenerator[Any, None]:
        yield SystemMessage(subtype="init", data={})
        yield AssistantMessage(
            content=[ToolUseBlock(id="toolu_1", name="lookup", input={})],
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
async def test_sdk_mode_replies_get_assistant_turn_runs(
    traced_client: type[_ScriptedClient],
    posted_runs: list[RunTree],
) -> None:
    input_data = MagicMock(sys_prompt="", prompt="list the files")
    input_data.model.value = "claude-test"
    credentials = APIKeyCredentials(
        id="test-anthropic-id",
        provider="anthropic",
        api_key=SecretStr("mock-anthropic-key"),
        title="Mock Anthropic key",
        expires_at=None,
    )

    with (
        patch.object(claude_agent_sdk, "ClaudeSDKClient", traced_client),
        tracing_context(enabled=True),
    ):
        outputs = [
            output
            async for output in OrchestratorBlock()._execute_tools_sdk_mode(
                input_data=input_data,
                credentials=credentials,
                tool_functions=[],
                prompt=[{"role": "user", "content": "list the files"}],
                execution_params=MagicMock(graph_exec_id="graph-exec-1"),
                execution_processor=MagicMock(),
            )
        ]

    assert ("finished", "done") in outputs
    conversation = next(run for run in posted_runs if run.name == "claude.conversation")
    turns = [run for run in posted_runs if run.name == "claude.assistant.turn"]
    assert [turn.parent_run_id for turn in turns] == [
        conversation.id,
        conversation.id,
    ], "each assistant reply should get a turn run under the conversation"
