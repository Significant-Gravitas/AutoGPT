import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)
from openai_codex.generated.v2_all import AgentMessageDeltaNotification
from openai_codex.models import Notification

from backend.copilot.sdk.codex_compat_gateway import CodexAnthropicGateway
from backend.copilot.sdk.env import build_sdk_env
from backend.integrations.codex.models import (
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexInvocationResult,
    CodexTokenUsage,
)
from backend.integrations.credential_lease import CredentialLease


class _ClaudeHarnessAgentSession:
    def __init__(self) -> None:
        self.tool_result: CodexDynamicToolResult | None = None

    async def invoke(
        self,
        request,
        dynamic_tools,
        tool_handler,
        event_handler=None,
    ) -> CodexInvocationResult:
        echo_tool = next(
            tool for tool in dynamic_tools if tool.description == "Echo a value"
        )
        assert not echo_tool.name.startswith("mcp__")
        self.tool_result = await tool_handler(
            CodexDynamicToolCall(
                thread_id="thread-cli",
                turn_id="turn-cli",
                call_id="call-cli",
                tool=echo_tool.name,
                arguments={"value": "ping"},
            )
        )
        assert event_handler is not None
        await event_handler(
            Notification(
                method="item/agentMessage/delta",
                payload=AgentMessageDeltaNotification(
                    delta="Claude harness round trip complete",
                    itemId="item-cli",
                    threadId="thread-cli",
                    turnId="turn-cli",
                ),
            )
        )
        return CodexInvocationResult(
            response_id="response-cli",
            final_response="Claude harness round trip complete",
            status="completed",
            usage=CodexTokenUsage(
                input_tokens=20,
                cached_input_tokens=0,
                output_tokens=5,
                reasoning_output_tokens=0,
                total_tokens=25,
            ),
        )


class _ClaudeHarnessTransport:
    def __init__(self, session: _ClaudeHarnessAgentSession) -> None:
        self.session = session

    @asynccontextmanager
    async def agent_session(
        self,
        _lease: CredentialLease,
    ) -> AsyncIterator[_ClaudeHarnessAgentSession]:
        yield self.session


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bundled_claude_cli_executes_mcp_tool_through_codex_gateway(
    tmp_path,
) -> None:
    tool_called = asyncio.Event()
    stderr_lines: list[str] = []

    @tool("echo", "Echo a value", {"value": str})
    async def echo(arguments):
        tool_called.set()
        return {"content": [{"type": "text", "text": f"echo:{arguments['value']}"}]}

    mcp_server = create_sdk_mcp_server(
        name="codex-conformance",
        tools=[echo],
    )
    agent_session = _ClaudeHarnessAgentSession()
    gateway = CodexAnthropicGateway(
        credential_lease=cast(CredentialLease, object()),
        model="gpt-5.6-terra",
        effort="medium",
        transport=_ClaudeHarnessTransport(agent_session),
    )

    async with gateway:
        options = ClaudeAgentOptions(
            model="gpt-5.6-terra",
            system_prompt="Use the supplied echo tool once.",
            mcp_servers={"codex-conformance": mcp_server},
            allowed_tools=["mcp__codex-conformance__echo"],
            disallowed_tools=[
                "Bash",
                "Edit",
                "Glob",
                "Grep",
                "Read",
                "Task",
                "WebFetch",
                "WebSearch",
                "Write",
            ],
            cwd=str(tmp_path),
            max_turns=3,
            stderr=stderr_lines.append,
            env=build_sdk_env(
                session_id="codex-cli-conformance",
                sdk_cwd=str(tmp_path),
                model="gpt-5.6-terra",
                codex_gateway_url=gateway.base_url,
                codex_gateway_token=gateway.auth_token,
            ),
        )
        received = []
        async with ClaudeSDKClient(options=options) as client:
            await client.query("Call echo with ping, then report completion.")
            try:
                async with asyncio.timeout(45):
                    async for message in client.receive_response():
                        received.append(message)
            except TimeoutError as exc:
                raise AssertionError(
                    "Claude CLI did not finish the Codex gateway round trip; "
                    f"stderr={stderr_lines!r}"
                ) from exc

    assert tool_called.is_set()
    assert agent_session.tool_result is not None
    assert agent_session.tool_result.success
    assert "echo:ping" in agent_session.tool_result.content
    assert any(
        isinstance(message, ResultMessage) and not message.is_error
        for message in received
    )
    assert gateway.result is not None
    assert gateway.result.usage is not None
    assert gateway.result.usage.total_tokens == 25
