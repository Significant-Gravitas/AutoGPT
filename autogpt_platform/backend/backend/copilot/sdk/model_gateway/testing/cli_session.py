"""Drive the real Claude Code CLI against a running gateway, AutoPilot-style.

The CLI gets what an AutoPilot launch gets that matters here: built-in tools
off, one in-process MCP tool (``read_doc``), a PostToolUse hook that adds
context (so the CLI sends an in-conversation system note), the hardening env,
and the route's real window through ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` and
``CLAUDE_CODE_MAX_OUTPUT_TOKENS``. Each ``run`` is one CLI process; passing
the previous run's session id resumes it in a new one.
"""

import asyncio
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    ThinkingBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)
from claude_agent_sdk.types import (
    HookContext,
    HookInput,
    PostToolUseHookSpecificOutput,
    SyncHookJSONOutput,
)
from pydantic import BaseModel, Field

from backend.copilot.sdk.model_gateway.front import ModelGateway
from backend.copilot.sdk.model_gateway.testing.script import handbook_page

HOOK_NOTE = "GATEWAY-HOOK-NOTE: budget is fine; keep going."
SYSTEM_PROMPT = "You are a terse assistant. Use the handbook tool when asked."
_TURN_TIMEOUT_SECONDS = 120


class TurnRecord(BaseModel):
    session_id: str | None = None
    is_error: bool = True
    result: str = ""
    compactions: int = 0
    tool_uses: list[str] = Field(default_factory=list)
    signatures: list[str] = Field(default_factory=list)


class CliSession:
    def __init__(
        self,
        *,
        cli_path: Path,
        gateway: ModelGateway,
        root: Path,
        window_tokens: int,
        output_tokens: int,
        page_chars: int = 2_000,
    ) -> None:
        self._cli_path = cli_path
        self._gateway = gateway
        self._workdir = root / "work"
        self._config_dir = root / "claude-config"
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._window_tokens = window_tokens
        self._output_tokens = output_tokens
        self._page_chars = page_chars
        self.stderr: list[str] = []

    async def run(
        self, prompts: list[str], *, resume: str | None = None
    ) -> list[TurnRecord]:
        """Every prompt as one turn of a single CLI process."""
        records: list[TurnRecord] = []
        async with ClaudeSDKClient(options=self._options(resume)) as client:
            for prompt in prompts:
                await client.query(prompt)
                record = TurnRecord()
                async with asyncio.timeout(_TURN_TIMEOUT_SECONDS):
                    async for message in client.receive_response():
                        _note(record, message)
                records.append(record)
        return records

    def env(self) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": self._gateway.base_url,
            "ANTHROPIC_AUTH_TOKEN": self._gateway.auth_token,
            "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "",
            "CLAUDE_CONFIG_DIR": str(self._config_dir),
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(self._window_tokens),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(self._output_tokens),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_ENABLE_TASKS": "0",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            "CLAUDE_CODE_MAX_RETRIES": "2",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "ANTHROPIC_CUSTOM_HEADERS": "Accept-Encoding: identity",
            "NO_PROXY": "127.0.0.1,localhost",
        }

    def _options(self, resume: str | None) -> ClaudeAgentOptions:
        page_chars = self._page_chars

        @tool("read_doc", "Read one page of the handbook.", {"page": int})
        async def read_doc(arguments: dict) -> dict:
            page = int(arguments.get("page") or 1)
            return {
                "content": [{"type": "text", "text": handbook_page(page, page_chars)}]
            }

        return ClaudeAgentOptions(
            cli_path=str(self._cli_path),
            model=self._gateway.route.model,
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            mcp_servers={
                "handbook": create_sdk_mcp_server("handbook", "1.0.0", [read_doc])
            },
            allowed_tools=["mcp__handbook__read_doc"],
            hooks={"PostToolUse": [HookMatcher(matcher=None, hooks=[_post_tool_note])]},
            max_turns=12,
            cwd=str(self._workdir),
            env=self.env(),
            resume=resume,
            setting_sources=[],
            stderr=self.stderr.append,
        )


async def _post_tool_note(
    input_data: HookInput, tool_use_id: str | None, context: HookContext
) -> SyncHookJSONOutput:
    return SyncHookJSONOutput(
        hookSpecificOutput=PostToolUseHookSpecificOutput(
            hookEventName="PostToolUse", additionalContext=HOOK_NOTE
        )
    )


def _note(record: TurnRecord, message: object) -> None:
    if isinstance(message, SystemMessage) and message.subtype == "compact_boundary":
        record.compactions += 1
    elif isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                record.tool_uses.append(block.name)
            elif isinstance(block, ThinkingBlock):
                record.signatures.append(block.signature)
    elif isinstance(message, ResultMessage):
        record.session_id = message.session_id
        record.is_error = message.is_error
        record.result = message.result or ""
