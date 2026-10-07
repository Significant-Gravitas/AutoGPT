"""A sub-agent's result must reach the turn that started it.

Claude Code 2.1.198 made sub-agents run in the background by default: the
Agent/Task tool answers at once with ``{"isAsync": true, "status":
"async_launched"}`` and the sub-agent's report only arrives later, as a task
notification that starts a new model turn.  A copilot turn ends at the first
``ResultMessage`` and closes the CLI (``_run_stream_attempt`` →
``_safe_close_sdk_client``), so the background sub-agent is stopped before its
report reaches the reply.

This drives the bundled CLI through ``ClaudeSDKClient`` with the environment
``build_sdk_env()`` builds, against a scripted fake Anthropic API, and follows
the same lifecycle as ``_run_stream_attempt``: open the client, send the query,
read ``receive_response()`` up to the ``ResultMessage``, close.  The fake main
model calls the Agent tool once and then writes its reply, quoting the
sub-agent's report only if the tool result carried it.  No API key needed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from aiohttp import web
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    UserMessage,
)

from backend.copilot.config import ChatConfig
from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path
from backend.copilot.sdk.env import build_sdk_env

_MAIN_USER_MARKER = "MAIN-USER-7f3a"
_SUBAGENT_PROMPT = "SUBAGENT-PROMPT-9c41: report the magic word."
_SUBAGENT_REPORT = "SUBAGENT-REPORT-2b8e: the magic word is pelican."
_REPLY_WITH_REPORT = "MAIN-REPLY-WITH-REPORT"
_REPLY_WITHOUT_REPORT = "MAIN-REPLY-WITHOUT-REPORT"

# Long enough that a background sub-agent is still running when the main
# model's reply ends the turn, short enough to keep the synchronous run quick.
_SUBAGENT_WORK_SECONDS = 2.0


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(
        f"event: {evt['type']}\ndata: {json.dumps(evt)}\n\n" for evt in events
    ).encode()


def _message_start(msg_id: str) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-test",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


def _text_stream(msg_id: str, text: str) -> bytes:
    return _sse(
        [
            _message_start(msg_id),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    )


def _tool_use_stream(msg_id: str, tool_id: str, name: str, args: dict) -> bytes:
    return _sse(
        [
            _message_start(msg_id),
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    )


def _texts(node: Any) -> list[str]:
    """Every string under *node*, so markers are found however the CLI nests
    content (plain strings, text blocks, tool_result content lists)."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [t for v in node.values() for t in _texts(v)]
    if isinstance(node, list):
        return [t for v in node for t in _texts(v)]
    return []


def _tool_results(messages: list[dict[str, Any]]) -> list[str]:
    return [
        " ".join(_texts(block.get("content")))
        for msg in messages
        if msg.get("role") == "user" and isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


class _ScriptedAnthropic:
    """Fake Messages API: a main model that delegates once, and a sub-agent
    that takes a while to report."""

    def __init__(self) -> None:
        self.agent_tool_name: str | None = None
        self.subagent_started = asyncio.Event()
        self.subagent_report_sent = False
        self.main_saw_tool_result: str | None = None
        self._ids = 0

    def _next_id(self) -> str:
        self._ids += 1
        return f"msg_{self._ids}"

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = json.loads(await request.text())
        messages: list[dict[str, Any]] = body.get("messages", [])
        first_user = " ".join(_texts(messages[0])) if messages else ""

        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)

        if _SUBAGENT_PROMPT in first_user:
            self.subagent_started.set()
            await asyncio.sleep(_SUBAGENT_WORK_SECONDS)
            await response.write(_text_stream(self._next_id(), _SUBAGENT_REPORT))
            self.subagent_report_sent = True
        elif _MAIN_USER_MARKER in first_user:
            await response.write(self._main_turn(body, messages))
        else:
            # Side calls the CLI makes on its own (titles, summaries).
            await response.write(_text_stream(self._next_id(), "ok"))
        await response.write_eof()
        return response

    def _main_turn(self, body: dict[str, Any], messages: list[dict]) -> bytes:
        results = _tool_results(messages)
        if not results:
            names = {t.get("name") for t in body.get("tools", [])}
            self.agent_tool_name = next(
                (n for n in ("Agent", "Task") if n in names), None
            )
            if self.agent_tool_name is None:
                return _text_stream(self._next_id(), "no Agent tool offered")
            return _tool_use_stream(
                self._next_id(),
                "toolu_subagent_1",
                self.agent_tool_name,
                {
                    "description": "Find the magic word",
                    "prompt": _SUBAGENT_PROMPT,
                    "subagent_type": "general-purpose",
                },
            )
        self.main_saw_tool_result = results[-1]
        reply = (
            f"{_REPLY_WITH_REPORT}: {_SUBAGENT_REPORT}"
            if _SUBAGENT_REPORT in results[-1]
            else _REPLY_WITHOUT_REPORT
        )
        return _text_stream(self._next_id(), reply)


async def _not_found(_request: web.Request) -> web.Response:
    return web.Response(status=404)


async def _start(server: _ScriptedAnthropic) -> tuple[web.AppRunner, int]:
    app = web.Application()
    app.router.add_post("/v1/messages", server.handle)
    app.router.add_route("*", "/{tail:.*}", _not_found)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port: int = site._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]
    return runner, port


def _direct_anthropic_config() -> ChatConfig:
    return ChatConfig(
        use_claude_code_subscription=False,
        use_openrouter=False,
        api_key=None,
        base_url=None,
        thinking_standard_model="anthropic/claude-sonnet-4-6",
        thinking_advanced_model="anthropic/claude-opus-4-7",
        claude_agent_autocompact_pct_override=50,
        claude_agent_context_window=200_000,
        aux_api_key="or-aux-key",
    )


async def run_copilot_style_turn(
    cli_path: Path, workdir: Path
) -> tuple[_ScriptedAnthropic, list[Any], list[str]]:
    """One turn with the copilot's client lifecycle; returns what the fake
    API saw, the SDK messages up to the ``ResultMessage``, and CLI stderr."""
    server = _ScriptedAnthropic()
    runner, port = await _start(server)
    stderr_lines: list[str] = []
    received: list[Any] = []
    try:
        with patch("backend.copilot.sdk.env.config", _direct_anthropic_config()):
            env = build_sdk_env(session_id="subagent-turn-end", sdk_cwd=str(workdir))
        env.update(
            {
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                "ANTHROPIC_API_KEY": "sk-test-fake-key-not-real",
                "CLAUDE_CONFIG_DIR": str(workdir / ".claude-config"),
            }
        )
        options = ClaudeAgentOptions(
            cli_path=str(cli_path),
            system_prompt="You are a test main agent.",
            setting_sources=[],
            allowed_tools=["Task", "Agent"],
            cwd=str(workdir),
            max_turns=5,
            env=env,
            stderr=stderr_lines.append,
        )
        sdk_client = ClaudeSDKClient(options=options)
        client = await sdk_client.__aenter__()
        try:
            await client.query(f"{_MAIN_USER_MARKER} find the magic word")
            async with asyncio.timeout(90):
                async for message in client.receive_response():
                    received.append(message)
        finally:
            await sdk_client.__aexit__(None, None, None)
    finally:
        await runner.cleanup()
    return server, received, stderr_lines


def _reply_text(received: list[Any]) -> str:
    return " ".join(
        block.text
        for msg in received
        if isinstance(msg, AssistantMessage)
        for block in msg.content
        if isinstance(block, TextBlock)
    )


def _agent_tool_result(received: list[Any]) -> str:
    return " ".join(
        " ".join(_texts(block.content))
        for msg in received
        if isinstance(msg, UserMessage) and isinstance(msg.content, list)
        for block in msg.content
        if isinstance(block, ToolResultBlock)
        and block.tool_use_id == "toolu_subagent_1"
    )


@pytest.mark.asyncio
async def test_subagent_report_reaches_the_turn_that_started_it(tmp_path):
    cli_path = _resolve_cli_path()
    if cli_path is None or not cli_path.is_file():
        pytest.skip("No Claude Code CLI binary available")

    server, received, stderr = await run_copilot_style_turn(cli_path, tmp_path)

    assert server.agent_tool_name, (
        "The CLI offered no Agent/Task tool, so nothing was delegated; "
        f"stderr tail: {stderr[-20:]!r}"
    )
    assert any(isinstance(m, ResultMessage) for m in received)
    assert server.subagent_started.is_set(), "The sub-agent never made a call"
    tool_result = _agent_tool_result(received)
    assert _SUBAGENT_REPORT in tool_result, (
        "The Agent tool did not return the sub-agent's report "
        f"(tool result: {tool_result[:300]!r}). A background launch ends the "
        "turn and closes the CLI before the sub-agent reports, so its work "
        "is lost; build_sdk_env() must set CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1."
    )
    assert server.subagent_report_sent
    assert _REPLY_WITH_REPORT in _reply_text(received)
