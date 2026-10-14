"""Wire test: a mid-turn follow-up is still in context after ``--resume``.

A follow-up the user sends while a tool runs reaches the model as the
PostToolUse hook's ``additionalContext``.  The transcript watermark uploaded
after that turn counts the follow-up's DB row as covered, so the next turn
does not re-send it: the only copy the model has is the one the Claude Code
CLI kept in its session file.  A CLI that stops storing or replaying hook
context would silently drop follow-ups from every resumed session.

The test runs the real CLI twice against a local stub of the Anthropic
Messages API.  Process 1 runs a turn with one tool call whose PostToolUse hook
returns a follow-up; the session file then goes through the upload pipeline;
process 2 resumes it, and the resumed request must carry the follow-up.
Skipped only where no CLI binary exists (the SDK's Windows wheel bundles
none); ``CLAUDE_AGENT_CLI_PATH`` points it at one.
"""

import asyncio
import gzip
import json
import os
import zlib
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)
from claude_agent_sdk.types import HookContext, HookInput, SyncHookJSONOutput

from backend.copilot.pending_messages import PendingMessage, format_pending_as_followup
from backend.copilot.route_profile import codex_profile
from backend.copilot.transcript import cli_session_path

from .cli_openrouter_compat_test import _resolve_cli_path, assert_pinned_cli
from .env import GatewayEndpoint, route_launch_env
from .service import (
    _strip_synthetic_reprompt_from_cli_jsonl,
    read_cli_session_from_disk,
)

_FIRST_TURN = "FIRST_TURN_MARK build me an agent"
_FOLLOWUP = "FOLLOWUP_MARK make it run hourly"
_RESUMED_TURN = "RESUMED_TURN_MARK what did I ask for?"
_TOOL_NAME = "mcp__followup__echo"
_PROCESS_TIMEOUT_SECONDS = 120


@pytest.fixture
def cli_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = _resolve_cli_path()
    if path is None:
        pytest.skip(
            "No Claude Code CLI binary: none bundled with claude-agent-sdk and "
            "CLAUDE_AGENT_CLI_PATH is unset."
        )
    # A run inside Claude Code inherits its CLAUDE_* flags, which change the
    # CLI's behaviour.
    for key in [k for k in os.environ if k.startswith("CLAUDE")]:
        monkeypatch.delenv(key)
    # The upload pipeline finds the session file through this variable.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    return path


@pytest.mark.asyncio
async def test_midturn_followup_is_replayed_to_a_resumed_session(
    cli_path: Path, tmp_path: Path
):
    stub = _StubAnthropic()
    port = await stub.start()
    try:
        session_id = await _run_turn(cli_path, tmp_path, port, _FIRST_TURN, resume=None)
        uploaded = _through_the_upload_pipeline(tmp_path, session_id)
        await _run_turn(cli_path, tmp_path, port, _RESUMED_TURN, resume=session_id)
    finally:
        await stub.stop()

    first_turn_wire = json.dumps(stub.turn_request(_TOOL_NAME + "-result")["messages"])
    assert "FOLLOWUP_MARK" in first_turn_wire, "the hook's follow-up never reached"
    assert b"FOLLOWUP_MARK" in uploaded, (
        "the follow-up is not in the session file that would be uploaded, so a "
        "resumed session cannot have it"
    )
    resumed_wire = json.dumps(stub.turn_request(_RESUMED_TURN)["messages"])
    assert "FIRST_TURN_MARK" in resumed_wire, "the resume lost history"
    assert "FOLLOWUP_MARK" in resumed_wire, (
        "the resumed session lost the mid-turn follow-up: the CLI no longer "
        "replays PostToolUse additionalContext, so the transcript watermark "
        "must stop counting follow-up rows as covered"
    )
    assert_pinned_cli(stub.user_agents, cli_path)


def _through_the_upload_pipeline(tmp_path: Path, session_id: str) -> bytes:
    """What a finished turn uploads, put back where a later turn restores it."""
    sdk_cwd = str(_sdk_cwd(tmp_path))
    content = read_cli_session_from_disk(sdk_cwd, session_id, "[test]")
    assert content, "the CLI wrote no session file where the upload looks for it"
    content = _strip_synthetic_reprompt_from_cli_jsonl(content)
    Path(cli_session_path(sdk_cwd, session_id)).write_bytes(content)
    return content


def _sdk_cwd(tmp_path: Path) -> Path:
    return tmp_path / "copilot-session"


async def _followup_hook(
    _input: HookInput, _tool_use_id: str | None, _context: HookContext
) -> SyncHookJSONOutput:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": format_pending_as_followup(
                [PendingMessage(content=_FOLLOWUP)]
            ),
        }
    }


@tool("echo", "Echo the given text back.", {"text": str})
async def _echo(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"{_TOOL_NAME}-result {args}"}]}


async def _run_turn(
    cli_path: Path, tmp_path: Path, port: int, text: str, *, resume: str | None
) -> str:
    """Run one turn in a fresh CLI process; return the session id."""
    sdk_cwd = _sdk_cwd(tmp_path)
    sdk_cwd.mkdir(exist_ok=True)
    # The stub stands where the Codex route's loopback gateway would.
    env = route_launch_env(
        codex_profile(
            "claude-sonnet-4-5", effort=None, routing_source="env", codex_engine=None
        ),
        sdk_cwd=str(sdk_cwd),
        gateway=GatewayEndpoint(
            base_url=f"http://127.0.0.1:{port}", auth_token="stub-token"
        ),
    )
    env["CLAUDE_CONFIG_DIR"] = os.environ["CLAUDE_CONFIG_DIR"]
    env["HOME"] = str(tmp_path)
    options = ClaudeAgentOptions(
        cli_path=str(cli_path),
        cwd=str(sdk_cwd),
        model="claude-sonnet-4-5",
        system_prompt="You are a test assistant.",
        setting_sources=[],
        extra_args={"strict-mcp-config": None},
        mcp_servers={"followup": create_sdk_mcp_server(name="followup", tools=[_echo])},
        allowed_tools=[_TOOL_NAME],
        hooks={"PostToolUse": [HookMatcher(matcher="*", hooks=[_followup_hook])]},
        resume=resume,
        env=env,
    )
    session_id = ""
    async with asyncio.timeout(_PROCESS_TIMEOUT_SECONDS):
        async with ClaudeSDKClient(options=options) as client:
            await client.query(text)
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    session_id = message.session_id
    assert session_id, "the CLI reported no session id"
    return session_id


class _StubAnthropic:
    """A loopback Messages API that records every request body.  It asks for
    one tool call on the first turn and answers everything else with text."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.user_agents: list[str] = []
        self._asked_for_tool = False
        self._runner: web.AppRunner | None = None

    async def start(self) -> int:
        app = web.Application(client_max_size=64 * 1024**2)
        app.router.add_post("/v1/messages", self._messages)
        app.router.add_post("/v1/messages/count_tokens", _count_tokens)
        app.router.add_route("*", "/{tail:.*}", _not_found)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        return self._runner.addresses[0][1]

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    def turn_request(self, marker: str) -> dict:
        """The first request whose messages carry *marker*."""
        matches = [b for b in self.bodies if marker in json.dumps(b.get("messages"))]
        assert matches, f"no request carried {marker!r}"
        return matches[0]

    async def _messages(self, request: web.Request) -> web.StreamResponse:
        self.user_agents.append(request.headers.get("User-Agent", ""))
        body = await _read_json(request)
        self.bodies.append(body)
        model = body.get("model", "claude-sonnet-4-5")
        block = self._next_block(body)
        if body.get("stream") is not True:
            return web.json_response(_message(model, block))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in _stream_events(model, block):
            await response.write(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            )
        await response.write_eof()
        return response

    def _next_block(self, body: dict) -> dict:
        offered = any(t.get("name") == _TOOL_NAME for t in body.get("tools") or [])
        is_first_turn = _FIRST_TURN in json.dumps(body.get("messages"))
        if self._asked_for_tool or not offered or not is_first_turn:
            return {"type": "text", "text": "ok"}
        self._asked_for_tool = True
        return {
            "type": "tool_use",
            "id": "toolu_stub_echo",
            "name": _TOOL_NAME,
            "input": {"text": "hello"},
        }


async def _count_tokens(_request: web.Request) -> web.Response:
    return web.json_response({"input_tokens": 1})


async def _not_found(_request: web.Request) -> web.Response:
    return web.Response(status=404)


async def _read_json(request: web.Request) -> dict:
    raw = await request.read()
    encoding = request.headers.get("Content-Encoding", "").lower()
    if "gzip" in encoding:
        raw = gzip.decompress(raw)
    elif "deflate" in encoding:
        raw = zlib.decompress(raw)
    return json.loads(raw)


_USAGE = {
    "input_tokens": 10,
    "output_tokens": 1,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def _stop_reason(block: dict) -> str:
    return "tool_use" if block["type"] == "tool_use" else "end_turn"


def _message(model: str, block: dict) -> dict:
    return {
        "id": "msg_stub",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [block],
        "stop_reason": _stop_reason(block),
        "stop_sequence": None,
        "usage": _USAGE,
    }


def _stream_events(model: str, block: dict) -> list[dict]:
    start = _message(model, block) | {"content": [], "stop_reason": None}
    return [
        {"type": "message_start", "message": start},
        *_block_events(block),
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": _stop_reason(block), "stop_sequence": None},
            "usage": _USAGE,
        },
        {"type": "message_stop"},
    ]


def _block_events(block: dict) -> list[dict]:
    if block["type"] == "text":
        return [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": block["text"]},
            },
        ]
    return [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": block | {"input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            },
        },
    ]
