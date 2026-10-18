"""Wire test: a resumed session sends the prompt its process was launched with.

With its prompt snapshot on, the Claude Code CLI records the system prompt
and every tool description on a session's first request and resends that
record on every later request and ``--resume`` until a compaction, ignoring
the prompt a later process was given. The building-mode restart resumes the
turn's session with a prompt that now carries the agent-building guide, so
with the snapshot on the guide never reaches the model.

Each case runs the real CLI twice against a local stub of the Anthropic
Messages API: process 1 starts a session with prompt V1, process 2 resumes it
with prompt V2, and the resumed request's system blocks must carry V2 and not
V1. Skipped only where no CLI binary exists (the SDK's Windows wheel bundles
none); ``CLAUDE_AGENT_CLI_PATH`` points it at one.
"""

import asyncio
import gzip
import json
import os
import zlib
from pathlib import Path

import pytest
from aiohttp import web
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage

from backend.copilot.route_profile import codex_profile

from .cli_openrouter_compat_test import _resolve_cli_path, assert_pinned_cli
from .env import GatewayEndpoint, route_launch_env
from .service import _build_system_prompt_value

_PROMPT_V1 = "You are a test assistant. PROMPT_MARK_V1"
_PROMPT_V2 = "You are a test assistant. PROMPT_MARK_V2"
_FIRST_TURN = "FIRST_TURN_MARK hello"
_RESUMED_TURN = "RESUMED_TURN_MARK hello again"
_PROCESS_TIMEOUT_SECONDS = 120


@pytest.fixture
def cli_path(monkeypatch: pytest.MonkeyPatch) -> Path:
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
    return path


@pytest.mark.asyncio
async def test_resumed_session_sends_the_relaunched_prompt(
    cli_path: Path, tmp_path: Path
):
    """The building-mode restart's shape: same session, new prompt."""
    stub = _StubAnthropic()
    port = await stub.start()
    try:
        session_id = await _run_turn(
            cli_path,
            tmp_path,
            port,
            _prompt_value(_PROMPT_V1),
            _FIRST_TURN,
            resume=None,
        )
        await _run_turn(
            cli_path,
            tmp_path,
            port,
            _prompt_value(_PROMPT_V2),
            _RESUMED_TURN,
            resume=session_id,
        )
    finally:
        await stub.stop()

    assert "PROMPT_MARK_V1" in _system_text(stub.turn_request(_FIRST_TURN))
    resumed = stub.turn_request(_RESUMED_TURN)
    assert _FIRST_TURN in json.dumps(resumed["messages"]), "the resume lost history"
    system = _system_text(resumed)
    assert "PROMPT_MARK_V2" in system, "the resumed turn lost the relaunched prompt"
    assert (
        "PROMPT_MARK_V1" not in system
    ), "the resumed turn sent the prompt recorded on the session's first request"
    assert_pinned_cli(stub.user_agents, cli_path)


def _prompt_value(text: str):
    return _build_system_prompt_value(text, live=True)


async def _run_turn(
    cli_path: Path,
    tmp_path: Path,
    port: int,
    system_prompt,
    text: str,
    *,
    resume: str | None,
) -> str:
    """Run one turn in a fresh CLI process; return the session id."""
    sdk_cwd = tmp_path / "copilot-session"
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
    env["CLAUDE_CONFIG_DIR"] = str(tmp_path / "config")
    env["HOME"] = str(tmp_path)
    options = ClaudeAgentOptions(
        cli_path=str(cli_path),
        cwd=str(sdk_cwd),
        model="claude-sonnet-4-5",
        system_prompt=system_prompt,
        setting_sources=[],
        extra_args={"strict-mcp-config": None},
        max_turns=1,
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
    """A loopback Messages API that records every request body and answers
    each with a one-word reply."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.user_agents: list[str] = []
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
        if body.get("stream") is not True:
            return web.json_response(_message(model))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in _stream_events(model):
            await response.write(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            )
        await response.write_eof()
        return response


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


def _message(model: str) -> dict:
    return {
        "id": "msg_stub",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": _USAGE,
    }


def _stream_events(model: str) -> list[dict]:
    start = _message(model) | {"content": [], "stop_reason": None}
    return [
        {"type": "message_start", "message": start},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "ok"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": _USAGE,
        },
        {"type": "message_stop"},
    ]


def _system_text(body: dict) -> str:
    system = body.get("system")
    if isinstance(system, str):
        return system
    return "\n".join(block.get("text", "") for block in system or [])
