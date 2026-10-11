"""The real CLI hands the model every part of the post-compaction re-send whole.

Drives the Claude Code CLI against a stub Anthropic API, makes it compact,
and checks the request that follows: the ``SessionStart`` hooks wired by
``create_security_hooks`` re-send a context well over the CLI's 10,000-
character hook-output cap, and the model must get all of it, in order, with
no ``<persisted-output>`` preview in its place.  A CLI bump that changes the
cap, shares it between callbacks or reorders them fails here.

Uses the CLI ``CHAT_CLAUDE_AGENT_CLI_PATH`` / ``CLAUDE_AGENT_CLI_PATH`` names,
else the one bundled with ``claude-agent-sdk``; skips only when neither is set
and none is bundled.
"""

import asyncio
import json
import os
import platform
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import claude_agent_sdk
import pytest
from aiohttp import web
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from .after_compaction import fit_after_compaction, utf16_len
from .security_hooks import create_security_hooks

_CLI_HOOK_CAP = 10_000
_LEAD = "The conversation was just compacted; here are its first blocks again:\n\n"
_REMINDER = "SessionStart hook additional context: "
# Small enough that two big turns push the CLI over its compaction trigger.
_CONTEXT_WINDOW_TOKENS = 50_000
_BIG_TURN = "The quick brown fox jumps over the lazy dog near the riverbank. " * 1_800


def _cli_path() -> str | None:
    override = os.environ.get("CHAT_CLAUDE_AGENT_CLI_PATH") or os.environ.get(
        "CLAUDE_AGENT_CLI_PATH"
    )
    if override:
        return override
    name = "claude.exe" if platform.system() == "Windows" else "claude"
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / name
    return str(bundled) if bundled.is_file() else None


def _block(tag: str, lines: int, line: str) -> str:
    body = "\n".join(f"- {line} {i}" for i in range(lines))
    return f"<{tag}>\nHeader line for {tag}.\n{body}\n</{tag}>\n\n"


def _first_turn_prefix() -> str:
    return (
        _block("available_skills", 260, "name: skill — does a thing — triggers: x")
        + _block("expert_workflows", 150, "Workflow (library_agent_id: abc) : desc")
        + _block("team_context", 160, "Expert — role (expert id: e1); workflows")
        + "<session_context>\nsession_id: s1\n</session_context>\n\n"
        + "<env_context>\nworking_dir: /tmp/copilot-s1\n</env_context>\n\n"
        + _block("user_context", 40, "Business detail line for the user")
    )


@pytest.mark.asyncio
async def test_over_cap_context_reaches_the_model_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cli = _cli_path()
    if not cli:
        pytest.skip("No Claude Code CLI binary bundled or configured")
    # A run inside Claude Code inherits its CLAUDE_* flags.
    for key in [k for k in os.environ if k.startswith("CLAUDE")]:
        monkeypatch.delenv(key)
    prefix = _first_turn_prefix()
    assert utf16_len(_LEAD + prefix) > 2 * _CLI_HOOK_CAP
    fitted = fit_after_compaction(_LEAD, prefix)
    assert len(fitted.parts) > 1

    async def _context() -> list[str]:
        return fitted.parts

    requests = await asyncio.wait_for(
        _requests_around_a_compaction(cli, tmp_path, _context), timeout=180
    )

    after = [r for r in requests if _REMINDER in json.dumps(r)]
    assert after, "no request after the compaction carried the SessionStart context"
    (reminder,) = [t for t in _texts(after[0]) if _REMINDER in t]
    assert "<persisted-output>" not in reminder
    assert _REMINDER + "\n".join(fitted.parts) in reminder


async def _requests_around_a_compaction(
    cli: str,
    tmp_path: Path,
    context_after_compaction: Callable[[], Awaitable[list[str]]],
) -> list[dict]:
    requests: list[dict] = []
    runner, port = await _start_stub_api(requests)
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_AUTH_TOKEN": "stub",
        "CLAUDE_CONFIG_DIR": str(tmp_path / "config"),
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(_CONTEXT_WINDOW_TOKENS),
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(_CONTEXT_WINDOW_TOKENS),
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "90",
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    options = ClaudeAgentOptions(
        cwd=str(work),
        env=env,
        model="claude-sonnet-4-5",
        hooks=create_security_hooks(
            "u1",
            sdk_cwd=str(work),
            context_after_compaction=context_after_compaction,
            allowed_tools=(),
        ),
        setting_sources=[],
        permission_mode="bypassPermissions",
        system_prompt="You are a test.",
        cli_path=cli,
    )
    try:
        async with ClaudeSDKClient(options=options) as client:
            for turn in (_BIG_TURN, _BIG_TURN, _BIG_TURN):
                await client.query(turn)
                async for _ in client.receive_response():
                    pass
    finally:
        await runner.cleanup()
    return requests


async def _start_stub_api(requests: list[dict]) -> tuple[web.AppRunner, int]:
    """An Anthropic API that answers every call with a short streamed reply,
    reporting usage that grows with the conversation."""

    async def messages(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        requests.append(payload)
        compacting = "detailed summary of the conversation" in json.dumps(
            (payload.get("messages") or [{}])[-1]
        )
        text = "<summary>Earlier turns.</summary>" if compacting else "noted"
        tokens = 1 if compacting else len(json.dumps(payload.get("messages"))) // 4
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in _reply_events(payload.get("model", ""), text, tokens):
            await response.write(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            )
        await response.write_eof()
        return response

    async def count_tokens(_: web.Request) -> web.Response:
        return web.json_response({"input_tokens": 1})

    app = web.Application(client_max_size=1024**3)
    app.router.add_post("/v1/messages", messages)
    app.router.add_post("/v1/messages/count_tokens", count_tokens)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, runner.addresses[0][1]


def _reply_events(model: str, text: str, input_tokens: int) -> list[dict]:
    usage = {"input_tokens": input_tokens, "output_tokens": 1}
    return [
        {
            "type": "message_start",
            "message": {
                "id": f"msg_stub_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
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
            "usage": usage,
        },
        {"type": "message_stop"},
    ]


def _texts(payload: dict) -> list[str]:
    texts: list[str] = []
    for message in payload.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        texts.extend(
            block.get("text", "")
            for block in content or []
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return texts
