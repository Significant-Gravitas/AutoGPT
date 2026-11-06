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
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from aiohttp import web
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from backend.copilot.context_blocks import ContextBlock, render_context_blocks

from .after_compaction import fit_after_compaction, utf16_len
from .cli_openrouter_compat_test import _resolve_cli_path, assert_pinned_cli
from .compaction import CompactionTracker
from .security_hooks import create_security_hooks

_CLI_HOOK_CAP = 10_000
_LEAD = "The conversation was just compacted; here are its first blocks again:\n\n"
_REMINDER = "SessionStart hook additional context: "
# Small enough that two big turns push the CLI over its compaction trigger.
_CONTEXT_WINDOW_TOKENS = 50_000
_BIG_TURN = "The quick brown fox jumps over the lazy dog near the riverbank. " * 1_800


def _block(tag: str, lines: int, line: str) -> ContextBlock:
    return ContextBlock(
        tag=tag,
        head=f"Header line for {tag}.",
        items=tuple(f"- {line} {i}" for i in range(lines)),
    )


def _first_turn_blocks() -> list[ContextBlock]:
    return [
        _block("available_skills", 260, "name: skill — does a thing — triggers: x"),
        _block("expert_workflows", 150, "Workflow (library_agent_id: abc) : desc"),
        _block("team_context", 160, "Expert — role (expert id: e1); workflows"),
        ContextBlock(tag="session_context", head="session_id: s1"),
        ContextBlock(tag="env_context", head="working_dir: /tmp/copilot-s1"),
        _block("user_context", 40, "Business detail line for the user"),
    ]


@pytest.mark.asyncio
async def test_over_cap_context_reaches_the_model_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cli = _resolve_cli_path()
    if cli is None:
        pytest.skip("No Claude Code CLI binary bundled or configured")
    # A run inside Claude Code inherits its CLAUDE_* flags.
    for key in [k for k in os.environ if k.startswith("CLAUDE")]:
        monkeypatch.delenv(key)
    blocks = _first_turn_blocks()
    assert utf16_len(_LEAD + render_context_blocks(blocks)) > 2 * _CLI_HOOK_CAP
    fitted = fit_after_compaction(_LEAD, blocks)
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


@pytest.mark.asyncio
async def test_each_compaction_counts_once_for_the_tracker_and_the_resend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The PreCompact hook (what ``CompactionTracker`` counts) and the
    SessionStart re-send each fire once per compaction the CLI runs, so the
    turn summary's compaction and re-send counts describe the same ones."""
    cli = _resolve_cli_path()
    if cli is None:
        pytest.skip("No Claude Code CLI binary bundled or configured")
    for key in [k for k in os.environ if k.startswith("CLAUDE")]:
        monkeypatch.delenv(key)
    tracker = CompactionTracker()
    rebuilds: list[int] = []

    async def _context() -> list[str]:
        rebuilds.append(len(rebuilds))
        return [_LEAD + _block("user_context", 4, "Business detail").text]

    requests = await asyncio.wait_for(
        _requests_around_a_compaction(
            cli, tmp_path, _context, on_compact=tracker.on_compact, turns=5
        ),
        timeout=180,
    )

    summaries = sum(1 for r in requests if _is_summary_request(r))
    assert summaries >= 2
    assert tracker.attempt_count == len(rebuilds) == summaries


async def _requests_around_a_compaction(
    cli: Path,
    tmp_path: Path,
    context_after_compaction: Callable[[], Awaitable[list[str]]],
    *,
    on_compact: Callable[[str], None] | None = None,
    turns: int = 3,
) -> list[dict]:
    requests: list[dict] = []
    user_agents: list[str] = []
    runner, port = await _start_stub_api(requests, user_agents)
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
            on_compact=on_compact,
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
            for _ in range(turns):
                await client.query(_BIG_TURN)
                async for _ in client.receive_response():
                    pass
    finally:
        await runner.cleanup()
    assert_pinned_cli(user_agents, cli)
    return requests


async def _start_stub_api(
    requests: list[dict], user_agents: list[str]
) -> tuple[web.AppRunner, int]:
    """An Anthropic API that answers every call with a short streamed reply,
    reporting usage that grows with the conversation."""

    async def messages(request: web.Request) -> web.StreamResponse:
        user_agents.append(request.headers.get("User-Agent", ""))
        payload = await request.json()
        requests.append(payload)
        compacting = _is_summary_request(payload)
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


def _is_summary_request(payload: dict) -> bool:
    """The CLI's compaction request: its last message asks for the summary."""
    last = (payload.get("messages") or [{}])[-1]
    return "detailed summary of the conversation" in json.dumps(last)


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
