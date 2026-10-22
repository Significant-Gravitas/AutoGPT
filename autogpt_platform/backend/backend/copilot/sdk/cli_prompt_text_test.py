"""Text the copilot hands the Claude Code CLI must reach the model as written.

Runs the real Claude Code CLI against a stub Anthropic server, so a
``claude-agent-sdk`` bump whose CLI ignores ``verbatim_prompts`` or
``--disable-slash-commands``, or rewrites the text it is given, fails here.
The first two tests check one of the two on its own; the third checks a
session restored through ``process_cli_restore``. The SDK finds its bundled CLI,
which the Linux wheel ships; ``CHAT_CLAUDE_AGENT_CLI_PATH`` (or
``CLAUDE_AGENT_CLI_PATH``) points it at another. Only off Linux, with no CLI
to be found, do the tests skip.
"""

import asyncio
import gzip
import json
import os
import platform
import socket
import uuid
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    CLINotFoundError,
    SystemMessage,
)

from backend.copilot.transcript import (
    MENTION_ATTACHMENT_TYPES,
    TranscriptDownload,
    cli_session_path,
)

from .cli_version import require_verbatim_prompts_cli
from .env import (
    DISABLE_SLASH_COMMANDS_FLAG,
    build_sdk_env,
    cli_prompt_text,
    cli_user_message,
)
from .launch_env import launch_options
from .service import process_cli_restore
from .tool_adapter import get_copilot_tool_names, get_sdk_disallowed_tools

_MODEL = "claude-sonnet-4-5"
_GUARDED = [
    cli_prompt_text(text)
    for text in ["/compact", "/clear", "/model claude-opus-4-1", "/dataviz a chart"]
]
# Sent unguarded, so this turn checks the flag on its own.
_UNGUARDED = "/model claude-opus-4-1"
_TURNS = ["hello", *_GUARDED, _UNGUARDED, "still there?"]
_NOTE = "The note's own words."
_PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)

Send = Callable[[ClaudeSDKClient], Awaitable[None]]


@pytest.fixture
def cli_path(monkeypatch: pytest.MonkeyPatch) -> str:
    """The CLI the copilot would start: the override if one is set, else the
    one the SDK finds, as returned by the copilot's version check."""
    override = os.environ.get("CHAT_CLAUDE_AGENT_CLI_PATH") or os.environ.get(
        "CLAUDE_AGENT_CLI_PATH"
    )
    if override and not Path(override).is_file():
        pytest.fail(f"The CLI path override {override!r} is not a file")
    # A run inside Claude Code inherits its CLAUDE_* flags, which change the CLI.
    for key in [
        k
        for k in os.environ
        if k.startswith("CLAUDE") and k != "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"
    ]:
        monkeypatch.delenv(key)
    try:
        return require_verbatim_prompts_cli(override or None)
    except CLINotFoundError:
        if platform.system() == "Linux":
            raise
        pytest.skip("No Claude Code CLI: this platform's SDK wheel bundles none")


@pytest.mark.asyncio
async def test_slash_text_reaches_the_model_as_user_text(cli_path: str, tmp_path: Path):
    """``--disable-slash-commands`` and the leading space, without
    ``verbatim_prompts``."""
    sdk_cwd = _session_dir(tmp_path)
    async with _stub_server() as (base_url, requests):
        options = _options(
            cli_path, base_url, tmp_path, sdk_cwd, verbatim_prompts=False
        )
        init = await _run(
            options, [_query(text, "slash-commands-test") for text in _TURNS]
        )

    assert init["slash_commands"] == [], init["slash_commands"]
    assert init["skills"] == [], init["skills"]
    sent = [_user_texts(r)[-1] for r in requests]
    assert sent[: len(_GUARDED) + 1] == ["hello", *_GUARDED], sent
    assert sent[-1] == "still there?", sent
    assert {r["model"] for r in requests} == {_MODEL}
    # Nothing compacted or cleared the conversation in between.
    assert set(_GUARDED) <= set(_user_texts(requests[-1]))
    assert not (tmp_path / "config" / "settings.json").exists()


@pytest.mark.asyncio
async def test_prompt_text_reaches_the_model_as_written(cli_path: str, tmp_path: Path):
    """``verbatim_prompts`` on a fresh turn, a turn with an image and a resumed
    turn: the model gets each text exactly as sent, unguarded."""
    sdk_cwd = _session_dir(tmp_path)
    (sdk_cwd / "note.txt").write_text(_NOTE)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "note.txt").write_text(_NOTE)
    (elsewhere / "unnamed.txt").write_text(_NOTE)
    fresh = f"compare @note.txt with @{(elsewhere / 'note.txt').as_posix()}"
    with_image = f"/compact what is in @{elsewhere.as_posix()}/"
    resumed = "/model claude-opus-4-1 then @note.txt"
    session_id = str(uuid.uuid4())

    async with _stub_server() as (base_url, requests):
        first = _options(
            cli_path,
            base_url,
            tmp_path,
            sdk_cwd,
            verbatim_prompts=True,
            session_id=session_id,
        )
        await _run(
            first, [_query(fresh, session_id), _image_turn(with_image, session_id)]
        )
        later = _options(
            cli_path,
            base_url,
            tmp_path,
            sdk_cwd,
            verbatim_prompts=True,
            resume=session_id,
        )
        await _run(later, [_query(resumed, session_id)])

    # On an image turn the CLI ends the text with a newline before its note.
    sent_texts = [_user_texts(r)[-1].rstrip("\n") for r in requests]
    assert sent_texts == [fresh, with_image, resumed]
    assert "image" in _block_types(requests[1]["messages"][-1])
    assert {r["model"] for r in requests} == {_MODEL}
    sent = json.dumps(requests)
    assert _NOTE not in sent
    assert "unnamed.txt" not in sent


@pytest.mark.asyncio
async def test_restored_session_resumes_without_attachments_from_earlier_text(
    cli_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A session written without ``verbatim_prompts`` goes through
    ``process_cli_restore`` and resumes with its turns but not the attachments
    the CLI added for them."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    sdk_cwd = _session_dir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    note = elsewhere / "note.txt"
    note.write_text(_NOTE)
    (elsewhere / "unnamed.txt").write_text(_NOTE)
    earlier = f"compare @{note.as_posix()} with @{elsewhere.as_posix()}/"
    session_id = str(uuid.uuid4())

    async with _stub_server() as (base_url, requests):
        written = _options(
            cli_path,
            base_url,
            tmp_path,
            sdk_cwd,
            verbatim_prompts=False,
            session_id=session_id,
        )
        await _run(written, [_query(earlier, session_id)])
        session_file = Path(cli_session_path(str(sdk_cwd), session_id))
        stored = session_file.read_bytes()
        session_file.unlink()
        note.unlink()
        scrubbed, restored = process_cli_restore(
            TranscriptDownload(content=stored, message_count=2),
            str(sdk_cwd),
            session_id,
            "[test]",
        )
        resumed = _options(
            cli_path,
            base_url,
            tmp_path,
            sdk_cwd,
            verbatim_prompts=True,
            resume=session_id,
        )
        await _run(resumed, [_query("and now?", session_id)])

    assert restored
    assert len(requests) == 2
    assert {"file", "directory"} <= _attachment_types(stored)
    assert not _attachment_types(scrubbed.encode()) & MENTION_ATTACHMENT_TYPES
    later = json.dumps(requests[1])
    assert _user_texts(requests[1])[-2:] == [earlier, "and now?"]
    assert _NOTE not in later
    assert "unnamed.txt" not in later


def _session_dir(tmp_path: Path) -> Path:
    sdk_cwd = tmp_path / "copilot-session"
    sdk_cwd.mkdir()
    return sdk_cwd


def _options(
    cli_path: str,
    base_url: str,
    tmp_path: Path,
    sdk_cwd: Path,
    **overrides: Any,
) -> ClaudeAgentOptions:
    """The copilot's CLI options, pointed at the stub server."""
    env = build_sdk_env(
        sdk_cwd=str(sdk_cwd), codex_gateway_url=base_url, codex_gateway_token="stub"
    )
    env["CLAUDE_CONFIG_DIR"] = str(tmp_path / "config")
    env["HOME"] = str(tmp_path)
    return ClaudeAgentOptions(
        system_prompt="You are a test assistant.",
        setting_sources=[],
        extra_args={"strict-mcp-config": None, DISABLE_SLASH_COMMANDS_FLAG: None},
        allowed_tools=get_copilot_tool_names(use_e2b=True),
        disallowed_tools=get_sdk_disallowed_tools(use_e2b=True),
        cwd=str(sdk_cwd),
        env=env,
        model=_MODEL,
        cli_path=cli_path,
        **overrides,
    )


async def _run(options: ClaudeAgentOptions, sends: list[Send]) -> dict[str, Any]:
    """One CLI process, one turn per send; returns the CLI's init data."""
    init: dict[str, Any] = {}

    async def turns() -> None:
        nonlocal init
        async with ClaudeSDKClient(launch_options(options)) as client:
            for send in sends:
                await send(client)
                async for message in client.receive_response():
                    if isinstance(message, SystemMessage) and message.subtype == "init":
                        init = init or message.data

    await asyncio.wait_for(turns(), timeout=120)
    return init


def _query(text: str, session_id: str) -> Send:
    async def send(client: ClaudeSDKClient) -> None:
        await client.query(text, session_id=session_id)

    return send


def _image_turn(text: str, session_id: str) -> Send:
    """A turn with an attached image, written the way the copilot writes it."""
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _PNG_1X1},
    }
    line = cli_user_message([image, {"type": "text", "text": text}], session_id)

    async def send(client: ClaudeSDKClient) -> None:
        assert client._transport is not None
        await client._transport.write(json.dumps(line) + "\n")

    return send


@asynccontextmanager
async def _stub_server() -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    """A stub Messages API on a free port, and the requests it receives."""
    requests: list[dict[str, Any]] = []
    runner = web.AppRunner(_stub_app(requests))
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    await web.SockSite(runner, sock).start()
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}", requests
    finally:
        await runner.cleanup()


def _stub_app(requests: list[dict[str, Any]]) -> web.Application:
    async def messages(request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        encoding = request.headers.get("Content-Encoding", "")
        if "gzip" in encoding:
            raw = gzip.decompress(raw)
        elif "deflate" in encoding:
            raw = zlib.decompress(raw)
        payload = json.loads(raw)
        requests.append(payload)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in _reply_events(payload.get("model", _MODEL)):
            await response.write(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            )
        await response.write_eof()
        return response

    async def count_tokens(_: web.Request) -> web.Response:
        return web.json_response({"input_tokens": 1})

    app = web.Application(client_max_size=64 * 1024**2)
    app.router.add_post("/v1/messages", messages)
    app.router.add_post("/v1/messages/count_tokens", count_tokens)
    return app


def _reply_events(model: str) -> list[dict[str, Any]]:
    usage = {"input_tokens": 10, "output_tokens": 1}
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_stub",
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
            "delta": {"type": "text_delta", "text": "ok"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": usage,
        },
        {"type": "message_stop"},
    ]


def _user_texts(payload: dict[str, Any]) -> list[str]:
    """Each user message's own text in the request, without the reminders and
    image notes the CLI adds."""
    texts = []
    for message in payload["messages"]:
        if message["role"] != "user":
            continue
        texts += [
            block["text"]
            for block in _blocks(message)
            if block.get("type") == "text"
            and not block["text"].startswith(("<system-reminder>", "[Image: "))
        ]
    return texts


def _attachment_types(session: bytes) -> set[str]:
    """The types of the attachment entries in a CLI session file."""
    entries = [json.loads(line) for line in session.decode().splitlines() if line]
    return {
        entry["attachment"].get("type", "")
        for entry in entries
        if entry.get("type") == "attachment"
    }


def _block_types(message: dict[str, Any]) -> list[str]:
    return [block.get("type", "") for block in _blocks(message)]


def _blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message["content"]
    return [{"type": "text", "text": content}] if isinstance(content, str) else content
