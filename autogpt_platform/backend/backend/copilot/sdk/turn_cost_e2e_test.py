"""Two chat turns through ``stream_chat_completion_sdk`` with the bundled CLI.

The second turn restores the first turn's uploaded CLI session on a fresh disk
and resumes it, as a turn on another pod does. Only the database, Redis and
transcript storage are stubbed; the CLI talks to a localhost fake provider.
"""

import contextlib
import json
import shutil
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from claude_agent_sdk import ClaudeSDKClient

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.transcript import TranscriptDownload

from .cli_openrouter_compat_test import _resolve_cli_path
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
_MODEL = "claude-sonnet-5-5"
_CALL_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 100,
    "cache_read_input_tokens": 100_000,
    "cache_creation_input_tokens": 2000,
}
_STUBBED_HERE = {
    f"{_SVC}.ClaudeSDKClient",
    f"{_SVC}._make_sdk_cwd",
    "os.makedirs",
    f"{_SVC}.download_transcript",
    f"{_SVC}.upload_transcript",
    f"{_SVC}.strip_for_upload",
    f"{_SVC}.validate_transcript",
    f"{_SVC}.build_sdk_env",
}


@pytest.mark.asyncio
async def test_second_turn_on_a_new_pod_is_charged_only_its_own_call(
    tmp_path, monkeypatch
):
    cli = _resolve_cli_path()
    if cli is None or not cli.is_file():
        pytest.skip("No Claude Code CLI binary available")
    config_dir = tmp_path / "config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    runner, port = await _start_fake_provider()
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_API_KEY": "sk-test-fake-key-not-real",
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    }
    session_id = str(uuid.uuid4())
    stored: dict[str, TranscriptDownload] = {}
    launched: list[tuple[str | None, str | None]] = []
    results: list[float] = []

    def _bundled_cli_client(*args, options, **kwargs):
        launched.append((options.resume, options.session_id))
        options.cli_path = str(cli)
        options.env = env
        options.model = _MODEL
        options.fallback_model = None
        options.mcp_servers = {}
        options.hooks = None
        options.allowed_tools = []
        options.disallowed_tools = []
        options.system_prompt = "Reply ok."
        client = ClaudeSDKClient(*args, options=options, **kwargs)
        receive = client.receive_response

        async def _receive_and_note_totals():
            async for message in receive():
                total = getattr(message, "total_cost_usd", None)
                if total is not None:
                    results.append(total)
                yield message

        client.receive_response = _receive_and_note_totals
        return client

    async def _upload(*, session_id, content, message_count, mode, **_):
        stored[session_id] = TranscriptDownload(
            content=content, message_count=message_count, mode=mode
        )

    async def _download(user_id, session_id, log_prefix="", **_):
        return stored.get(session_id)

    try:
        charged = []
        history: list[ChatMessage] = []
        for prompt in ("hello", "again"):
            history.append(ChatMessage(role="user", content=prompt))
            session = ChatSession(
                session_id=session_id,
                user_id="test-user",
                usage=[],
                started_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                messages=list(history),
                title="Cost check",
            )
            persist = await _run_turn(
                session, prompt, tmp_path, env, _bundled_cli_client, _upload, _download
            )
            persist.assert_awaited_once()
            charged.append(persist.await_args.kwargs["cost_usd"])
            history = list(session.messages)
            # The next turn runs on another pod: only uploaded state survives.
            shutil.rmtree(config_dir / "projects", ignore_errors=True)
    finally:
        await runner.cleanup()

    assert launched == [(None, session_id), (session_id, None)]
    call_cost, resumed_total = results
    assert call_cost > 0
    # The CLI restored turn 1's total from the uploaded session file.
    assert resumed_total == pytest.approx(2 * call_cost)
    assert charged == [pytest.approx(call_cost), pytest.approx(call_cost)]


async def _run_turn(session, prompt, tmp_path, env, client, upload, download):
    patches = [
        (target, kwargs)
        for target, kwargs in _make_sdk_patches(
            session,
            original_transcript="",
            compacted_transcript=None,
            client_side_effect=client,
        )
        if target not in _STUBBED_HERE
    ]
    sdk_cwd = tmp_path / "copilot-cwd"
    sdk_cwd.mkdir(exist_ok=True)
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        stack.enter_context(patch(f"{_SVC}.ClaudeSDKClient", side_effect=client))
        stack.enter_context(patch(f"{_SVC}._make_sdk_cwd", return_value=str(sdk_cwd)))
        stack.enter_context(patch(f"{_SVC}.build_sdk_env", return_value=env))
        stack.enter_context(patch(f"{_SVC}.upload_transcript", side_effect=upload))
        stack.enter_context(patch(f"{_SVC}.download_transcript", side_effect=download))
        stack.enter_context(
            patch(
                f"{_SVC}.build_skills_update_notice",
                new_callable=AsyncMock,
                return_value=None,
            )
        )
        stack.enter_context(
            patch(f"{_SVC}.create_security_hooks", return_value=MagicMock())
        )
        for name, value in (
            ("inject_user_context", None),
            ("build_session_context", ""),
        ):
            stack.enter_context(
                patch(f"{_SVC}.{name}", new_callable=AsyncMock, return_value=value)
            )
        persist = stack.enter_context(
            patch(f"{_SVC}.persist_and_record_usage", new_callable=AsyncMock)
        )
        stack.enter_context(
            patch(f"{_SVC}.record_turn_cost_from_openrouter", new_callable=AsyncMock)
        )
        async for _ in stream_chat_completion_sdk(
            session_id=session.session_id,
            message=prompt,
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            pass
    return persist


async def _start_fake_provider() -> tuple[web.AppRunner, int]:
    async def messages(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(_streamed_reply().encode())
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    app.router.add_route("*", "/{tail:.*}", lambda _r: web.Response(status=404))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    return runner, site._server.sockets[0].getsockname()[1]


def _streamed_reply() -> str:
    message = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": _MODEL,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {**_CALL_USAGE, "output_tokens": 0},
    }
    events = [
        {"type": "message_start", "message": message},
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
            "usage": {"output_tokens": _CALL_USAGE["output_tokens"]},
        },
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
