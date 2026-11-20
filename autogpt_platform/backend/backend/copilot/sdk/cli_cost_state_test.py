"""The bundled CLI counts a resumed session's cost on from what our reader finds.

A turn is charged ``total_cost_usd`` minus ``cli_session_cost_usd`` of the file
the CLI resumed, so a CLI bump that changes where the resumed total comes from
must fail here rather than mis-bill. Runs against a localhost fake provider.
"""

import asyncio
import json
import uuid
from pathlib import Path

import pytest
from aiohttp import web

from backend.copilot.sdk.cost_tracking import read_cli_session_usage
from backend.copilot.transcript import cli_session_cost_usd, cli_session_path

from .cli_openrouter_compat_test import _resolve_cli_path

_MODEL = "claude-sonnet-5-5"
_CALL_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 100,
    "cache_read_input_tokens": 100_000,
    "cache_creation_input_tokens": 2000,
}


@pytest.mark.asyncio
async def test_resumed_cli_total_is_restored_cost_plus_this_call(tmp_path, monkeypatch):
    cli = _resolve_cli_path()
    if cli is None or not cli.is_file():
        pytest.skip("No Claude Code CLI binary available")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    cwd = tmp_path / "run"
    cwd.mkdir()
    session_id = str(uuid.uuid4())
    runner, port = await _start_fake_provider()
    try:
        first = await _run_cli(cli, cwd, port, "--session-id", session_id)
        session_file = Path(cli_session_path(str(cwd), session_id))
        resumed = []
        for foreign_row in (False, True):
            if foreign_row:
                # A row from another session must move neither the CLI nor us.
                with session_file.open("a") as f:
                    row = {"type": "cost-state", "sessionId": "x", "totalCostUSD": 5}
                    f.write(json.dumps(row) + "\n")
            restored = cli_session_cost_usd(session_file.read_text(), session_id)
            result = await _run_cli(cli, cwd, port, "--resume", session_id)
            resumed.append((restored, result["total_cost_usd"]))
            for key, value in _CALL_USAGE.items():
                assert result["usage"][key] == value
    finally:
        await runner.cleanup()

    call_cost = first["total_cost_usd"]
    assert call_cost > 0
    assert first["usage"]["input_tokens"] == _CALL_USAGE["input_tokens"]
    for restored, total in resumed:
        assert restored > 0
        assert total - restored == pytest.approx(call_cost)
    snapshot = read_cli_session_usage(str(cwd), session_id, "")
    assert snapshot.cost_usd == pytest.approx(resumed[-1][1])
    assert snapshot.tokens == {key: 3 * value for key, value in _CALL_USAGE.items()}


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


async def _run_cli(cli: Path, cwd: Path, port: int, *session_args: str) -> dict:
    # A clean environment: a Claude Code session's own CLAUDE_* flags change
    # what the CLI does.
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(cwd.parent),
        "CLAUDE_CONFIG_DIR": str(cwd.parent / "config"),
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_API_KEY": "sk-test-fake-key-not-real",
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    }
    query = {"type": "user", "message": {"role": "user", "content": "Say ok."}}
    proc = await asyncio.create_subprocess_exec(
        str(cli),
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--input-format",
        "stream-json",
        "--model",
        _MODEL,
        "--tools",
        "",
        "--setting-sources",
        "",
        "--system-prompt",
        "Reply ok.",
        *session_args,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate((json.dumps(query) + "\n").encode()), timeout=60
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    assert proc.returncode == 0, stderr.decode()[-2000:]
    lines = [json.loads(line) for line in stdout.splitlines() if line.startswith(b"{")]
    return next(line for line in lines if line.get("type") == "result")


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
