"""The turn's CLI process, spawned while the turn still builds its query."""

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest
from aiohttp import web
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from backend.copilot.sdk import cli_prelaunch
from backend.copilot.sdk.cli_openrouter_compat_test import _resolve_cli_path
from backend.copilot.sdk.cli_prelaunch import CLIPrelaunch
from backend.copilot.sdk.subagent_turn_end_test import _text_stream


class _FakeTransport:
    spawned: ClassVar[list["_FakeTransport"]] = []

    def __init__(self, prompt, options: ClaudeAgentOptions) -> None:
        self.options = options
        self.connected = False
        self.closed = False
        _FakeTransport.spawned.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True


class _UnspawnableTransport(_FakeTransport):
    async def connect(self) -> None:
        raise OSError("the CLI binary is missing")


@pytest.fixture
def spawned(mocker) -> list[_FakeTransport]:
    _FakeTransport.spawned = []
    mocker.patch.object(cli_prelaunch, "SubprocessCLITransport", _FakeTransport)
    return _FakeTransport.spawned


@pytest.mark.asyncio
async def test_the_process_goes_to_one_launch_with_the_options_it_was_spawned_for(
    spawned,
):
    options = ClaudeAgentOptions()

    prelaunch = await CLIPrelaunch.start(options, "[test]")

    assert prelaunch is not None
    [transport] = spawned
    assert transport.connected and transport.options is options
    # A retry rebuilds its options, so it gets nothing and spawns its own.
    assert prelaunch.claim(ClaudeAgentOptions()) is None
    assert prelaunch.claim(options) is transport
    assert prelaunch.claim(options) is None


@pytest.mark.asyncio
async def test_a_process_no_launch_claimed_is_stopped(spawned):
    options = ClaudeAgentOptions()
    prelaunch = await CLIPrelaunch.start(options, "[test]")
    assert prelaunch is not None

    await prelaunch.close()

    assert spawned[0].closed
    assert prelaunch.claim(options) is None


@pytest.mark.asyncio
async def test_a_claimed_process_is_left_to_the_client_that_runs_it(spawned):
    options = ClaudeAgentOptions()
    prelaunch = await CLIPrelaunch.start(options, "[test]")
    assert prelaunch is not None
    prelaunch.claim(options)

    await prelaunch.close()

    assert not spawned[0].closed


@pytest.mark.asyncio
async def test_a_process_that_cannot_spawn_leaves_the_launch_to_spawn_its_own(
    mocker,
):
    _FakeTransport.spawned = []
    mocker.patch.object(cli_prelaunch, "SubprocessCLITransport", _UnspawnableTransport)

    assert await CLIPrelaunch.start(ClaudeAgentOptions(), "[test]") is None
    assert _FakeTransport.spawned[0].closed


@pytest.mark.asyncio
async def test_a_spawn_the_turn_cancels_midway_is_stopped(mocker):
    entered = asyncio.Event()

    class _HangingTransport(_FakeTransport):
        async def connect(self) -> None:
            entered.set()
            await asyncio.Event().wait()

    _FakeTransport.spawned = []
    mocker.patch.object(cli_prelaunch, "SubprocessCLITransport", _HangingTransport)

    start = asyncio.create_task(CLIPrelaunch.start(ClaudeAgentOptions(), "[test]"))
    await entered.wait()
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start

    assert _FakeTransport.spawned[0].closed


_REPLY = "PRELAUNCH-REPLY-5d1e"


async def _fake_anthropic(request: web.Request) -> web.StreamResponse:
    await request.read()
    response = web.StreamResponse(
        status=200, headers={"Content-Type": "text/event-stream"}
    )
    await response.prepare(request)
    await response.write(_text_stream("msg_1", _REPLY))
    await response.write_eof()
    return response


@pytest.mark.asyncio
async def test_the_spawned_cli_answers_the_query_it_was_spawned_ahead_of(
    tmp_path: Path,
):
    """The bundled CLI against a fake Messages API, with the client lifecycle
    ``_run_stream_attempt`` uses: the process spawned before the query was
    built is the one that answers it. No API key needed."""
    cli_path = _resolve_cli_path()
    if cli_path is None or not cli_path.is_file():
        pytest.skip("No Claude Code CLI binary available")
    app = web.Application()
    app.router.add_post("/v1/messages", _fake_anthropic)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    options = ClaudeAgentOptions(
        cli_path=str(cli_path),
        system_prompt="You are a test.",
        setting_sources=[],
        cwd=str(tmp_path),
        max_turns=1,
        env={
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            "ANTHROPIC_API_KEY": "sk-test-fake-key-not-real",
            "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude-config"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
    )
    try:
        prelaunch = await CLIPrelaunch.start(options, "[test]")
        assert prelaunch is not None
        await asyncio.sleep(0.2)  # the turn building its query
        transport = prelaunch.claim(options)
        assert transport is not None
        client = ClaudeSDKClient(options=options, transport=transport)
        await client.__aenter__()
        try:
            await client.query("ping")
            async with asyncio.timeout(90):
                received = [message async for message in client.receive_response()]
        finally:
            await client.__aexit__(None, None, None)
        await prelaunch.close()
    finally:
        await runner.cleanup()

    reply = " ".join(
        block.text
        for message in received
        if isinstance(message, AssistantMessage)
        for block in message.content
        if isinstance(block, TextBlock)
    )
    assert _REPLY in reply
    assert any(isinstance(message, ResultMessage) for message in received)
