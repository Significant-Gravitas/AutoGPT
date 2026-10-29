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
    assert transport.connected
    # A retry rebuilds its options, so it gets nothing and spawns its own.
    assert prelaunch.claim(ClaudeAgentOptions()) is None
    assert prelaunch.claim(options) is transport
    assert prelaunch.claim(options) is None


@pytest.mark.asyncio
async def test_the_process_starts_from_the_environment_every_launch_gets(
    spawned, monkeypatch
):
    monkeypatch.setenv("SOME_PLATFORM_SECRET", "sk-FAKE-platform-secret")
    options = ClaudeAgentOptions(env={"ANTHROPIC_API_KEY": "sk-ant-the-routes-key"})

    await CLIPrelaunch.start(options, "[test]")

    [transport] = spawned
    assert transport.options.env["ANTHROPIC_API_KEY"] == "sk-ant-the-routes-key"
    assert transport.options.env["SOME_PLATFORM_SECRET"] == ""
    assert options.env == {"ANTHROPIC_API_KEY": "sk-ant-the-routes-key"}


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


class _SlowToStopTransport(_FakeTransport):
    """Its close waits until the test lets it finish, like the SDK's close
    waiting for the process to exit before it kills it."""

    stopping: ClassVar[asyncio.Event]
    may_finish: ClassVar[asyncio.Event]

    async def close(self) -> None:
        _SlowToStopTransport.stopping.set()
        await _SlowToStopTransport.may_finish.wait()
        self.closed = True


@pytest.fixture
def slow_to_stop(mocker) -> list[_FakeTransport]:
    _FakeTransport.spawned = []
    _SlowToStopTransport.stopping = asyncio.Event()
    _SlowToStopTransport.may_finish = asyncio.Event()
    mocker.patch.object(cli_prelaunch, "SubprocessCLITransport", _SlowToStopTransport)
    return _FakeTransport.spawned


async def _stopping_begins() -> None:
    # Bounded, so a cleanup that never starts the stop fails instead of hanging.
    await asyncio.wait_for(_SlowToStopTransport.stopping.wait(), timeout=5)


async def _let_the_stop_finish() -> None:
    _SlowToStopTransport.may_finish.set()
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_cancelled_spawn_is_stopped_when_cancelled_again_while_stopping(
    slow_to_stop, mocker
):
    entered = asyncio.Event()

    async def _hang(self) -> None:
        entered.set()
        await asyncio.Event().wait()

    mocker.patch.object(_SlowToStopTransport, "connect", _hang)
    start = asyncio.create_task(CLIPrelaunch.start(ClaudeAgentOptions(), "[test]"))
    await entered.wait()
    start.cancel()
    await _stopping_begins()
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start

    await _let_the_stop_finish()
    assert slow_to_stop[0].closed


@pytest.mark.asyncio
async def test_an_unclaimed_process_is_stopped_when_the_turn_is_cancelled_while_stopping(
    slow_to_stop,
):
    prelaunch = await CLIPrelaunch.start(ClaudeAgentOptions(), "[test]")
    assert prelaunch is not None

    closing = asyncio.create_task(prelaunch.close())
    await _stopping_begins()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    await _let_the_stop_finish()
    assert slow_to_stop[0].closed


@pytest.mark.asyncio
async def test_the_bundled_cli_unclaimed_is_stopped_without_cancelling_the_turn(
    tmp_path: Path,
):
    """The real transport, stopped from the task ``_stop`` makes: its process
    exits, and nothing is left to cancel the task that asked for the stop."""
    cli_path = _resolve_cli_path()
    if cli_path is None or not cli_path.is_file():
        pytest.skip("No Claude Code CLI binary available")
    options = ClaudeAgentOptions(
        cli_path=str(cli_path),
        setting_sources=[],
        cwd=str(tmp_path),
        env={
            "ANTHROPIC_API_KEY": "sk-test-fake-key-not-real",
            "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude-config"),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
    )
    prelaunch = await CLIPrelaunch.start(options, "[test]")
    assert prelaunch is not None
    process = prelaunch._transport._process
    assert process is not None and process.returncode is None

    await prelaunch.close()
    await asyncio.sleep(0.1)

    turn = asyncio.current_task()
    assert turn is not None and turn.cancelling() == 0
    assert process.returncode is not None


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
