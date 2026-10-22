"""A turn runs its first launch on the CLI it spawned ahead of the query, a
retry spawns its own, and a turn that ends before any launch stops it."""

import contextlib
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from backend.copilot.response_model import StreamError, StreamStart
from backend.copilot.sdk import cli_prelaunch
from backend.copilot.sdk import retry_scenarios_test as scenarios
from backend.copilot.sdk.cli_prelaunch import CLIPrelaunch
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.service import stream_chat_completion_sdk

_SVC = "backend.copilot.sdk.service"
_turns = scenarios.TestStreamChatCompletionRetryIntegration()


class _FakeTransport:
    spawned: ClassVar[list["_FakeTransport"]] = []

    def __init__(self, prompt, options: ClaudeAgentOptions) -> None:
        self.options = options
        self.closed = False
        self.stderr_errors: list[Exception] = []
        _FakeTransport.spawned.append(self)

    async def connect(self) -> None:
        # The SDK reads the CLI's stderr from the spawn on, so a line the CLI
        # writes while it boots reaches the turn's callback right away; the
        # SDK logs and drops whatever the callback raises, as this does.
        if self.options.stderr is not None:
            try:
                self.options.stderr("a line the CLI writes while it boots")
            except Exception as error:
                self.stderr_errors.append(error)

    async def close(self) -> None:
        self.closed = True


async def _run_turn(client_factory, message: str = "hello") -> list[Any]:
    _FakeTransport.spawned = []
    session = _turns._make_session()
    # The route saves the message before the turn runs, so the turn finds it
    # already in the session and doesn't save it again.
    session.messages[-1].content = message
    patches = scenarios._make_sdk_patches(
        session,
        original_transcript=build_test_transcript(
            [("user", "prior question"), ("assistant", "prior answer")]
        ),
        compacted_transcript=build_test_transcript(
            [("user", "[summary]"), ("assistant", "summary reply")]
        ),
        client_side_effect=client_factory,
    )
    events = []
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        # The real prelaunch (the conftest stubs it out) over a fake process.
        stack.enter_context(patch(f"{_SVC}.CLIPrelaunch", CLIPrelaunch))
        stack.enter_context(
            patch.object(cli_prelaunch, "SubprocessCLITransport", _FakeTransport)
        )
        async for event in stream_chat_completion_sdk(
            session_id="test-session-id",
            message=message,
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            events.append(event)
    return events


@pytest.mark.asyncio
async def test_the_first_launch_runs_on_the_cli_spawned_ahead_of_the_query():
    launches: list[dict] = []

    def _client(*args, **kwargs):
        launches.append(kwargs)
        return _turns._make_client_mock(result_message=_turns._make_result_message())

    events = await _run_turn(_client)

    [launch] = launches
    [spawned] = _FakeTransport.spawned
    assert launch["transport"] is spawned
    # The same options, each copy with the environment every launch gets.
    assert spawned.options == launch["options"]
    # The client that took it owns it: the turn's end doesn't stop it twice.
    assert not spawned.closed
    assert not [e for e in events if isinstance(e, StreamError)]
    assert any(isinstance(e, StreamStart) for e in events)


@pytest.mark.asyncio
async def test_the_cli_can_write_to_stderr_before_the_first_launch():
    def _client(*args, **kwargs):
        return _turns._make_client_mock(result_message=_turns._make_result_message())

    await _run_turn(_client)

    [spawned] = _FakeTransport.spawned
    assert spawned.stderr_errors == []


@pytest.mark.asyncio
async def test_a_retry_with_reduced_context_spawns_its_own_cli():
    launches: list[dict] = []

    def _client(*args, **kwargs):
        launches.append(kwargs)
        if len(launches) == 1:
            return _turns._make_client_mock(raises_on_enter=True)
        return _turns._make_client_mock(result_message=_turns._make_result_message())

    events = await _run_turn(_client)

    assert [launch["transport"] for launch in launches] == [
        _FakeTransport.spawned[0],
        None,
    ]
    assert launches[1]["options"] is not launches[0]["options"]
    assert not [e for e in events if isinstance(e, StreamError)]


@pytest.mark.asyncio
async def test_a_turn_that_ends_before_its_first_launch_stops_the_cli():
    launches: list[dict] = []

    def _client(*args, **kwargs):
        launches.append(kwargs)
        return _turns._make_client_mock(result_message=_turns._make_result_message())

    events = await _run_turn(_client, message="   ")

    assert launches == []
    [spawned] = _FakeTransport.spawned
    assert spawned.closed
    assert any(isinstance(e, StreamError) and e.code == "empty_prompt" for e in events)
