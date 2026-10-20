"""A turn runs its first launch on the CLI it spawned ahead of the query, a
retry spawns its own, a turn that ends before any launch stops it, and a
first turn's CLI boots while its memories are recalled."""

import asyncio
import contextlib
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

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
        _FakeTransport.spawned.append(self)

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


async def _run_turn(
    client_factory,
    message: str = "hello",
    transport: type[_FakeTransport] = _FakeTransport,
    extra: tuple[tuple[str, Any], ...] = (),
) -> list[Any]:
    _FakeTransport.spawned = []
    session = _turns._make_session()
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
            patch.object(cli_prelaunch, "SubprocessCLITransport", transport)
        )
        for target, new in extra:
            stack.enter_context(patch(target, new=new))
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
    assert spawned.options is launch["options"]
    # The client that took it owns it: the turn's end doesn't stop it twice.
    assert not spawned.closed
    assert not [e for e in events if isinstance(e, StreamError)]
    assert any(isinstance(e, StreamStart) for e in events)


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


@pytest.mark.asyncio
async def test_a_first_turns_cli_boots_while_its_memories_are_recalled():
    """The recalled memories go into the first user message, not the CLI's
    options, so the CLI is spawned before they arrive, and they still reach
    the message."""
    recalled = asyncio.Event()
    spawned_while_recalling: list[bool] = []

    async def _recall(user_id, message, expert_id=None):
        await recalled.wait()
        return "<memories>the user ships on Fridays</memories>"

    class _SpawnThenRecall(_FakeTransport):
        async def connect(self) -> None:
            spawned_while_recalling.append(not recalled.is_set())
            recalled.set()

    injected = AsyncMock(return_value=None)
    launches: list[dict] = []

    def _client(*args, **kwargs):
        launches.append(kwargs)
        return _turns._make_client_mock(result_message=_turns._make_result_message())

    events = await _run_turn(
        _client,
        transport=_SpawnThenRecall,
        extra=(
            (f"{_SVC}.is_enabled_for_user", AsyncMock(return_value=True)),
            (f"{_SVC}.fetch_warm_context", _recall),
            (f"{_SVC}.inject_user_context", injected),
        ),
    )

    assert spawned_while_recalling == [True]
    assert injected.await_args is not None
    assert (
        injected.await_args.kwargs["warm_ctx"]
        == "<memories>the user ships on Fridays</memories>"
    )
    assert launches[0]["transport"] is _FakeTransport.spawned[0]
    assert not [e for e in events if isinstance(e, StreamError)]
