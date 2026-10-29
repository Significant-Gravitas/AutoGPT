"""A turn runs its first launch on the CLI it spawned ahead of the query, a
retry spawns its own, and a turn that ends before any launch stops it."""

import asyncio
import contextlib
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from backend.copilot.pending_messages import PendingMessage
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


@contextlib.contextmanager
def _patched_turn(client_factory, session, *, resolve=None, persist=None):
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
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        stack.enter_context(patch(f"{_SVC}.CLIPrelaunch", CLIPrelaunch))
        stack.enter_context(
            patch.object(cli_prelaunch, "SubprocessCLITransport", _FakeTransport)
        )
        for target, return_value in (
            ("_update_title_async", None),
            ("clear_pending_question", None),
            ("active_mode", None),
            ("build_builder_system_prompt_suffix", ""),
            ("build_session_context", ""),
            ("build_turn_budget_block", ""),
            ("persist_and_record_usage", None),
            ("capture_chat_turn", None),
        ):
            stack.enter_context(
                patch(
                    f"{_SVC}.{target}",
                    new_callable=AsyncMock,
                    return_value=return_value,
                )
            )
        stack.enter_context(
            patch(
                f"{_SVC}.inject_user_context",
                new=AsyncMock(
                    side_effect=lambda _understanding, message, *_args, **_kwargs: (
                        message
                    )
                ),
            )
        )
        if resolve is not None:
            stack.enter_context(patch(f"{_SVC}.resolve_answered", new=resolve))
        if persist is not None:
            stack.enter_context(
                patch(
                    "backend.copilot.pending_message_helpers.persist_pending_as_user_rows",
                    new=persist,
                )
            )
        yield


async def _run_turn(client_factory, message: str = "hello") -> list[Any]:
    _FakeTransport.spawned = []
    session = _turns._make_session()
    # The route saves the message before the turn runs, so the turn finds it
    # already in the session and doesn't save it again.
    session.messages[-1].content = message
    events = []
    with _patched_turn(client_factory, session):
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


@pytest.mark.asyncio
async def test_cancelling_unfolded_result_persistence_stops_the_unclaimed_cli():
    _FakeTransport.spawned = []
    session = _turns._make_session()
    session.messages[-1].content = "   "
    result = PendingMessage(
        content='<held_call_result tool="post_to_chat_platform">posted</held_call_result>'
    )
    persistence_started = asyncio.Event()
    persisted: list[list[PendingMessage]] = []
    launches: list[dict] = []

    async def resolve(_user_id, _session, deliver, **_kwargs):
        deliver(result)
        return
        yield

    async def persist(_session, _builder, pending, **_kwargs):
        persisted.append(list(pending))
        persistence_started.set()
        await asyncio.Event().wait()

    def client_factory(*args, **kwargs):
        launches.append(kwargs)
        return _turns._make_client_mock(result_message=_turns._make_result_message())

    async def run_turn() -> None:
        async for _ in stream_chat_completion_sdk(
            session_id="test-session-id",
            message="   ",
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            pass

    with _patched_turn(client_factory, session, resolve=resolve, persist=persist):
        turn = asyncio.create_task(run_turn())
        try:
            async with asyncio.timeout(10):
                await persistence_started.wait()
            turn.cancel()
            with pytest.raises(asyncio.CancelledError):
                async with asyncio.timeout(10):
                    await turn
        finally:
            if not turn.done():
                turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)

    assert launches == []
    assert persisted == [[result]]
    [spawned] = _FakeTransport.spawned
    assert spawned.closed


@pytest.mark.asyncio
async def test_failed_unfolded_result_persistence_stops_the_unclaimed_cli():
    _FakeTransport.spawned = []
    session = _turns._make_session()
    session.messages[-1].content = "   "
    result = PendingMessage(
        content='<held_call_result tool="post_to_chat_platform">posted</held_call_result>'
    )
    persisted: list[list[PendingMessage]] = []
    launches: list[dict] = []

    async def resolve(_user_id, _session, deliver, **_kwargs):
        deliver(result)
        return
        yield

    async def persist(_session, _builder, pending, **_kwargs):
        persisted.append(list(pending))
        raise RuntimeError("fake database write failure")

    def client_factory(*args, **kwargs):
        launches.append(kwargs)
        return _turns._make_client_mock(result_message=_turns._make_result_message())

    events = []
    with _patched_turn(client_factory, session, resolve=resolve, persist=persist):
        async for event in stream_chat_completion_sdk(
            session_id="test-session-id",
            message="   ",
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            events.append(event)

    assert launches == []
    assert persisted == [[result]]
    [spawned] = _FakeTransport.spawned
    assert spawned.closed
    assert any(isinstance(e, StreamError) and e.code == "empty_prompt" for e in events)
