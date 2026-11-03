import asyncio
import logging
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import stream_registry, usage_activation_fixtures
from backend.copilot.constants import COMPACTION_TOOL_NAME
from backend.copilot.executor.processor import CoPilotProcessor
from backend.copilot.executor.scheduled_turn_alert import (
    UNCLASSIFIED,
    ScheduledTurnWatch,
    tool_error_type,
)
from backend.copilot.executor.utils import (
    CoPilotExecutionEntry,
    CoPilotLogMetadata,
    ScheduledTurnOrigin,
)
from backend.copilot.model import ChatSession
from backend.copilot.response_model import (
    StreamError,
    StreamTextDelta,
    StreamToolOutputAvailable,
    StreamTurnPaused,
)
from backend.copilot.scheduled_chats.outcomes import RunOutcome
from backend.copilot.scheduled_chats.streaks import advance
from backend.copilot.tools.models import ErrorResponse, ExecutionStartedResponse
from backend.copilot.turn_pause import PauseReason, TurnPause
from backend.integrations.codex.transport import CodexCredentialIntegrityError
from backend.monitoring.instrumentation import COPILOT_SCHEDULED_TURN_FAILURES

usage_snapshot = usage_activation_fixtures.usage_snapshot

_ALERT = "backend.copilot.executor.scheduled_turn_alert"
_PRISMA_ERROR = (
    "Client is not connected to the query engine, you must call `connect()` "
    "before attempting to query data."
)


@pytest.fixture(autouse=True)
def mock_stream_publication(mocker):
    mocker.patch(
        "backend.copilot.executor.processor.stream_registry.publish_chunk",
        new_callable=AsyncMock,
    )


@pytest.fixture(autouse=True)
def allowance_admits_every_turn(mocker):
    """The pickup budget gate has its own tests (processor_budget_gate_test)."""
    mocker.patch(
        "backend.copilot.executor.processor.refusal_for",
        AsyncMock(return_value=None),
    )


def _entry(scheduled: ScheduledTurnOrigin | None) -> CoPilotExecutionEntry:
    return CoPilotExecutionEntry(
        session_id="sess-weekly-report",
        turn_id="turn-1",
        user_id="user-1",
        message="Run the weekly report agent",
        is_user_message=False,
        scheduled=scheduled,
    )


def _weekly() -> ScheduledTurnOrigin:
    return ScheduledTurnOrigin(schedule_id="sched-weekly", cron="0 10 * * 1")


def _run_agent_failure() -> StreamToolOutputAvailable:
    """What the Monday turns produced: run_agent caught the Prisma error and
    answered with an ErrorResponse, so the tool call still "succeeded"."""
    return StreamToolOutputAvailable(
        toolCallId="call-1",
        toolName="run_agent",
        output=ErrorResponse(
            message=f"Failed to process request: {_PRISMA_ERROR}",
            error=_PRISMA_ERROR,
            session_id="sess-weekly-report",
        ).model_dump_json(),
    )


def _run_agent_success() -> StreamToolOutputAvailable:
    return StreamToolOutputAvailable(
        toolCallId="call-1",
        toolName="run_agent",
        output=ExecutionStartedResponse(
            message="started",
            session_id="sess-weekly-report",
            execution_id="exec-1",
            graph_id="graph-1",
            graph_name="Weekly Report",
        ).model_dump_json(),
    )


def _failures(reason: str, tool: str) -> float:
    return COPILOT_SCHEDULED_TURN_FAILURES.labels(reason=reason, tool=tool)._value.get()


@pytest.fixture(autouse=True)
def database_manager() -> Iterator[MagicMock]:
    """Where a scheduled turn records its outcome. Patched for every test so
    none waits on a DatabaseManager that is not running."""
    client = MagicMock(record_scheduled_run=AsyncMock())
    with patch(f"{_ALERT}.get_database_manager_async_client", return_value=client):
        yield client


def _twitter_refused() -> StreamToolOutputAvailable:
    return StreamToolOutputAvailable(
        toolCallId="call-2",
        toolName="run_block",
        output=ErrorResponse(
            message="Twitter rejected the saved credential (HTTP 401).",
            error="credential_rejected",
            details={"provider": "twitter"},
            session_id="sess-weekly-report",
        ).model_dump_json(),
    )


def test_the_watch_sees_a_refused_connection_as_one_failing_step():
    """The briefing case: one step refused, the rest worked."""
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    watch.observe(_run_agent_success())
    watch.observe(_twitter_refused())

    outcome = watch.outcome(None)

    assert outcome.failure is not None
    assert outcome.failure.key == "connection:twitter"
    assert not outcome.whole_run


def test_a_compacted_history_is_not_work_done():
    """The every-minute monitor has a long history, so its runs compact. A
    run whose only other output is the failure still failed outright."""
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    watch.observe(
        StreamToolOutputAvailable(
            toolCallId="compaction", toolName=COMPACTION_TOOL_NAME, output="done"
        )
    )
    watch.observe(_twitter_refused())

    outcome = watch.outcome(None)

    assert outcome.failure is not None and outcome.whole_run


def test_a_cli_tool_that_returns_an_error_payload_is_not_a_failure():
    """Read can return a file that holds an ErrorResponse; that is the file,
    not the run failing."""
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    watch.observe(
        StreamToolOutputAvailable(
            toolCallId="c",
            toolName="Read",
            output=ErrorResponse(
                message="refused", error="credential_rejected"
            ).model_dump_json(),
        )
    )

    assert watch.tool_failures == []


def _looked_up(tool: str) -> StreamToolOutputAvailable:
    return StreamToolOutputAvailable(
        toolCallId=f"call-{tool}", toolName=tool, output='{"type": "capabilities"}'
    )


def _discord_not_connected() -> StreamToolOutputAvailable:
    return StreamToolOutputAvailable(
        toolCallId="call-run",
        toolName="run_capability",
        output=ErrorResponse(
            message="This chat has no Discord credential connected.",
            error="missing_credentials",
            details={"provider": "discord"},
            session_id="sess-weekly-report",
        ).model_dump_json(),
    )


def _discord_monitor_run(*, sent: bool = False) -> RunOutcome:
    """The every-minute Discord monitor: the model finds the block and reads
    its schema before it calls it, then the call fails or works."""
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    watch.observe(_looked_up("find_capability"))
    watch.observe(_looked_up("describe_capability"))
    watch.observe(_run_agent_success() if sent else _discord_not_connected())
    return watch.outcome(None)


def test_looking_up_the_step_that_failed_is_not_work_done():
    """Finding the Discord block and reading its schema did none of the
    work, so a run whose only real step failed failed outright."""
    outcome = _discord_monitor_run()

    assert outcome.failure is not None
    assert outcome.failure.key == "connection:discord"
    assert outcome.whole_run


def test_finding_the_agent_that_then_failed_is_not_work_done():
    """The weekly report finds its agent, then run_agent fails (SECRT-2799):
    the run did none of its work, so it failed outright."""
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    watch.observe(_looked_up("find_library_agent"))
    watch.observe(_run_agent_failure())

    outcome = watch.outcome(None)

    assert outcome.failure is not None and outcome.whole_run


def test_the_discord_monitor_pauses_on_its_third_run():
    streak = None
    steps = []
    for _ in range(3):
        step = advance(streak, _discord_monitor_run())
        streak = step.streak
        steps.append(step)

    assert [s.pause for s in steps] == [False, False, True]


def test_a_monitor_run_that_sends_clears_its_streak():
    step = advance(None, _discord_monitor_run())
    step = advance(step.streak, _discord_monitor_run())

    assert advance(step.streak, _discord_monitor_run(sent=True)).streak is None


def test_a_run_that_only_looked_things_up_tried_nothing():
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    watch.observe(_looked_up("find_capability"))
    watch.observe(_looked_up("describe_capability"))

    assert watch.outcome(None).idle


def test_the_watch_counts_only_platform_tools_as_trying():
    """A run that only read files with the CLI's own tools tried nothing."""
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    watch.observe(
        StreamToolOutputAvailable(toolCallId="c", toolName="Read", output="contents")
    )

    assert watch.outcome(None).idle


def test_interactive_turns_are_not_watched():
    assert ScheduledTurnWatch.for_entry(_entry(None)) is None


def _paused(reason: PauseReason, *, started: bool = True) -> StreamTurnPaused:
    return StreamTurnPaused(
        pause=TurnPause(
            reason=reason, checkpoint_saved=started, started=started
        ).as_part()
    )


def _refused_while_landing() -> StreamToolOutputAvailable:
    """A tool call refused once the run had to land: the refusal reaches the
    model as a failed tool result."""
    return StreamToolOutputAvailable(
        toolCallId="call-refused",
        toolName="run_agent",
        output="Tool calls are paused: this task has used the budget it has for now.",
        success=False,
    )


def _landed_run(reason: PauseReason) -> ScheduledTurnWatch:
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    watch.observe(_refused_while_landing())
    watch.observe(_paused(reason))
    return watch


@pytest.mark.parametrize(
    ("reason", "key"),
    [("task", "turn:max_budget_exhausted"), ("steps", "turn:max_turns_exhausted")],
)
def test_a_run_that_lands_at_its_own_limit_ran_out_as_before(reason, key):
    """Its own budget and step limit pause a run now, where they used to end
    it with an error. Three in a row still pause the schedule."""
    outcome = _landed_run(reason).outcome(None)

    assert outcome.failure is not None
    assert outcome.failure.key == key and outcome.failure.kind == "budget"
    assert outcome.whole_run


@pytest.mark.parametrize("reason", ["daily", "weekly", "trial"])
def test_a_run_paused_by_the_accounts_allowance_counts_for_nothing(reason):
    """The owner's allowance ran out mid-run: their state, not the
    schedule's, and the runs after it say so as skips. The tool calls the
    landing refused are no failure either."""
    watch = _landed_run(reason)

    assert watch.outcome(None).neutral
    with patch(f"{_ALERT}.sentry_sdk") as sentry:
        watch.report(None)
    sentry.capture_message.assert_not_called()


def test_a_run_at_its_own_limit_alerts_as_that_limit_not_as_a_tool():
    watch = _landed_run("task")

    with patch(f"{_ALERT}.sentry_sdk") as sentry:
        watch.report(None)

    sentry.capture_message.assert_called_once_with(
        "Scheduled copilot turn failed: the turn errored", level="error"
    )
    scope = sentry.new_scope.return_value.__enter__.return_value
    _, context = scope.set_context.call_args.args
    assert context["turn_error_type"] == "max_budget_exhausted"
    assert context["tool_failures"] == []


def test_tool_error_type_reads_error_payloads_only():
    assert tool_error_type(_run_agent_failure().output) == UNCLASSIFIED
    assert (
        tool_error_type({"type": "error", "error": "library_agent_not_found"})
        == "library_agent_not_found"
    )
    assert tool_error_type({"type": "error", "message": "nope"}) == UNCLASSIFIED
    assert tool_error_type(_run_agent_success().output) is None
    assert tool_error_type("plain text from a built-in tool") is None


def test_scheduled_turn_whose_tool_errors_raises_one_alert():
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    before = _failures("tool_error", "run_agent")

    with patch(f"{_ALERT}.sentry_sdk") as sentry:
        watch.observe(StreamTextDelta(id="t", delta="Running the report"))
        watch.observe(_run_agent_failure())
        watch.report(None)

    sentry.capture_message.assert_called_once_with(
        "Scheduled copilot turn failed: run_agent returned an error", level="error"
    )
    scope = sentry.new_scope.return_value.__enter__.return_value
    scope.set_tag.assert_called_once_with(
        "copilot_scheduled_turn_failure", "tool_error"
    )
    _, context = scope.set_context.call_args.args
    assert context["schedule_id"] == "sched-weekly"
    assert context["cron"] == "0 10 * * 1"
    assert context["session_id"] == "sess-weekly-report"
    assert context["tool_failures"] == [
        {"tool": "run_agent", "error_type": UNCLASSIFIED}
    ]
    assert _failures("tool_error", "run_agent") == before + 1


def test_neither_the_log_nor_sentry_carries_error_text(caplog):
    """A tool's error can quote the user's data; only codes leave the turn."""
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    private = "Customer list for jane@example.com could not be read"

    with (
        patch(f"{_ALERT}.sentry_sdk") as sentry,
        caplog.at_level(logging.WARNING, _ALERT),
    ):
        watch.observe(
            StreamToolOutputAvailable(
                toolCallId="call-1",
                toolName="run_agent",
                output=ErrorResponse(message=private, error=private).model_dump_json(),
            )
        )
        watch.observe(
            StreamToolOutputAvailable(
                toolCallId="call-2",
                toolName="run_block",
                output=ErrorResponse(
                    message=private, error="block_not_found"
                ).model_dump_json(),
            )
        )
        watch.report(None)
        watch.report(private)

    assert "failed tools: run_agent (unclassified), run_block (block_not_found)" in (
        caplog.text
    )
    assert "turn error: unclassified" in caplog.text
    assert "jane@example.com" not in caplog.text
    sent = str(sentry.mock_calls)
    assert "block_not_found" in sent
    assert "jane@example.com" not in sent


def test_a_platform_tool_failing_without_an_error_payload_still_counts():
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None

    with patch(f"{_ALERT}.sentry_sdk") as sentry:
        watch.observe(
            StreamToolOutputAvailable(
                toolCallId="call-1",
                toolName="Read",
                output="File does not exist.",
                success=False,
            )
        )
        watch.observe(
            StreamToolOutputAvailable(
                toolCallId="call-2",
                toolName="read_workspace_file",
                output="File does not exist.",
                success=False,
            )
        )
        watch.report(None)

    assert [f.model_dump() for f in watch.tool_failures] == [
        {"tool": "read_workspace_file", "error_type": UNCLASSIFIED}
    ]
    sentry.capture_message.assert_called_once_with(
        "Scheduled copilot turn failed: read_workspace_file returned an error",
        level="error",
    )


def test_scheduled_turn_that_errors_raises_an_alert():
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    before = _failures("turn_error", "")

    with patch(f"{_ALERT}.sentry_sdk") as sentry:
        watch.report("copilot_session_not_found")

    sentry.capture_message.assert_called_once_with(
        "Scheduled copilot turn failed: the turn errored", level="error"
    )
    assert _failures("turn_error", "") == before + 1


@pytest.mark.parametrize("turn_error", [None, stream_registry.CANCELLED_MESSAGE])
def test_healthy_or_cancelled_scheduled_turn_stays_quiet(turn_error):
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None

    with patch(f"{_ALERT}.sentry_sdk") as sentry:
        watch.observe(_run_agent_success())
        watch.report(turn_error)

    sentry.capture_message.assert_not_called()


def test_a_broken_alert_never_breaks_the_turn_cleanup():
    watch = ScheduledTurnWatch.for_entry(_entry(_weekly()))
    assert watch is not None
    watch.observe(_run_agent_failure())

    with patch(f"{_ALERT}.sentry_sdk") as sentry:
        sentry.capture_message.side_effect = RuntimeError("sentry down")
        watch.report(None)


class _Stream:
    def __init__(self, events):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self) -> None:
        pass


async def _run_turn(
    entry: CoPilotExecutionEntry, events, mark_done: AsyncMock | None = None
) -> MagicMock:
    with (
        patch(
            "backend.copilot.executor.processor.ChatConfig",
            return_value=MagicMock(test_mode=True, use_claude_agent_sdk=True),
        ),
        patch(
            "backend.copilot.executor.processor.stream_chat_completion_dummy",
            return_value=MagicMock(),
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.stream_and_publish",
            return_value=_Stream(events),
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            new=mark_done or AsyncMock(),
        ),
        patch(
            "backend.copilot.model.get_chat_session",
            new=AsyncMock(return_value=ChatSession.new("user-1", dry_run=False)),
        ),
        patch(f"{_ALERT}.sentry_sdk") as sentry,
    ):
        await CoPilotProcessor()._execute_async(
            entry,
            threading.Event(),
            MagicMock(),
            CoPilotLogMetadata(logger=logging.getLogger("test-copilot")),
        )
    return sentry


@pytest.mark.asyncio
async def test_executor_alerts_when_a_scheduled_turns_run_agent_fails():
    sentry = await _run_turn(_entry(_weekly()), [_run_agent_failure()])

    sentry.capture_message.assert_called_once_with(
        "Scheduled copilot turn failed: run_agent returned an error", level="error"
    )


@pytest.mark.asyncio
async def test_executor_alerts_a_scheduled_turns_error_by_its_code():
    spending_limit = StreamError(
        errorText="This turn reached its spending limit. Start a new turn.",
        code="max_budget_exhausted",
    )

    sentry = await _run_turn(_entry(_weekly()), [spending_limit])

    scope = sentry.new_scope.return_value.__enter__.return_value
    _, context = scope.set_context.call_args.args
    assert context["turn_error_type"] == "max_budget_exhausted"
    assert "spending limit" not in str(sentry.mock_calls)


@pytest.mark.asyncio
async def test_executor_does_not_alert_for_an_interactive_turn():
    sentry = await _run_turn(_entry(None), [_run_agent_failure()])

    sentry.capture_message.assert_not_called()


@pytest.mark.asyncio
async def test_executor_records_a_scheduled_turns_outcome(database_manager):
    """Every scheduled run counts toward its schedule's failure streak; the
    error's text never leaves the executor, only a hash of it."""
    await _run_turn(_entry(_weekly()), [_run_agent_failure()])

    database_manager.record_scheduled_run.assert_awaited_once()
    run = database_manager.record_scheduled_run.await_args.args[0]
    assert run.user_id == "user-1"
    assert run.schedule_key == "sched-weekly"
    assert run.session_id == "sess-weekly-report"
    assert run.title == "Run the weekly report agent"
    assert run.outcome.whole_run
    assert run.outcome.failure.key.startswith("tool:run_agent:")
    assert "query engine" not in run.model_dump_json()


@pytest.mark.asyncio
async def test_executor_records_after_the_turn_is_marked_done(database_manager):
    """The owner never waits on the streak: the turn is done first."""
    order = MagicMock()
    database_manager.record_scheduled_run.side_effect = lambda *_: order.record()
    mark_done = AsyncMock(side_effect=lambda *_, **__: order.mark_done())

    await _run_turn(_entry(_weekly()), [_run_agent_failure()], mark_done=mark_done)

    assert [c[0] for c in order.mock_calls] == ["mark_done", "record"]


@pytest.mark.asyncio
async def test_executor_records_nothing_for_an_interactive_turn(database_manager):
    await _run_turn(_entry(None), [_run_agent_failure()])

    database_manager.record_scheduled_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_broken_record_never_breaks_the_turn_cleanup(database_manager, caplog):
    database_manager.record_scheduled_run.side_effect = RuntimeError("db manager down")

    with caplog.at_level(logging.WARNING, logger=_ALERT):
        await _run_turn(_entry(_weekly()), [_run_agent_success()])

    assert "toward its schedule's failure streak" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", ["resume", "send"])
async def test_a_chats_own_turn_at_its_reset_alerts_but_has_no_streak(
    database_manager, purpose
):
    """A turn resumed or sent at the budget reset is the user's own chat, not
    a schedule: it can fail unwatched, so it alerts, but there is nothing to
    pause and no streak to count toward."""
    entry = _entry(
        ScheduledTurnOrigin(schedule_id=f"at-reset-{purpose}-1", purpose=purpose)
    )

    sentry = await _run_turn(entry, [_run_agent_failure()])

    sentry.capture_message.assert_called_once()
    database_manager.record_scheduled_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_cancelled_record_still_closes_the_turns_browser(database_manager):
    """Recording waits on the DatabaseManager and can be cancelled meanwhile,
    as a shutdown cancels the turn. The cleanups after it still run, or the
    turn's Chromium tree outlives it on the pod."""
    database_manager.record_scheduled_run.side_effect = asyncio.CancelledError()

    with patch(
        "backend.copilot.executor.processor.close_browser_daemon",
        new=AsyncMock(return_value=False),
    ) as close_browser:
        with pytest.raises(asyncio.CancelledError):
            await _run_turn(_entry(_weekly()), [_run_agent_success()])

    close_browser.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_scheduled_turn_refused_at_pickup_counts_for_nothing(database_manager):
    """The allowance ran out between the schedule firing and the turn being
    picked up, so the turn never ran. Read as a run that tried nothing, it
    would continue an open connection failure and could finish pausing a
    monitor for a cause it never reached."""
    not_started = TurnPause(reason="daily", checkpoint_saved=False, started=False)

    with (
        patch(
            "backend.copilot.executor.processor.refusal_for",
            AsyncMock(return_value=not_started),
        ),
        patch("backend.copilot.budget_gate.upsert_chat_session", AsyncMock()),
    ):
        await _run_turn(_entry(_weekly()), [])

    database_manager.record_scheduled_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_executor_alerts_when_the_codex_checkpoint_fails_after_a_scheduled_turn():
    session = ChatSession.new(
        "user-1", dry_run=False, llm_auth_provider="codex", llm_credential_id="cred-1"
    )
    lease = MagicMock()
    lease.credentials = SimpleNamespace(type="oauth2", id="cred-1")
    lease.release = AsyncMock(
        side_effect=CodexCredentialIntegrityError("codex_credential_checkpoint_failed")
    )
    transport = MagicMock()
    transport.acquire_runtime_lease = AsyncMock(return_value=lease)
    mark_completed = AsyncMock()
    entry = _entry(_weekly()).model_copy(
        update={"llm_auth_provider": "codex", "llm_credential_id": "cred-1"}
    )
    before = _failures("turn_error", "")

    with (
        patch(
            "backend.copilot.model.get_chat_session",
            new=AsyncMock(return_value=session),
        ),
        patch(
            "backend.integrations.codex.transport.get_codex_transport",
            return_value=transport,
        ),
        patch("backend.integrations.codex.credential_codec.bundle_from_credentials"),
        patch(
            "backend.copilot.sdk.service.stream_chat_completion_sdk",
            return_value=MagicMock(),
        ),
        patch(
            "backend.copilot.executor.processor.wrap_stream_with_heartbeat",
            return_value=MagicMock(),
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.stream_and_publish",
            return_value=_Stream([_run_agent_success()]),
        ),
        patch(
            "backend.copilot.executor.processor.stream_registry.mark_session_completed",
            mark_completed,
        ),
        patch(f"{_ALERT}.sentry_sdk") as sentry,
    ):
        await CoPilotProcessor()._execute_async(
            entry,
            threading.Event(),
            MagicMock(),
            CoPilotLogMetadata(logger=logging.getLogger("test-copilot")),
        )

    mark_completed.assert_awaited_once_with(
        entry.session_id,
        error_message="codex_credential_checkpoint_failed",
        error_code=None,
        turn_id=entry.turn_id,
    )
    sentry.capture_message.assert_called_once_with(
        "Scheduled copilot turn failed: the turn errored", level="error"
    )
    assert _failures("turn_error", "") == before + 1
