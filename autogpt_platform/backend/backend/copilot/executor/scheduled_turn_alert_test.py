import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import stream_registry
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
from backend.copilot.response_model import StreamTextDelta, StreamToolOutputAvailable
from backend.copilot.tools.models import ErrorResponse, ExecutionStartedResponse
from backend.integrations.codex.transport import CodexCredentialIntegrityError
from backend.monitoring.instrumentation import COPILOT_SCHEDULED_TURN_FAILURES

_ALERT = "backend.copilot.executor.scheduled_turn_alert"
_PRISMA_ERROR = (
    "Client is not connected to the query engine, you must call `connect()` "
    "before attempting to query data."
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


def test_interactive_turns_are_not_watched():
    assert ScheduledTurnWatch.for_entry(_entry(None)) is None


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


async def _run_turn(entry: CoPilotExecutionEntry, events) -> MagicMock:
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
            new=AsyncMock(),
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
async def test_executor_does_not_alert_for_an_interactive_turn():
    sentry = await _run_turn(_entry(None), [_run_agent_failure()])

    sentry.capture_message.assert_not_called()


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
        turn_id=entry.turn_id,
    )
    sentry.capture_message.assert_called_once_with(
        "Scheduled copilot turn failed: the turn errored", level="error"
    )
    assert _failures("turn_error", "") == before + 1
