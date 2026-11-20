"""Tests for ``copilot.turn_tracing``: when a turn's first token and first
text are timed, and how its end reaches PostHog."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_mock

from backend.copilot import ai_spans, turn_tracing
from backend.copilot.constants import (
    BUDGET_EXHAUSTED_NOTICE,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    EMPTY_COMPLETION_NOTICE,
    NO_COMMENTARY_NOTICE,
    STREAM_INCOMPLETE_MARKER,
)
from backend.copilot.executor.utils import CoPilotExecutionEntry, ScheduledTurnOrigin
from backend.copilot.model import ChatMessage
from backend.copilot.reply_timing import describe_chat
from backend.copilot.response_model import (
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolInputStart,
)
from backend.copilot.tree import TurnEnvelope
from backend.copilot.turn_tracing import TurnTrace, turn_origin, turn_transaction
from backend.util.posthog_events import PostHogEvent


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The turn's monotonic clock, set by the test before each chunk."""
    now = [100.0]
    monkeypatch.setattr(
        turn_tracing,
        "time",
        SimpleNamespace(monotonic=lambda: now[0], time=time.time),
    )
    return now


@pytest.fixture
def timings(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[float]]:
    recorded: dict[str, list[float]] = {}

    def set_timing(name: str, milliseconds: float, *, span=None) -> None:
        recorded.setdefault(name, []).append(milliseconds)

    monkeypatch.setattr(turn_tracing.tracing, "set_timing", set_timing)
    return recorded


def test_first_token_and_text_are_timed_at_their_first_chunks(clock, timings):
    turn = turn_tracing.TurnTrace(MagicMock())

    for at, chunk in [
        (100.5, StreamReasoningDelta(id="r", delta="thinking")),
        (101.0, StreamTextDelta(id="t", delta="Here")),
        (101.5, StreamToolInputStart(toolCallId="c", toolName="run_block")),
        (105.0, StreamTextDelta(id="t2", delta="Done")),
    ]:
        clock[0] = at
        turn.observe(chunk)

    assert timings["copilot.turn.ttft_ms"] == [500.0]
    assert timings["copilot.turn.first_text_ms"] == [1000.0]


NOTICES = [
    STREAM_INCOMPLETE_MARKER,
    EMPTY_COMPLETION_NOTICE,
    BUDGET_EXHAUSTED_NOTICE,
    NO_COMMENTARY_NOTICE,
    f"{COPILOT_RETRYABLE_ERROR_PREFIX} The provider is busy. Try again.",
]


@pytest.mark.parametrize("notice", NOTICES)
def test_a_notice_the_platform_streams_is_not_the_answer(clock, timings, notice):
    """A turn that ends without an answer streams a notice in its place, at
    the very end: timing it as the answer would put the turn's whole length
    in the first-text timing and call the turn one that answered first."""
    span = MagicMock()
    turn = turn_tracing.TurnTrace(span)

    for at, chunk in [
        (100.5, StreamReasoningDelta(id="r", delta="thinking")),
        (130.0, StreamTextDelta(id="n", delta=notice)),
    ]:
        clock[0] = at
        turn.observe(chunk)
    turn.close()

    assert timings["copilot.turn.ttft_ms"] == [500.0]
    assert "copilot.turn.first_text_ms" not in timings
    span.set_tag.assert_any_call("copilot.turn.shape", "none")


@pytest.mark.parametrize("notice", NOTICES)
def test_a_turn_whose_only_output_is_a_notice_has_no_first_token(
    clock, timings, notice
):
    turn = turn_tracing.TurnTrace(MagicMock())

    clock[0] = 130.0
    turn.observe(StreamTextDelta(id="n", delta=notice))

    assert "copilot.turn.ttft_ms" not in timings
    assert "copilot.turn.first_text_ms" not in timings


def test_an_answer_that_quotes_a_notice_is_still_the_answer(clock, timings):
    turn = turn_tracing.TurnTrace(MagicMock())

    clock[0] = 101.0
    turn.observe(StreamTextDelta(id="t", delta=f"It said: {EMPTY_COMPLETION_NOTICE}"))

    assert timings["copilot.turn.first_text_ms"] == [1000.0]


CANCELLED = "Operation cancelled"


@pytest.fixture
def capture(mocker: pytest_mock.MockerFixture) -> MagicMock:
    return mocker.patch.object(ai_spans.posthog_client, "capture")


def _entry() -> CoPilotExecutionEntry:
    return CoPilotExecutionEntry(
        session_id="sess-1",
        turn_id="turn-1",
        user_id="user-1",
        message="hi",
        envelope=TurnEnvelope(tree_id="turn-1"),
    )


def test_a_turn_that_ran_ends_as_its_replys_trace(capture: MagicMock):
    with turn_transaction(_entry()) as turn:
        turn.engine_started("baseline")
        turn.observe(
            StreamToolInputAvailable(toolCallId="c1", toolName="run_block", input={})
        )
        turn.set_outcome(None, CANCELLED)

    capture.assert_called_once()
    distinct_id, event, properties = capture.call_args.args
    assert (distinct_id, event) == ("user-1", PostHogEvent.AI_TRACE)
    assert properties["$ai_trace_id"] == "turn-1"
    assert properties["engine"] == "baseline"
    assert properties["outcome"] == "completed"
    assert properties["tool_call_count"] == 1
    assert properties["turn_origin"] == "chat"
    assert properties["$ai_latency"] >= 0


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "chat"),
        ({"is_user_message": False}, "system"),
        ({"envelope": TurnEnvelope(tree_id="turn-0", depth=1)}, "spawned"),
        ({"scheduled": ScheduledTurnOrigin(schedule_id="sched-1")}, "scheduled"),
        ({"unattended": True}, "scheduled"),
    ],
    ids=["a message", "the platform", "a subtask", "a schedule", "unattended"],
)
def test_turn_origin_says_what_started_the_turn(
    changes: dict[str, object], expected: str
):
    entry = _entry().model_copy(update=changes)

    assert turn_origin(entry) == expected


@pytest.mark.parametrize(
    ("changes", "session_origin", "expected"),
    [
        ({"tool_name": "autopilot_block"}, None, "automation"),
        ({"tool_name": "autopilot_block"}, "automation", "automation"),
        (
            {"tool_name": "autopilot_block", "is_user_message": False},
            None,
            "automation",
        ),
        ({"tool_name": "chat"}, "automation", "chat"),
        ({"tool_name": None}, "automation", "automation"),
        ({"tool_name": None}, "interactive", "chat"),
        (
            {
                "tool_name": "autopilot_block",
                "envelope": TurnEnvelope(tree_id="turn-0", depth=1),
            },
            None,
            "spawned",
        ),
        ({"tool_name": "autopilot_block", "unattended": True}, None, "scheduled"),
    ],
    ids=[
        "the block, legacy chat",
        "the block",
        "the block's system note",
        "someone typing in the block's chat",
        "an old entry in an automation chat",
        "an old entry in a typed chat",
        "a subtask",
        "unattended",
    ],
)
def test_a_workflows_autopilot_block_turn_is_automation(
    changes: dict[str, object], session_origin: str | None, expected: str
):
    entry = _entry().model_copy(update=changes)

    assert turn_origin(entry, session_origin) == expected  # type: ignore[arg-type]


def test_the_origin_the_executor_sets_is_the_one_the_trace_reports(
    capture: MagicMock,
):
    with turn_transaction(_entry()) as turn:
        turn.set_origin("automation")
        turn.engine_started("sdk")

    assert capture.call_args.args[2]["turn_origin"] == "automation"


def test_a_failed_turn_carries_its_error_code_not_its_message(capture: MagicMock):
    with turn_transaction(_entry()) as turn:
        turn.engine_started("sdk")
        turn.set_outcome("the user's private text", CANCELLED, "sdk_stream_error")

    properties = capture.call_args.args[2]
    assert properties["$ai_is_error"] is True
    assert properties["$ai_error"] == "sdk_stream_error"
    assert "private" not in repr(properties)


def test_a_turn_that_raised_still_ends_its_trace(capture: MagicMock):
    with pytest.raises(RuntimeError), turn_transaction(_entry()) as turn:
        turn.engine_started("sdk")
        turn.set_outcome("boom", CANCELLED)
        raise RuntimeError("boom")

    assert capture.call_args.args[2]["outcome"] == "failed"


@pytest.mark.parametrize(
    ("notice", "code"),
    [
        (STREAM_INCOMPLETE_MARKER, "stream_incomplete"),
        (EMPTY_COMPLETION_NOTICE, "empty_completion"),
    ],
    ids=["the CLI ended without a result", "the model finished without a word"],
)
def test_a_turn_that_ends_on_a_notice_in_place_of_a_reply_failed(
    capture: MagicMock, notice: str, code: str
):
    with turn_transaction(_entry()) as turn:
        turn.engine_started("sdk")
        turn.observe(StreamTextDelta(id="t", delta="Let me look into that."))
        turn.observe(StreamTextDelta(id="n", delta=notice))
        turn.set_outcome(None, CANCELLED)

    properties = capture.call_args.args[2]
    assert properties["outcome"] == "failed"
    assert properties["$ai_is_error"] is True
    assert properties["$ai_error"] == code


def test_a_reply_after_a_notice_completes_the_turn(capture: MagicMock):
    """The SDK engine carries a turn on with the user's queued messages after
    its CLI ended without a result."""
    with turn_transaction(_entry()) as turn:
        turn.engine_started("sdk")
        turn.observe(StreamTextDelta(id="n", delta=STREAM_INCOMPLETE_MARKER))
        turn.observe(StreamTextDelta(id="t", delta="On to your next message."))
        turn.set_outcome(None, CANCELLED)

    assert capture.call_args.args[2]["outcome"] == "completed"


def test_a_turn_stopped_after_a_notice_is_cancelled(capture: MagicMock):
    with turn_transaction(_entry()) as turn:
        turn.engine_started("sdk")
        turn.observe(StreamTextDelta(id="n", delta=STREAM_INCOMPLETE_MARKER))
        turn.set_outcome(CANCELLED, CANCELLED)

    assert capture.call_args.args[2]["outcome"] == "cancelled"


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (RuntimeError("admission failed"), "failed"),
        (asyncio.CancelledError(), "cancelled"),
    ],
    ids=["an error", "a cancellation"],
)
def test_a_turn_that_raised_before_its_outcome_still_ends_with_one(
    error: BaseException, outcome: str
):
    turns: list[TurnTrace] = []
    with pytest.raises(type(error)), turn_transaction(_entry()) as turn:
        turns.append(turn)
        raise error

    assert turns[0].outcome == outcome


def test_a_turn_that_never_reached_its_engine_sends_nothing(capture: MagicMock):
    """A refused or misrouted turn made no model or tool calls."""
    with turn_transaction(_entry()) as turn:
        turn.set_outcome("copilot_session_not_found", CANCELLED)

    capture.assert_not_called()


@pytest.mark.parametrize(
    "oldest_sequence, first",
    [
        # The whole chat, loaded from its first message.
        (0, "true"),
        # A long chat loaded without its oldest messages: the one user
        # message in the window isn't the chat's first.
        (1000, "false"),
    ],
)
def test_a_chat_loaded_without_its_oldest_messages_is_not_a_first_turn(
    oldest_sequence, first
):
    span = MagicMock()
    turn = turn_tracing.TurnTrace(span)
    messages = [
        ChatMessage(
            role="assistant", content="still working", sequence=oldest_sequence
        ),
        ChatMessage(role="user", content="next", sequence=oldest_sequence + 1),
    ]
    session = SimpleNamespace(messages=messages, expert_id=None)

    describe_chat(turn.span, session, "next")

    span.set_tag.assert_any_call("copilot.turn.first", first)


def test_a_load_test_accounts_turn_says_so(capture: MagicMock):
    with turn_transaction(_entry()) as turn:
        turn.load_test_account("dummy")
        turn.engine_started("dummy")
        turn.set_outcome(None, CANCELLED)

    properties = capture.call_args.args[2]
    assert properties["engine"] == "dummy"
    assert properties["load_test"] == "dummy"


def test_a_persons_turn_carries_no_load_test_property(capture: MagicMock):
    with turn_transaction(_entry()) as turn:
        turn.engine_started("sdk")
        turn.set_outcome(None, CANCELLED)

    assert "load_test" not in capture.call_args.args[2]


def test_marking_a_load_test_account_tags_its_turn():
    span = MagicMock()

    TurnTrace(span).load_test_account("scripted")

    span.set_tag.assert_called_once_with("copilot.load_test", "scripted")
