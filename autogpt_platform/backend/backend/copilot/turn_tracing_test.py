"""Tests for ``copilot.turn_tracing``: when a turn's first token and first
text are timed, and how its end reaches PostHog."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_mock

from backend.copilot import ai_spans, turn_tracing
from backend.copilot.constants import (
    BUDGET_EXHAUSTED_NOTICE,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    EMPTY_FINISH_NOTICE,
    NO_COMMENTARY_NOTICE,
    STREAM_INCOMPLETE_MARKER,
)
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.response_model import (
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolInputStart,
)
from backend.copilot.tree import TurnEnvelope
from backend.copilot.turn_tracing import turn_transaction
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
    EMPTY_FINISH_NOTICE,
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
    turn.observe(StreamTextDelta(id="t", delta=f"It said: {EMPTY_FINISH_NOTICE}"))

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


def test_a_turn_that_never_reached_its_engine_sends_nothing(capture: MagicMock):
    """A refused or misrouted turn made no model or tool calls."""
    with turn_transaction(_entry()) as turn:
        turn.set_outcome("copilot_session_not_found", CANCELLED)

    capture.assert_not_called()
