"""Tests for ``copilot.turn_tracing``: how a turn's end reaches PostHog."""

from unittest.mock import MagicMock

import pytest
import pytest_mock

from backend.copilot import ai_spans
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.response_model import StreamToolInputAvailable
from backend.copilot.tree import TurnEnvelope
from backend.copilot.turn_tracing import turn_transaction
from backend.util.posthog_events import PostHogEvent

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
