"""Chat analytics: one event per turn, and no event without a user."""

from unittest.mock import Mock

import pytest

from backend.copilot import tracking
from backend.util import posthog_client


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: client)
    return client.capture


def test_a_chat_turn_is_one_event_carrying_the_message_length(capture: Mock) -> None:
    tracking.track_user_message("user-1", "session-1", 42)

    capture.assert_called_once()
    kwargs = capture.call_args.kwargs
    assert kwargs["event"] == "chat_message_sent"
    assert kwargs["properties"]["message_length"] == 42
    assert kwargs["properties"]["session_id"] == "session-1"


def test_an_expert_chat_turn_is_the_same_event_with_the_expert(capture: Mock) -> None:
    tracking.track_user_message("user-1", "session-1", 7, expert_id="expert-1")

    capture.assert_called_once()
    assert capture.call_args.kwargs["event"] == "chat_message_sent"
    assert capture.call_args.kwargs["properties"]["expert_id"] == "expert-1"


def test_a_turn_without_a_user_sends_nothing(capture: Mock) -> None:
    tracking.track_user_message(None, "session-1", 42)

    capture.assert_not_called()


def test_a_tool_call_carries_the_copilot_source(capture: Mock) -> None:
    tracking.track_tool_called("user-1", "session-1", "run_agent", "call-1")

    kwargs = capture.call_args.kwargs
    assert kwargs["distinct_id"] == "user-1"
    assert kwargs["event"] == "chat_tool_called"
    assert kwargs["properties"]["source"] == "chat_copilot"
    assert kwargs["properties"]["tool_name"] == "run_agent"


def test_a_tool_call_without_a_user_is_not_sent_under_a_made_up_id(
    capture: Mock,
) -> None:
    tracking.track_tool_called(None, "session-1", "run_agent", "call-1")

    capture.assert_not_called()


def test_the_library_check_outcome_carries_the_copilot_source(capture: Mock) -> None:
    tracking.track_library_check_outcome("user-1", "session-1", "no_matches")

    kwargs = capture.call_args.kwargs
    assert kwargs["event"] == "chat_library_check_outcome"
    assert kwargs["properties"]["source"] == "chat_copilot"
    assert kwargs["properties"]["outcome"] == "no_matches"


def test_a_chat_outcome_carries_its_type_and_session(capture: Mock) -> None:
    tracking.track_chat_outcome(
        "user-1", "session-1", "agent_run_success", graph_id="graph-1"
    )

    kwargs = capture.call_args.kwargs
    assert kwargs["distinct_id"] == "user-1"
    assert kwargs["event"] == "chat_outcome"
    assert kwargs["properties"]["outcome_type"] == "agent_run_success"
    assert kwargs["properties"]["session_id"] == "session-1"
    assert kwargs["properties"]["graph_id"] == "graph-1"
    assert kwargs["properties"]["source"] == "chat_copilot"
