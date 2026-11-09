"""Tests for the turn-closing registry."""

from backend.copilot import turn_end


def test_a_question_stops_the_turn_and_refuses_every_later_call():
    turn_end.request_end("sess-q")
    try:
        assert turn_end.is_ending("sess-q")
        assert turn_end.refusal("sess-q", "web_search") == turn_end.ENDED_REFUSAL
        assert turn_end.refusal("sess-q", "start_task") == turn_end.ENDED_REFUSAL
        assert turn_end.refusal("sess-other", "web_search") is None
        assert not turn_end.is_ending(None)
    finally:
        turn_end.clear("sess-q")
    assert not turn_end.is_ending("sess-q")


def test_a_started_task_allows_more_tasks_and_a_closing_line_only():
    turn_end.request_wrap_up("sess-task")
    try:
        assert not turn_end.is_ending("sess-task")
        assert turn_end.refusal("sess-task", "start_task") is None
        assert turn_end.refusal("sess-task", "web_search") == turn_end.WRAP_UP_REFUSAL
    finally:
        turn_end.clear("sess-task")


def test_a_question_after_a_task_still_stops_the_turn():
    turn_end.request_wrap_up("sess-both")
    turn_end.request_end("sess-both")
    turn_end.request_wrap_up("sess-both")
    try:
        assert turn_end.is_ending("sess-both")
    finally:
        turn_end.clear("sess-both")


def test_clearing_an_unknown_session_is_a_no_op():
    turn_end.clear("never-seen")
    assert turn_end.refusal("never-seen", "x") is None
