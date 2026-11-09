"""Tests for the turn-ending registry."""

from backend.copilot import turn_end


def test_a_request_ends_only_its_own_session_until_cleared():
    turn_end.request_end("sess-a")

    assert turn_end.is_ending("sess-a")
    assert not turn_end.is_ending("sess-b")
    assert not turn_end.is_ending(None)

    turn_end.clear("sess-a")
    assert not turn_end.is_ending("sess-a")


def test_clearing_an_unknown_session_is_a_no_op():
    turn_end.clear("never-seen")
    assert not turn_end.is_ending("never-seen")


def test_a_started_task_ends_the_turn_but_lets_the_step_finish():
    turn_end.request_end("sess-task", refuse_later_calls=False)
    try:
        assert turn_end.is_ending("sess-task")
        assert not turn_end.refuses_calls("sess-task")

        turn_end.request_end("sess-task")
        assert turn_end.refuses_calls("sess-task")

        turn_end.request_end("sess-task", refuse_later_calls=False)
        assert turn_end.refuses_calls("sess-task")
    finally:
        turn_end.clear("sess-task")


def test_a_question_refuses_later_calls():
    turn_end.request_end("sess-q")
    try:
        assert turn_end.refuses_calls("sess-q")
    finally:
        turn_end.clear("sess-q")
