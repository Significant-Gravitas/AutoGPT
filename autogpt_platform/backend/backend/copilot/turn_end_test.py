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
