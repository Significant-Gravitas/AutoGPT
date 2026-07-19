"""Tests for the baseline turn's engine-switch exit behavior.

The tool loop breaks at the iteration boundary when a switch is pending
(engine_switch.is_pending), and the stream's terminal events tell the
frontend to flip its mode picker before StreamFinish.
"""

from backend.copilot import engine_switch
from backend.copilot.baseline.service import _engine_switch_finish_events
from backend.copilot.response_model import StreamModeChanged, StreamStatus


def _register(session_id: str) -> None:
    engine_switch.request_switch(session_id, user_id="user-1")


def test_no_events_without_pending_switch():
    assert _engine_switch_finish_events("sess-none") == []


def test_pending_switch_emits_mode_change_then_status():
    _register("sess-exit-1")
    try:
        events = _engine_switch_finish_events("sess-exit-1")
        assert isinstance(events[0], StreamModeChanged)
        assert events[0].mode == "extended_thinking"
        assert isinstance(events[1], StreamStatus)
        assert "Thinking" in events[1].message
    finally:
        engine_switch.pop_switch("sess-exit-1")


def test_loop_exit_predicate_matches_registry_state():
    """The tool loop breaks on the same predicate the finish events use —
    pending after register, cleared after pop."""
    assert engine_switch.is_pending("sess-exit-2") is False
    _register("sess-exit-2")
    try:
        assert engine_switch.is_pending("sess-exit-2") is True
    finally:
        engine_switch.pop_switch("sess-exit-2")
    assert engine_switch.is_pending("sess-exit-2") is False
