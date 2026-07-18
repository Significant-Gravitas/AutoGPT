"""Tests for the engine-switch continuation dispatch in the copilot manager.

The dispatch is the handoff between a finished baseline turn and the
server-initiated SDK continuation turn — the highest-risk link in the
engine-switch flow (see ``backend.copilot.engine_switch``). These tests
pin its retry/give-up contract.
"""

from unittest.mock import AsyncMock, patch

from backend.copilot.engine_switch import CONTINUATION_MESSAGE, SwitchRequest

from .manager import _SWITCH_DISPATCH_ATTEMPTS, _dispatch_engine_switch_continuation

_SWITCH = SwitchRequest(user_id="user-1", organization_id="org-1", team_id=None)


def test_dispatch_succeeds_first_try():
    with patch(
        "backend.copilot.executor.manager.schedule_turn", new_callable=AsyncMock
    ) as mock_schedule:
        _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    assert mock_schedule.await_count == 1
    kwargs = mock_schedule.call_args.kwargs
    assert kwargs["session_id"] == "sess-1"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["organization_id"] == "org-1"
    assert kwargs["message"] == CONTINUATION_MESSAGE
    assert kwargs["is_user_message"] is False
    assert kwargs["mode"] == "extended_thinking"


def test_dispatch_retries_until_success():
    with (
        patch(
            "backend.copilot.executor.manager.schedule_turn",
            new_callable=AsyncMock,
            side_effect=[RuntimeError("rmq down"), RuntimeError("rmq down"), None],
        ) as mock_schedule,
        patch("backend.copilot.executor.manager.time.sleep") as mock_sleep,
    ):
        _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    assert mock_schedule.await_count == 3
    assert mock_sleep.call_count == 2


def test_dispatch_gives_up_after_bounded_attempts_without_raising():
    with (
        patch(
            "backend.copilot.executor.manager.schedule_turn",
            new_callable=AsyncMock,
            side_effect=RuntimeError("rmq down"),
        ) as mock_schedule,
        patch("backend.copilot.executor.manager.time.sleep"),
    ):
        _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    assert mock_schedule.await_count == _SWITCH_DISPATCH_ATTEMPTS
