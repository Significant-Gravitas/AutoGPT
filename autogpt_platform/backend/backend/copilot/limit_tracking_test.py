"""``chat_blocked``: what it carries, and that sending it never gets in the way."""

from datetime import UTC, datetime, timedelta

import pytest

from backend.copilot import limit_tracking
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.limit_tracking import (
    LimitReading,
    executor_turn_surface,
    track_chat_blocked,
    track_limit_exceeded,
    track_turn_cap,
    track_turn_pause,
    track_turn_refusal,
    turn_surface,
)
from backend.copilot.llm_attribution import set_turn_origin
from backend.copilot.rate_limit import RateLimitExceeded
from backend.copilot.turn_pause import TurnPause
from backend.copilot.turn_queue import InflightCapExceeded, TurnRefusal

_RESETS = datetime(2026, 11, 10, 0, 0, tzinfo=UTC) + timedelta(days=1)

# Everything an event may carry besides the base properties ``capture`` adds:
# which limit, what happened, where, and the account against the limit.
_ALLOWED_KEYS = {
    "distinct_id",
    "reason",
    "limit",
    "outcome",
    "surface",
    "tier",
    "used",
    "cap",
    "resets_at",
}


def test_an_event_carries_the_limit_and_nothing_personal(chat_blocked_events):
    track_chat_blocked(
        "user-1",
        limit="daily",
        outcome="refused",
        surface="chat",
        reading=LimitReading(tier="PRO", used=3.5, cap=3.0),
        resets_at=_RESETS,
    )
    assert chat_blocked_events == [
        {
            "distinct_id": "user-1",
            "reason": "usage_limit",
            "limit": "daily",
            "outcome": "refused",
            "surface": "chat",
            "tier": "PRO",
            "used": 3.5,
            "cap": 3.0,
            "resets_at": _RESETS.isoformat(),
        }
    ]
    assert set(chat_blocked_events[0]) == _ALLOWED_KEYS


@pytest.mark.parametrize(
    "limit, reason",
    [
        ("daily", "usage_limit"),
        ("weekly", "usage_limit"),
        ("trial", "usage_limit"),
        ("task", "usage_limit"),
        ("rate", "rate_limit"),
        ("concurrency", "rate_limit"),
    ],
)
def test_the_reason_follows_the_analytics_plan(chat_blocked_events, limit, reason):
    track_chat_blocked("user-1", limit=limit, outcome="refused", surface="api")
    assert chat_blocked_events[0]["reason"] == reason


def test_sending_never_raises(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("posthog down")

    monkeypatch.setattr(limit_tracking.posthog_client, "capture", broken)
    track_chat_blocked("user-1", limit="rate", outcome="refused", surface="chat")


@pytest.mark.parametrize(
    "exc, resets_at",
    [
        (
            RateLimitExceeded(
                "weekly",
                _RESETS,
                reading=LimitReading(tier="PRO", used=20.0, cap=20.0),
            ),
            _RESETS.isoformat(),
        ),
        # A trial's allowance doesn't refill: its resets_at is the trial's end.
        (
            RateLimitExceeded(
                "trial", _RESETS, reading=LimitReading(tier="TRIAL", used=5, cap=5)
            ),
            None,
        ),
        # Nor does a window that allows nothing.
        (
            RateLimitExceeded("weekly", _RESETS, reading=LimitReading(cap=0.0)),
            None,
        ),
    ],
    ids=["weekly", "trial", "no-allowance"],
)
def test_a_cost_limit_sends_its_window_and_refill(chat_blocked_events, exc, resets_at):
    track_limit_exceeded("user-1", exc, surface="voice")
    (event,) = chat_blocked_events
    assert event["limit"] == exc.window
    assert event["surface"] == "voice"
    assert event["outcome"] == "refused"
    assert (event["tier"], event["used"], event["cap"]) == (
        exc.reading.tier,
        exc.reading.used,
        exc.reading.cap,
    )
    assert event["resets_at"] == resets_at


@pytest.mark.parametrize(
    "pause, expected",
    [
        (
            TurnPause(reason="daily", resets_at=_RESETS, checkpoint_saved=True),
            ("daily", "paused"),
        ),
        (
            TurnPause(reason="task", checkpoint_saved=False),
            ("task", "paused"),
        ),
        (
            TurnPause(
                reason="trial",
                checkpoint_saved=False,
                started=False,
                reading=LimitReading(tier="TRIAL", cap=0.0),
            ),
            ("trial", "refused"),
        ),
    ],
    ids=["paused-mid-turn", "task-budget", "not-started"],
)
def test_a_pause_on_a_limit_is_sent(chat_blocked_events, pause, expected):
    track_turn_pause("user-1", pause, surface="schedule")
    (event,) = chat_blocked_events
    assert (event["limit"], event["outcome"]) == expected
    assert event["surface"] == "schedule"
    assert event["tier"] == (pause.reading.tier if pause.reading else None)


def test_a_step_check_in_is_not_a_limit(chat_blocked_events):
    track_turn_pause("user-1", TurnPause(reason="steps", steps=200), surface="chat")
    assert chat_blocked_events == []


def test_the_reading_stays_off_the_paused_card():
    pause = TurnPause(reason="daily", reading=LimitReading(tier="PRO", cap=1.0))
    assert "reading" not in pause.as_part()


@pytest.mark.parametrize(
    "refusal, sent",
    [
        (
            TurnRefusal(
                reason="rate-limited",
                kind="usage",
                window="daily",
                resets_at=_RESETS,
                reading=LimitReading(tier="PRO", used=1.0, cap=1.0),
            ),
            True,
        ),
        (TurnRefusal(reason="paywalled", kind="no_plan"), False),
        (TurnRefusal(reason="is banned", kind="banned"), False),
        (TurnRefusal(reason="down", kind="outage", transient=True), False),
    ],
    ids=["usage", "no-plan", "banned", "outage"],
)
def test_only_a_usage_refusal_is_a_limit(chat_blocked_events, refusal, sent):
    track_turn_refusal("user-1", refusal, surface="schedule", outcome="refused")
    assert bool(chat_blocked_events) is sent
    if sent:
        assert chat_blocked_events[0]["resets_at"] == _RESETS.isoformat()


@pytest.mark.parametrize(
    "exc",
    [
        ConcurrentTurnLimitError("full", cap=5),
        InflightCapExceeded(used=15, cap=15),
    ],
    ids=["running-cap", "inflight-cap"],
)
def test_a_turn_cap_sends_its_count(chat_blocked_events, exc):
    track_turn_cap("user-1", exc, surface="block", outcome="queued")
    (event,) = chat_blocked_events
    assert event["limit"] == "concurrency"
    assert event["outcome"] == "queued"
    assert (event["used"], event["cap"]) == (exc.used, exc.cap)


@pytest.mark.parametrize(
    "origin, scheduled, platform, surface",
    [
        ("chat", False, None, "chat"),
        ("chat", False, "slack", "chat_bot"),
        ("scheduled", True, None, "schedule"),
        ("scheduled", False, None, "background"),
        ("automation", False, None, "block"),
        ("spawned", False, None, "background"),
        ("system", False, None, "background"),
    ],
)
def test_an_executor_turns_surface(origin, scheduled, platform, surface):
    assert (
        executor_turn_surface(origin, scheduled=scheduled, source_platform=platform)
        == surface
    )


def test_the_running_turns_surface_comes_from_its_origin():
    try:
        set_turn_origin(None)
        assert turn_surface(default="block") == "block"
        set_turn_origin("automation")
        assert turn_surface() == "block"
        set_turn_origin("scheduled")
        assert turn_surface() == "schedule"
    finally:
        set_turn_origin(None)
