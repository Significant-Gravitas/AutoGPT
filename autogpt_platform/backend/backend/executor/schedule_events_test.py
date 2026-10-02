from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from threading import Event
from unittest.mock import Mock, call

import pytest

from backend.data.activity_event import SCHEDULE_FIRE_EVENT_TYPES, ActivityEvent
from backend.executor import schedule_events


def test_rejected_activity_submission_preserves_schedule_and_product_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        schedule_events, "_submit", Mock(side_effect=RuntimeError("executor stopped"))
    )
    tracker = Mock()
    monkeypatch.setattr(
        schedule_events.product_analytics, "track_schedule_created", tracker
    )

    schedule_events.record_schedule_created(
        schedule_events.ScheduleCreatedRecord(
            user_id="user-1",
            schedule_id="schedule-1",
            title="Daily digest",
            target="agent",
        )
    )

    tracker.assert_called_once()
    assert tracker.call_args.kwargs["schedule_id"] == "schedule-1"


def test_concurrent_schedules_share_one_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = Event()
    release_first = Event()
    pool = Mock()
    extra_pool = Mock()
    first_work = Mock()
    second_work = Mock()
    pool.submit.side_effect = lambda work: work()

    def create_pool(**kwargs: object) -> Mock:
        if first_started.is_set():
            return extra_pool
        first_started.set()
        assert release_first.wait(timeout=5)
        return pool

    factory = Mock(side_effect=create_pool)
    monkeypatch.setattr(schedule_events, "_executor", None)
    monkeypatch.setattr(schedule_events, "ThreadPoolExecutor", factory)

    with ThreadPoolExecutor(max_workers=2) as callers:
        first = callers.submit(schedule_events._submit, first_work)
        assert first_started.wait(timeout=5)
        second = callers.submit(schedule_events._submit, second_work)
        try:
            second.result(timeout=0.2)
        except TimeoutError:
            pass
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)

    factory.assert_called_once_with(
        max_workers=2, thread_name_prefix="schedule-created-activity-event"
    )
    pool.submit.assert_has_calls([call(first_work), call(second_work)], any_order=True)
    first_work.assert_called_once_with()
    second_work.assert_called_once_with()


# ---------------------------------------------------------------------------
# Fire outcomes (SECRT-2787)
# ---------------------------------------------------------------------------


def _fired(**overrides) -> schedule_events.ScheduleFiredRecord:
    base = dict(
        user_id="user-1",
        schedule_id="schedule-1",
        status="dropped",
        reason="the account was at its concurrent-turn limit",
        fired_at=datetime(2026, 9, 30, 6, 12, 25, tzinfo=timezone.utc),
        message="  check   CI on PR #999  ",
        session_id="session-1",
        expert_id="expert-1",
        organization_id="org-1",
        scheduled_for=datetime(2026, 9, 30, 6, 12, tzinfo=timezone.utc),
        retry_schedule_id=None,
    )
    base.update(overrides)
    return schedule_events.ScheduleFiredRecord(**base)


def test_record_schedule_fired_writes_a_schedule_status_activity_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(schedule_events, "_submit", lambda work: work())
    client = Mock()
    monkeypatch.setattr(schedule_events, "get_database_manager_client", lambda: client)

    schedule_events.record_schedule_fired(_fired())

    client.create_activity_event.assert_called_once()
    kwargs = client.create_activity_event.call_args.kwargs
    assert kwargs["user_id"] == "user-1"
    draft = kwargs["draft"]
    assert draft.category == "SCHEDULE"
    assert draft.event_type == "schedule.dropped"
    assert draft.schedule_id == "schedule-1"
    assert draft.session_id == "session-1"
    assert draft.expert_id == "expert-1"
    assert draft.organization_id == "org-1"
    assert draft.title == (
        'Follow-up "check CI on PR #999" dropped: '
        "the account was at its concurrent-turn limit"
    )
    assert draft.data["status"] == "dropped"
    assert draft.data["scheduled_for"] == "2026-09-30T06:12:00+00:00"
    assert draft.data["fired_at"] == "2026-09-30T06:12:25+00:00"
    assert draft.data["message_preview"] == "check CI on PR #999"
    assert draft.data["is_recurring"] is False


def test_record_schedule_fired_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        schedule_events, "_submit", Mock(side_effect=RuntimeError("executor stopped"))
    )
    schedule_events.record_schedule_fired(_fired())


def test_fire_event_types_cover_every_status() -> None:
    assert set(SCHEDULE_FIRE_EVENT_TYPES) == {
        "schedule.dispatched",
        "schedule.skipped",
        "schedule.dropped",
        "schedule.failed",
    }
    assert set(schedule_events.UNDELIVERED_OUTCOME_STATUSES) == {
        "skipped",
        "dropped",
        "failed",
    }


def _event(event_type: str, **data_overrides) -> ActivityEvent:
    data = {
        "status": event_type.removeprefix("schedule."),
        "reason": "the account was at its concurrent-turn limit",
        "scheduled_for": "2026-09-30T06:12:00+00:00",
        "message_preview": "check CI on PR #999",
        "retry_schedule_id": "schedule-1-cap-retry",
        "cron": None,
    }
    data.update(data_overrides)
    return ActivityEvent(
        id="event-1",
        user_id="user-1",
        created_at=datetime(2026, 9, 30, 6, 12, 25, tzinfo=timezone.utc),
        category="SCHEDULE",
        event_type=event_type,
        title="x",
        schedule_id="schedule-1",
        session_id="session-1",
        expert_id="expert-1",
        data=data,
    )


def test_fire_outcome_round_trips_from_its_event() -> None:
    outcome = schedule_events.fire_outcome_from_event(_event("schedule.skipped"))
    assert outcome is not None
    assert outcome.status == "skipped"
    assert outcome.schedule_id == "schedule-1"
    assert outcome.session_id == "session-1"
    assert outcome.expert_id == "expert-1"
    assert outcome.reason == "the account was at its concurrent-turn limit"
    assert outcome.scheduled_for == "2026-09-30T06:12:00+00:00"
    assert outcome.message_preview == "check CI on PR #999"
    assert outcome.retry_schedule_id == "schedule-1-cap-retry"
    assert outcome.fired_at == datetime(2026, 9, 30, 6, 12, 25, tzinfo=timezone.utc)


def test_fire_outcome_tolerates_a_sparse_data_bag() -> None:
    event = _event("schedule.failed")
    event.data = {"reason": 42}
    outcome = schedule_events.fire_outcome_from_event(event)
    assert outcome is not None
    assert outcome.reason == ""
    assert outcome.message_preview == ""
    assert outcome.retry_schedule_id is None


def test_fire_outcome_ignores_other_schedule_events() -> None:
    assert schedule_events.fire_outcome_from_event(_event("schedule.created")) is None


def test_message_preview_truncates_long_prompts() -> None:
    preview = schedule_events.message_preview("word " * 100)
    assert len(preview) == 120
    assert preview.endswith("…")
