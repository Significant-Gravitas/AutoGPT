from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event
from unittest.mock import Mock, call

import pytest

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
