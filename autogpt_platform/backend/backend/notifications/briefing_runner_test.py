"""Scheduling and queue hand-off tests for alerts and Briefings."""

from collections.abc import Iterator
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import BriefingFrequency

from backend.data.notifications import NotificationResult
from backend.notifications import briefing_runner

NOW = datetime(2026, 8, 3, 7, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def server() -> None:
    return None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup() -> Iterator[None]:
    yield


def _user(user_id: str = "user-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=user_id,
        briefingFrequency=BriefingFrequency.DAILY.value,
        timezone="UTC",
        lastBriefingAt=None,
        alertsEnabled=True,
    )


def _result(success: bool) -> NotificationResult:
    return NotificationResult(success=success)


@pytest.mark.asyncio
async def test_no_matured_alerts_do_no_work():
    with patch.object(
        briefing_runner.alerts,
        "matured_alert_user_ids",
        AsyncMock(return_value=[]),
    ), patch.object(briefing_runner, "_flush_user_alerts", AsyncMock()) as flush_user:
        await briefing_runner.flush_matured_alerts()

    flush_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_alert_pass_continues_after_one_user_fails():
    with patch.object(
        briefing_runner.alerts,
        "matured_alert_user_ids",
        AsyncMock(return_value=["user-1", "user-2"]),
    ), patch.object(
        briefing_runner,
        "_flush_user_alerts",
        AsyncMock(side_effect=[RuntimeError("boom"), None]),
    ) as flush_user:
        await briefing_runner.flush_matured_alerts()

    assert [call.args[0] for call in flush_user.await_args_list] == [
        "user-1",
        "user-2",
    ]


@pytest.mark.asyncio
async def test_no_briefing_candidates_do_no_work():
    with patch.object(
        briefing_runner, "_briefing_candidates", AsyncMock(return_value=[])
    ), patch.object(briefing_runner, "_send_user_briefing", AsyncMock()) as send_user:
        await briefing_runner.send_due_briefings()

    send_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_briefing_pass_continues_after_one_user_fails():
    users = [_user("user-1"), _user("user-2")]
    with patch.object(
        briefing_runner, "_briefing_candidates", AsyncMock(return_value=users)
    ), patch.object(
        briefing_runner,
        "_send_user_briefing",
        AsyncMock(side_effect=[RuntimeError("boom"), None]),
    ) as send_user:
        await briefing_runner.send_due_briefings()

    assert [call.args[0].id for call in send_user.await_args_list] == [
        "user-1",
        "user-2",
    ]
    assert send_user.await_args_list[0].args[1] == send_user.await_args_list[1].args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [True, False])
async def test_alerts_are_marked_sent_only_after_queue_success(queued: bool):
    user = _user()
    built = SimpleNamespace(data=object(), condition_ids=["condition-1"])
    event = object()
    client = SimpleNamespace(find_unique=AsyncMock(return_value=user))

    with patch.object(
        briefing_runner.User, "prisma", return_value=client
    ), patch.object(
        briefing_runner.alerts,
        "build_alert_email",
        AsyncMock(return_value=built),
    ), patch.object(
        briefing_runner.alerts, "alert_event", return_value=event
    ), patch.object(
        briefing_runner,
        "queue_notification_async",
        AsyncMock(return_value=_result(queued)),
    ), patch.object(
        briefing_runner.alerts, "mark_alert_sent", AsyncMock()
    ) as mark_sent:
        await briefing_runner._flush_user_alerts(user.id)

    if queued:
        mark_sent.assert_awaited_once_with(["condition-1"])
    else:
        mark_sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_user_or_empty_alert_build_queues_nothing():
    missing_client = SimpleNamespace(find_unique=AsyncMock(return_value=None))
    build_alert = AsyncMock(return_value=None)
    queue = AsyncMock()

    with patch.object(
        briefing_runner.User, "prisma", return_value=missing_client
    ), patch.object(
        briefing_runner.alerts, "build_alert_email", build_alert
    ), patch.object(
        briefing_runner, "queue_notification_async", queue
    ):
        await briefing_runner._flush_user_alerts("missing")

    build_alert.assert_not_awaited()
    queue.assert_not_awaited()

    user = _user()
    found_client = SimpleNamespace(find_unique=AsyncMock(return_value=user))
    with patch.object(
        briefing_runner.User, "prisma", return_value=found_client
    ), patch.object(
        briefing_runner.alerts, "build_alert_email", build_alert
    ), patch.object(
        briefing_runner, "queue_notification_async", queue
    ):
        await briefing_runner._flush_user_alerts(user.id)

    build_alert.assert_awaited_once_with(user.id, user.alertsEnabled)
    queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_briefing_not_due_is_not_built():
    user = _user()
    build = AsyncMock()
    queue = AsyncMock()

    with patch.object(
        briefing_runner, "is_briefing_due", return_value=False
    ), patch.object(briefing_runner.briefing, "build_briefing", build), patch.object(
        briefing_runner, "queue_notification_async", queue
    ):
        await briefing_runner._send_user_briefing(user, NOW)

    build.assert_not_awaited()
    queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_briefing_is_not_queued():
    user = _user()
    queue = AsyncMock()

    with patch.object(
        briefing_runner, "is_briefing_due", return_value=True
    ), patch.object(
        briefing_runner.briefing,
        "build_briefing",
        AsyncMock(return_value=None),
    ), patch.object(
        briefing_runner, "queue_notification_async", queue
    ):
        await briefing_runner._send_user_briefing(user, NOW)

    queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_briefing_queue_does_not_mark_or_advance():
    user = _user()
    built = SimpleNamespace(data=object(), attention_condition_ids=["condition-1"])
    event = object()
    client = SimpleNamespace(update=AsyncMock())

    with patch.object(
        briefing_runner, "is_briefing_due", return_value=True
    ), patch.object(
        briefing_runner.briefing,
        "build_briefing",
        AsyncMock(return_value=built),
    ), patch.object(
        briefing_runner.briefing, "briefing_event", return_value=event
    ), patch.object(
        briefing_runner,
        "queue_notification_async",
        AsyncMock(return_value=_result(False)),
    ), patch.object(
        briefing_runner.briefing, "mark_attention_reported", AsyncMock()
    ) as mark_reported, patch.object(
        briefing_runner.User, "prisma", return_value=client
    ):
        await briefing_runner._send_user_briefing(user, NOW)

    mark_reported.assert_not_awaited()
    client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_briefing_marks_conditions_and_advances_last_sent():
    user = _user()
    built = SimpleNamespace(
        data=object(), attention_condition_ids=["condition-1", "condition-2"]
    )
    event = object()
    client = SimpleNamespace(update=AsyncMock())

    with patch.object(
        briefing_runner, "is_briefing_due", return_value=True
    ), patch.object(
        briefing_runner.briefing,
        "build_briefing",
        AsyncMock(return_value=built),
    ), patch.object(
        briefing_runner.briefing, "briefing_event", return_value=event
    ) as briefing_event, patch.object(
        briefing_runner,
        "queue_notification_async",
        AsyncMock(return_value=_result(True)),
    ) as queue, patch.object(
        briefing_runner.briefing, "mark_attention_reported", AsyncMock()
    ) as mark_reported, patch.object(
        briefing_runner.User, "prisma", return_value=client
    ):
        await briefing_runner._send_user_briefing(user, NOW)

    briefing_event.assert_called_once_with(user.id, built.data)
    queue.assert_awaited_once_with(event)
    mark_reported.assert_awaited_once_with(["condition-1", "condition-2"])
    client.update.assert_awaited_once_with(
        where={"id": user.id}, data={"lastBriefingAt": NOW}
    )


@pytest.mark.asyncio
async def test_candidate_query_uses_timezones_in_their_local_briefing_hour():
    user = _user()
    client = SimpleNamespace(find_many=AsyncMock(return_value=[user]))
    zones = ["UTC", "not-set"]

    with patch.object(
        briefing_runner.User, "prisma", return_value=client
    ), patch.object(briefing_runner, "_briefing_hour_timezones", return_value=zones):
        candidates = await briefing_runner._briefing_candidates(NOW)

    assert candidates == [user]
    assert client.find_many.await_args.kwargs["where"] == {
        "briefingFrequency": {"not": BriefingFrequency.OFF},
        "timezone": {"in": zones},
    }


def test_timezone_selection_follows_each_users_local_clock():
    utc_morning = datetime(2026, 8, 3, 7, 30, tzinfo=timezone.utc)
    new_york_morning = datetime(2026, 8, 3, 11, 30, tzinfo=timezone.utc)
    zones = {"UTC", "America/New_York", "Asia/Tokyo"}

    with patch.object(briefing_runner, "available_timezones", return_value=zones):
        assert set(briefing_runner._briefing_hour_timezones(utc_morning)) == {
            "UTC",
            "not-set",
        }
        assert briefing_runner._briefing_hour_timezones(new_york_morning) == [
            "America/New_York"
        ]
