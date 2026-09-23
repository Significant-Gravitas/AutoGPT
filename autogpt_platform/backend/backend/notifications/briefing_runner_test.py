"""Scheduling and queue hand-off tests for alerts and Briefings."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import BriefingFrequency

from backend.data.alerts import MaturedAlertPage
from backend.data.notifications import NotificationResult, PassWorkEvent, PassWorkKind
from backend.notifications import briefing_runner
from backend.notifications.conftest import make_db_client

NOW = datetime(2026, 8, 3, 7, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def server() -> None:
    return None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup() -> Iterator[None]:
    yield


def _candidates(users: list) -> object:
    """Stand-in for the paged async generator `_briefing_candidates` returns."""

    def _generator(*_args, **_kwargs):
        async def _iter():
            for user in users:
                yield user

        return _iter()

    return _generator


def _user(user_id: str = "user-1") -> SimpleNamespace:
    """Shaped like `BriefingCandidate`, which is what the RPC returns."""
    return SimpleNamespace(
        id=user_id,
        email="sam@example.com",
        briefing_frequency=BriefingFrequency.DAILY,
        timezone="UTC",
        last_briefing_at=None,
        alertsEnabled=True,
    )


def _result(success: bool) -> NotificationResult:
    return NotificationResult(success=success)


# ── the passes: publish, never assemble ─────────────────────────────────


@pytest.mark.asyncio
async def test_no_matured_alerts_publishes_nothing():
    with patch.object(
        briefing_runner.alerts,
        "matured_alert_user_ids",
        AsyncMock(return_value=MaturedAlertPage(user_ids=[], exhausted=True)),
    ), patch.object(briefing_runner, "queue_pass_work", AsyncMock()) as publish:
        await briefing_runner.flush_matured_alerts()

    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_alert_pass_publishes_one_message_per_user():
    """The tick must stay O(1) in work: it fans out and returns, so a slow
    user cannot hold the next tick open."""
    with patch.object(
        briefing_runner.alerts,
        "matured_alert_user_ids",
        AsyncMock(
            return_value=MaturedAlertPage(user_ids=["user-1", "user-2"], exhausted=True)
        ),
    ), patch.object(briefing_runner, "queue_pass_work", AsyncMock()) as publish:
        await briefing_runner.flush_matured_alerts()

    assert [c.args[1] for c in publish.await_args_list] == ["user-1", "user-2"]
    for call in publish.await_args_list:
        event = PassWorkEvent.model_validate_json(call.args[2])
        assert event.kind is PassWorkKind.ALERT_FLUSH


@pytest.mark.asyncio
async def test_the_alert_walk_keeps_going_past_a_deduplicated_page():
    """A full page of rows collapses to a handful of users once duplicates are
    dropped, and anyone holding two conditions produces exactly that. Treating
    the short user list as the end of the table would strand every user after
    the first page."""
    pages = [
        MaturedAlertPage(user_ids=["user-1", "user-2"], exhausted=False),
        MaturedAlertPage(user_ids=["user-3"], exhausted=True),
    ]
    with patch.object(
        briefing_runner.alerts,
        "matured_alert_user_ids",
        AsyncMock(side_effect=pages),
    ), patch.object(briefing_runner, "queue_pass_work", AsyncMock()) as publish:
        await briefing_runner.flush_matured_alerts()

    assert [c.args[1] for c in publish.await_args_list] == [
        "user-1",
        "user-2",
        "user-3",
    ]


@pytest.mark.asyncio
async def test_no_briefing_candidates_publishes_nothing():
    with patch.object(
        briefing_runner, "_briefing_candidates", _candidates([])
    ), patch.object(briefing_runner, "queue_pass_work", AsyncMock()) as publish:
        await briefing_runner.send_due_briefings()

    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_briefing_pass_publishes_only_users_that_are_due():
    users = [_user("user-1"), _user("user-2")]
    with patch.object(
        briefing_runner, "_briefing_candidates", _candidates(users)
    ), patch.object(
        briefing_runner, "is_briefing_due", side_effect=[True, False]
    ), patch.object(
        briefing_runner, "queue_pass_work", AsyncMock()
    ) as publish:
        await briefing_runner.send_due_briefings()

    assert [c.args[1] for c in publish.await_args_list] == ["user-1"]


@pytest.mark.asyncio
async def test_every_message_carries_the_passes_own_clock():
    """The consumer must not read the clock itself: a message that waits in the
    queue still belongs to the period it was scheduled for, and the dedupe key
    is built from that timestamp."""
    users = [_user("user-1"), _user("user-2")]
    with patch.object(
        briefing_runner, "_briefing_candidates", _candidates(users)
    ), patch.object(
        briefing_runner, "is_briefing_due", return_value=True
    ), patch.object(
        briefing_runner, "queue_pass_work", AsyncMock()
    ) as publish:
        await briefing_runner.send_due_briefings()

    stamps = {
        PassWorkEvent.model_validate_json(c.args[2]).scheduled_for
        for c in publish.await_args_list
    }
    assert len(stamps) == 1


# ── the work: claimed exactly once ──────────────────────────────────────


@pytest.mark.asyncio
async def test_work_is_claimed_before_anything_is_sent():
    """Queue delivery is at-least-once and there may be several replicas, so
    an unclaimed redelivery would be a second email."""
    event = PassWorkEvent(
        kind=PassWorkKind.ALERT_FLUSH, user_id="user-1", scheduled_for=NOW
    )
    with patch.object(
        briefing_runner, "claim_once", AsyncMock(return_value=False)
    ) as claim, patch.object(
        briefing_runner, "_flush_user_alerts", AsyncMock()
    ) as flush:
        await briefing_runner.run_pass_work(event)

    claim.assert_awaited_once()
    flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_claim_key_is_scoped_to_the_period():
    """A month-long key would suppress the *next* period's legitimate send."""
    first = PassWorkEvent(
        kind=PassWorkKind.BRIEFING, user_id="user-1", scheduled_for=NOW
    )
    later = PassWorkEvent(
        kind=PassWorkKind.BRIEFING,
        user_id="user-1",
        scheduled_for=NOW + timedelta(days=1),
    )
    seen = []

    async def record(key, **_kw):
        seen.append(key)
        return True

    with patch.object(
        briefing_runner, "claim_once", AsyncMock(side_effect=record)
    ), patch.object(briefing_runner, "_build_and_queue_briefing", AsyncMock()):
        await briefing_runner.run_pass_work(first)
        await briefing_runner.run_pass_work(later)

    assert len(set(seen)) == 2, "each period must claim its own key"
    assert all("user-1" in k for k in seen)


@pytest.mark.asyncio
async def test_two_publishers_of_one_slot_share_a_claim_key():
    """The claim exists to stop a second replica sending a second email, so it
    has to key on the slot rather than the instant the message was published."""
    seen = []

    async def record(key, **_kw):
        seen.append(key)
        return True

    same_slot = [
        PassWorkEvent(
            kind=PassWorkKind.BRIEFING,
            user_id="user-1",
            scheduled_for=NOW.replace(minute=m, second=s, microsecond=us),
        )
        for m, s, us in ((0, 0, 0), (17, 42, 987654))
    ]
    with patch.object(
        briefing_runner, "claim_once", AsyncMock(side_effect=record)
    ), patch.object(briefing_runner, "_build_and_queue_briefing", AsyncMock()):
        for event in same_slot:
            await briefing_runner.run_pass_work(event)

    assert len(set(seen)) == 1, f"one slot must be one key, got {seen}"


@pytest.mark.asyncio
async def test_welcome_is_claimed_by_checkout_session():
    """A Stripe redelivery arrives at a new timestamp, so a timestamp key would
    never suppress it — and every welcome carries an empty user_id, so the key
    has to come from the session."""
    seen = []

    async def record(key, **_kw):
        seen.append(key)
        return True

    with patch.object(
        briefing_runner, "claim_once", AsyncMock(side_effect=record)
    ), patch.object(briefing_runner.lifecycle, "send_welcome_for_session", AsyncMock()):
        for offset in (0, 90):
            await briefing_runner.run_pass_work(
                PassWorkEvent(
                    kind=PassWorkKind.WELCOME,
                    user_id="",
                    scheduled_for=NOW + timedelta(seconds=offset),
                    context={"session_id": "cs_test_123"},
                )
            )

    assert len(set(seen)) == 1, f"one session must be one key, got {seen}"
    assert "cs_test_123" in seen[0]


@pytest.mark.asyncio
async def test_a_failed_pass_releases_its_claim():
    """Otherwise the consumer's retry finds the claim it made itself, returns
    early, and the user's briefing is dropped for the whole period."""
    event = PassWorkEvent(
        kind=PassWorkKind.BRIEFING, user_id="user-1", scheduled_for=NOW
    )
    with patch.object(
        briefing_runner, "claim_once", AsyncMock(return_value=True)
    ), patch.object(
        briefing_runner, "release_claim", AsyncMock()
    ) as release, patch.object(
        briefing_runner,
        "_build_and_queue_briefing",
        AsyncMock(side_effect=RuntimeError("postmark exploded")),
    ):
        with pytest.raises(RuntimeError):
            await briefing_runner.run_pass_work(event)

    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_each_kind_routes_to_its_own_work():
    with patch.object(
        briefing_runner, "claim_once", AsyncMock(return_value=True)
    ), patch.object(
        briefing_runner, "_flush_user_alerts", AsyncMock()
    ) as flush, patch.object(
        briefing_runner, "_build_and_queue_briefing", AsyncMock()
    ) as build:
        await briefing_runner.run_pass_work(
            PassWorkEvent(kind=PassWorkKind.ALERT_FLUSH, user_id="u", scheduled_for=NOW)
        )
        await briefing_runner.run_pass_work(
            PassWorkEvent(kind=PassWorkKind.BRIEFING, user_id="u", scheduled_for=NOW)
        )

    flush.assert_awaited_once()
    build.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [True, False])
async def test_alerts_are_marked_sent_only_after_queue_success(queued: bool):
    user = _user()
    built = SimpleNamespace(data=object(), condition_ids=["condition-1"])
    event = object()
    client = make_db_client(
        get_user_notification_preference=AsyncMock(
            return_value=SimpleNamespace(alerts_enabled=user.alertsEnabled)
        )
    )

    with patch.object(briefing_runner, "_db", return_value=client), patch.object(
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
        if queued:
            await briefing_runner._flush_user_alerts(user.id)
        else:
            # A failed publish must reach the consumer so its retry can run;
            # swallowing it would drop the alert silently.
            with pytest.raises(RuntimeError):
                await briefing_runner._flush_user_alerts(user.id)

    if queued:
        mark_sent.assert_awaited_once_with(["condition-1"])
    else:
        mark_sent.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_empty_alert_build_queues_nothing():
    """Everything was deferred or resolved, so there is nothing to say."""
    user = _user()
    build_alert = AsyncMock(return_value=None)
    queue = AsyncMock()
    client = make_db_client(
        get_user_notification_preference=AsyncMock(
            return_value=SimpleNamespace(alerts_enabled=True)
        )
    )

    with patch.object(briefing_runner, "_db", return_value=client), patch.object(
        briefing_runner.alerts, "build_alert_email", build_alert
    ), patch.object(briefing_runner, "queue_notification_async", queue):
        await briefing_runner._flush_user_alerts(user.id)

    build_alert.assert_awaited_once_with(user.id, True)
    queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_database_failure_propagates_so_the_consumer_retries():
    """`get_user_notification_preference` raises for an unknown user rather
    than returning None. Swallowing that would drop the alert silently."""
    client = make_db_client(
        get_user_notification_preference=AsyncMock(side_effect=RuntimeError("db down"))
    )
    with patch.object(briefing_runner, "_db", return_value=client):
        with pytest.raises(RuntimeError):
            await briefing_runner._flush_user_alerts("user-1")


@pytest.mark.asyncio
async def test_empty_briefing_is_not_queued():
    user = _user()
    queue = AsyncMock()
    client = make_db_client(get_briefing_candidate=AsyncMock(return_value=user))

    with patch.object(briefing_runner, "_db", return_value=client), patch.object(
        briefing_runner.briefing,
        "build_briefing",
        AsyncMock(return_value=None),
    ), patch.object(briefing_runner, "queue_notification_async", queue):
        await briefing_runner._build_and_queue_briefing(user.id, NOW)

    queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_briefing_queue_does_not_mark_or_advance():
    user = _user()
    built = SimpleNamespace(data=object(), attention_condition_ids=["condition-1"])
    event = object()
    client = make_db_client(get_briefing_candidate=AsyncMock(return_value=user))

    with patch.object(
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
        briefing_runner, "_db", return_value=client
    ):
        # A failed publish raises so the consumer retries rather than
        # advancing the cadence clock on an email that never went out.
        with pytest.raises(RuntimeError):
            await briefing_runner._build_and_queue_briefing(user.id, NOW)

    mark_reported.assert_not_awaited()
    client.set_last_briefing_at.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_briefing_marks_conditions_and_advances_last_sent():
    user = _user()
    built = SimpleNamespace(
        data=object(), attention_condition_ids=["condition-1", "condition-2"]
    )
    event = object()
    client = make_db_client(get_briefing_candidate=AsyncMock(return_value=user))

    with patch.object(
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
        briefing_runner, "_db", return_value=client
    ):
        await briefing_runner._build_and_queue_briefing(user.id, NOW)

    briefing_event.assert_called_once_with(user.id, built.data)
    queue.assert_awaited_once_with(event)
    mark_reported.assert_awaited_once_with(["condition-1", "condition-2"])
    client.set_last_briefing_at.assert_awaited_once_with(user.id, NOW)


@pytest.mark.asyncio
async def test_candidate_query_uses_timezones_in_their_local_briefing_hour():
    user = _user()
    client = make_db_client(get_briefing_candidates=AsyncMock(return_value=[user]))
    zones = ["UTC", "not-set"]

    with patch.object(briefing_runner, "_db", return_value=client):
        candidates = [u async for u in briefing_runner._briefing_candidates(zones)]

    assert candidates == [user]
    # The timezone filter lives in the DatabaseManager, not here; this asserts
    # the runner hands it the zones and pages from the start.
    client.get_briefing_candidates.assert_awaited_once_with(
        zones, None, briefing_runner.CANDIDATE_PAGE_SIZE
    )


@pytest.mark.asyncio
async def test_every_candidate_is_considered_not_just_the_first_page():
    """A capped read strands everyone past the cap behind the same low-id page.

    The ordering is stable and the query does not exclude users already
    briefed, so a bounded single read returns the *same* users every hour and
    the tail of the table never receives a briefing at all.
    """
    page_size = briefing_runner.CANDIDATE_PAGE_SIZE
    all_users = [_user(f"user-{i:05d}") for i in range(page_size * 2 + 7)]

    async def paged(_zones, after, limit):
        remaining = [u for u in all_users if after is None or u.id > after]
        return remaining[:limit]

    client = make_db_client(get_briefing_candidates=AsyncMock(side_effect=paged))
    with patch.object(briefing_runner, "_db", return_value=client):
        seen = [u async for u in briefing_runner._briefing_candidates(["UTC"])]

    assert len(seen) == len(all_users)
    assert [u.id for u in seen] == [u.id for u in all_users]


@pytest.mark.asyncio
async def test_candidate_walk_stops_once_a_short_page_comes_back():
    """The walk must terminate rather than spin on an exhausted table."""
    client = make_db_client(get_briefing_candidates=AsyncMock(return_value=[_user()]))
    with patch.object(briefing_runner, "_db", return_value=client):
        seen = [u async for u in briefing_runner._briefing_candidates(["UTC"])]

    assert len(seen) == 1
    assert client.get_briefing_candidates.await_count == 1


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
