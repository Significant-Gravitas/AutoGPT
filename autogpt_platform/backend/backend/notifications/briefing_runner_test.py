"""Scheduling and queue hand-off tests for alerts and Briefings."""

from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import BriefingFrequency

from backend.data.alerts import MaturedAlertPage
from backend.data.notifications import NotificationScope, PassWorkEvent, PassWorkKind
from backend.notifications import briefing_runner
from backend.notifications.conftest import make_db_client

NOW = datetime(2026, 8, 3, 7, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def server() -> None:
    return None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup() -> Iterator[None]:
    yield


@pytest.fixture(autouse=True)
def condition_delivery_lease():
    class Guard:
        async def run(self, action):
            return await action

    @asynccontextmanager
    async def lease(_condition_ids):
        yield Guard()

    with patch.object(briefing_runner, "alert_condition_delivery_lease", lease):
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
        await briefing_runner.run_pass_work(event, AsyncMock(return_value=True))

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
    ), patch.object(briefing_runner, "_build_and_send_briefing", AsyncMock()):
        deliver = AsyncMock(return_value=True)
        await briefing_runner.run_pass_work(first, deliver)
        await briefing_runner.run_pass_work(later, deliver)

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
    ), patch.object(briefing_runner, "_build_and_send_briefing", AsyncMock()):
        deliver = AsyncMock(return_value=True)
        for event in same_slot:
            await briefing_runner.run_pass_work(event, deliver)

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
                ),
                AsyncMock(return_value=True),
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
        "_build_and_send_briefing",
        AsyncMock(side_effect=RuntimeError("postmark exploded")),
    ):
        with pytest.raises(RuntimeError):
            await briefing_runner.run_pass_work(event, AsyncMock(return_value=True))

    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_each_kind_routes_to_its_own_work():
    with patch.object(
        briefing_runner, "claim_once", AsyncMock(return_value=True)
    ), patch.object(
        briefing_runner, "_flush_user_alerts", AsyncMock()
    ) as flush, patch.object(
        briefing_runner, "_build_and_send_briefing", AsyncMock()
    ) as build:
        deliver = AsyncMock(return_value=True)
        await briefing_runner.run_pass_work(
            PassWorkEvent(
                kind=PassWorkKind.ALERT_FLUSH, user_id="u", scheduled_for=NOW
            ),
            deliver,
        )
        await briefing_runner.run_pass_work(
            PassWorkEvent(kind=PassWorkKind.BRIEFING, user_id="u", scheduled_for=NOW),
            deliver,
        )

    flush.assert_awaited_once()
    build.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("delivered", [True, False])
async def test_alerts_finalize_only_after_delivery(delivered: bool):
    user = _user()
    scope = NotificationScope()
    built = SimpleNamespace(
        data=object(),
        condition_ids=["condition-1"],
        authorization_scopes=[scope],
    )
    event = object()
    client = make_db_client(
        get_user_notification_preference=AsyncMock(
            return_value=SimpleNamespace(alerts_enabled=user.alertsEnabled)
        ),
        get_pending_alert_condition_scopes=AsyncMock(return_value=[scope]),
    )
    deliver = AsyncMock(return_value=delivered)

    with patch.object(briefing_runner, "_db", return_value=client), patch.object(
        briefing_runner.alerts,
        "build_alert_email",
        AsyncMock(return_value=built),
    ), patch.object(briefing_runner.alerts, "alert_event", return_value=event):
        await briefing_runner._flush_user_alerts(user.id, "delivery-1", deliver)

    if delivered:
        finalized = client.finalize_alert_delivery.await_args.args
        assert finalized[:3] == (user.id, ["condition-1"], [scope])
    else:
        client.finalize_alert_delivery.assert_not_awaited()


@pytest.mark.asyncio
async def test_alert_finalizes_before_condition_lease_is_released():
    order: list[str] = []
    scope = NotificationScope()
    built = SimpleNamespace(
        data=object(),
        condition_ids=["condition-1"],
        authorization_scopes=[scope],
    )

    class Guard:
        async def run(self, action):
            return await action

    @asynccontextmanager
    async def lease(_condition_ids):
        order.append("lease-enter")
        try:
            yield Guard()
        finally:
            order.append("lease-exit")

    async def finalize(*_args):
        order.append("finalize")

    async def deliver(_event):
        order.append("deliver")
        return True

    client = make_db_client(
        get_user_notification_preference=AsyncMock(
            return_value=SimpleNamespace(alerts_enabled=True)
        ),
        get_pending_alert_condition_scopes=AsyncMock(return_value=[scope]),
        finalize_alert_delivery=AsyncMock(side_effect=finalize),
    )
    with patch.object(briefing_runner, "_db", return_value=client), patch.object(
        briefing_runner.alerts,
        "build_alert_email",
        AsyncMock(return_value=built),
    ), patch.object(
        briefing_runner.alerts, "alert_event", return_value=object()
    ), patch.object(
        briefing_runner, "alert_condition_delivery_lease", lease
    ):
        await briefing_runner._flush_user_alerts("user-1", "delivery-1", deliver)

    assert order == ["lease-enter", "deliver", "finalize", "lease-exit"]


@pytest.mark.asyncio
async def test_stale_alert_source_is_dropped_without_delivery_or_mutation():
    user = _user()
    scope = NotificationScope()
    built = SimpleNamespace(
        data=object(),
        condition_ids=["condition-1"],
        authorization_scopes=[scope],
    )
    client = make_db_client(
        get_user_notification_preference=AsyncMock(
            return_value=SimpleNamespace(alerts_enabled=True)
        ),
        get_pending_alert_condition_scopes=AsyncMock(return_value=[scope]),
        get_stale_alert_condition_ids=AsyncMock(return_value=["condition-1"]),
    )
    deliver = AsyncMock(return_value=True)

    with patch.object(briefing_runner, "_db", return_value=client), patch.object(
        briefing_runner.alerts,
        "build_alert_email",
        AsyncMock(return_value=built),
    ):
        await briefing_runner._flush_user_alerts(user.id, "delivery-1", deliver)

    deliver.assert_not_awaited()
    client.resolve_alert_conditions_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_empty_alert_build_sends_nothing():
    """Everything was deferred or resolved, so there is nothing to say."""
    user = _user()
    build_alert = AsyncMock(return_value=None)
    deliver = AsyncMock(return_value=True)
    client = make_db_client(
        get_user_notification_preference=AsyncMock(
            return_value=SimpleNamespace(alerts_enabled=True)
        )
    )

    with patch.object(briefing_runner, "_db", return_value=client), patch.object(
        briefing_runner.alerts, "build_alert_email", build_alert
    ):
        await briefing_runner._flush_user_alerts(user.id, "delivery-1", deliver)

    build_alert.assert_awaited_once_with(user.id, True, [])
    deliver.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_database_failure_propagates_so_the_consumer_retries():
    """`get_user_notification_preference` raises for an unknown user rather
    than returning None. Swallowing that would drop the alert silently."""
    client = make_db_client(
        get_user_notification_preference=AsyncMock(side_effect=RuntimeError("db down"))
    )
    with patch.object(briefing_runner, "_db", return_value=client):
        with pytest.raises(RuntimeError):
            await briefing_runner._flush_user_alerts(
                "user-1", "delivery-1", AsyncMock(return_value=True)
            )


@pytest.mark.asyncio
async def test_empty_briefing_is_not_sent():
    user = _user()
    deliver = AsyncMock(return_value=True)
    client = make_db_client(get_briefing_candidate=AsyncMock(return_value=user))

    with patch.object(briefing_runner, "_db", return_value=client), patch.object(
        briefing_runner.briefing,
        "build_briefing",
        AsyncMock(return_value=None),
    ):
        await briefing_runner._build_and_send_briefing(
            user.id, NOW, "delivery-1", deliver
        )

    deliver.assert_not_awaited()


@pytest.mark.asyncio
async def test_skipped_briefing_delivery_does_not_finalize():
    user = _user()
    scope = NotificationScope()
    built = SimpleNamespace(
        data=object(),
        attention_condition_ids=["condition-1"],
        authorization_scopes=[scope],
    )
    event = object()
    client = make_db_client(
        get_briefing_candidate=AsyncMock(return_value=user),
        get_briefing_resource_scopes=AsyncMock(return_value=[scope]),
    )

    with patch.object(
        briefing_runner.briefing,
        "build_briefing",
        AsyncMock(return_value=built),
    ), patch.object(
        briefing_runner.briefing, "briefing_event", return_value=event
    ), patch.object(
        briefing_runner, "_db", return_value=client
    ):
        await briefing_runner._build_and_send_briefing(
            user.id, NOW, "delivery-1", AsyncMock(return_value=False)
        )

    client.finalize_briefing_delivery.assert_not_awaited()


@pytest.mark.asyncio
async def test_delivered_briefing_finalizes_conditions_and_cadence():
    user = _user()
    scope = NotificationScope()
    built = SimpleNamespace(
        data=object(),
        attention_condition_ids=["condition-1", "condition-2"],
        authorization_scopes=[scope],
    )
    event = object()
    client = make_db_client(
        get_briefing_candidate=AsyncMock(return_value=user),
        get_briefing_resource_scopes=AsyncMock(return_value=[scope]),
    )
    deliver = AsyncMock(return_value=True)

    with patch.object(
        briefing_runner.briefing,
        "build_briefing",
        AsyncMock(return_value=built),
    ), patch.object(
        briefing_runner.briefing, "briefing_event", return_value=event
    ) as briefing_event, patch.object(
        briefing_runner, "_db", return_value=client
    ):
        await briefing_runner._build_and_send_briefing(
            user.id, NOW, "delivery-1", deliver
        )

    briefing_event.assert_called_once_with(user.id, built.data, [scope], "delivery-1")
    deliver.assert_awaited_once_with(event)
    finalized = client.finalize_briefing_delivery.await_args.args
    assert finalized[:3] == (
        user.id,
        ["condition-1", "condition-2"],
        [scope],
    )
    assert finalized[4] == NOW


@pytest.mark.asyncio
async def test_briefing_finalizes_before_condition_lease_is_released():
    order: list[str] = []
    user = _user()
    scope = NotificationScope()
    built = SimpleNamespace(
        data=object(),
        attention_condition_ids=["condition-1"],
        authorization_scopes=[scope],
    )

    class Guard:
        async def run(self, action):
            return await action

    @asynccontextmanager
    async def lease(_condition_ids):
        order.append("lease-enter")
        try:
            yield Guard()
        finally:
            order.append("lease-exit")

    async def finalize(*_args):
        order.append("finalize")

    async def deliver(_event):
        order.append("deliver")
        return True

    client = make_db_client(
        get_briefing_candidate=AsyncMock(return_value=user),
        get_briefing_resource_scopes=AsyncMock(return_value=[scope]),
        finalize_briefing_delivery=AsyncMock(side_effect=finalize),
    )
    with patch.object(
        briefing_runner.briefing,
        "build_briefing",
        AsyncMock(return_value=built),
    ), patch.object(
        briefing_runner.briefing, "briefing_event", return_value=object()
    ), patch.object(
        briefing_runner, "_db", return_value=client
    ), patch.object(
        briefing_runner, "alert_condition_delivery_lease", lease
    ):
        await briefing_runner._build_and_send_briefing(
            user.id, NOW, "delivery-1", deliver
        )

    assert order == ["lease-enter", "deliver", "finalize", "lease-exit"]


@pytest.mark.asyncio
async def test_stale_briefing_attention_is_dropped_without_delivery_or_mutation():
    user = _user()
    scope = NotificationScope()
    built = SimpleNamespace(
        data=object(),
        attention_condition_ids=["condition-1"],
        authorization_scopes=[scope],
    )
    client = make_db_client(
        get_briefing_candidate=AsyncMock(return_value=user),
        get_briefing_resource_scopes=AsyncMock(return_value=[scope]),
        get_briefing_alert_condition_scopes=AsyncMock(return_value=[scope]),
        get_stale_alert_condition_ids=AsyncMock(return_value=["condition-1"]),
    )
    deliver = AsyncMock(return_value=True)

    with patch.object(
        briefing_runner.briefing,
        "build_briefing",
        AsyncMock(return_value=built),
    ), patch.object(briefing_runner, "_db", return_value=client):
        await briefing_runner._build_and_send_briefing(
            user.id, NOW, "delivery-1", deliver
        )

    deliver.assert_not_awaited()
    client.resolve_alert_conditions_by_ids.assert_not_awaited()


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
