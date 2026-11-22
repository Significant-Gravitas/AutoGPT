"""MailerLite gets the same subscription status and dates as PostHog, from the
same snapshot (SECRT-2795): written only into someone MailerLite already
holds, and only the fields that differ."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.notifications import AudienceAction
from backend.data.notifications import SubscriberField as F
from backend.data.notifications import SubscriptionStatus
from backend.data.posthog_lifecycle import LifecycleSnapshot, LifecycleStatus
from backend.data.posthog_lifecycle_test import (
    NOW,
    SIGNUP,
    TRIAL_START,
    coupon,
    snapshot,
    sub,
    trial,
    user,
)
from backend.notifications import lifecycle_audience, mailerlite, subscriber_fields
from backend.notifications.lifecycle_audience import (
    LifecycleContact,
    lifecycle_fields,
    mailerlite_lifecycle_enabled,
    queue_lifecycle_fields,
    resync_after_group_change,
)
from backend.notifications.notifications import NotificationManager

EMAIL = "sam@example.com"


def _day(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d")


def _contact(**overrides) -> LifecycleContact:
    values = {"id": "user-1", "email": EMAIL, "stripe_customer_id": "cus_1"}
    return LifecycleContact(**{**values, **overrides})


def test_every_lifecycle_status_is_a_mailerlite_status():
    """GTM segments on these values: one list, both sinks."""
    assert set(get_args(LifecycleStatus)) == {s.value for s in SubscriptionStatus}


def test_the_snapshot_becomes_the_status_and_its_days():
    started = NOW - timedelta(days=40)
    canceled = NOW - timedelta(days=3)
    result = lifecycle_fields(
        LifecycleSnapshot(
            subscription_status="subscription_canceled",
            signup_at=SIGNUP,
            trial_started_at=TRIAL_START,
            subscription_started_at=started,
            subscription_canceled_at=canceled,
        )
    )
    assert result == {
        F.STATUS: "subscription_canceled",
        F.SIGNUP: _day(SIGNUP),
        F.TRIAL_STARTED: _day(TRIAL_START),
        F.SUBSCRIPTION_STARTED: _day(started),
        F.SUBSCRIPTION_CANCELED: _day(canceled),
        # A date that doesn't apply is cleared, so none lingers.
        F.SUBSCRIPTION_ENDED: None,
    }


# ---- the four SECRT-2795 rules, as MailerLite receives them ------------------


def test_an_enterprise_user_without_a_paid_subscription_is_signed():
    # The mapping never reads the tier, so an admin grant is just a signup.
    fields = lifecycle_fields(snapshot(u=user(customer=None)))
    assert fields[F.STATUS] == "signed"
    assert fields[F.SUBSCRIPTION_STARTED] is None


def test_a_fully_discounted_subscription_is_not_subscribed():
    fields = lifecycle_fields(snapshot(subs=[sub(discounts=[coupon(percent_off=100)])]))
    assert fields[F.STATUS] == "signed"
    assert fields[F.SUBSCRIPTION_STARTED] is None


def test_a_trial_that_ran_out_has_expired():
    fields = lifecycle_fields(snapshot(t=trial(ends_at=NOW - timedelta(minutes=1))))
    assert fields[F.STATUS] == "trial_expired"
    assert fields[F.TRIAL_STARTED] == _day(TRIAL_START)


def test_the_subscription_start_is_the_first_paid_one():
    first = NOW - timedelta(days=200)
    old = sub("sub_old", "canceled", start=first, ended_at=NOW - timedelta(days=90))
    fields = lifecycle_fields(snapshot(subs=[old, sub("sub_new")]))
    assert fields[F.STATUS] == "subscribed"
    assert fields[F.SUBSCRIPTION_STARTED] == _day(first)


# ---- queueing ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_fields_are_queued_as_a_lifecycle_change():
    queued = AsyncMock(return_value=SimpleNamespace(success=True, message=""))
    result = snapshot(subs=[sub()])
    with patch("backend.notifications.subscriber_fields.queue_audience_change", queued):
        await queue_lifecycle_fields(_contact(), result)
    [event] = [call.args[0] for call in queued.await_args_list]
    assert event.action is AudienceAction.LIFECYCLE_FIELDS
    assert event.user_id == "user-1"
    assert event.fields == lifecycle_fields(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contact",
    [
        # Never reached checkout, so not a MailerLite subscriber.
        {"stripe_customer_id": None},
        {"marketing_opt_out_at": NOW},
        {"excluded_country": "IR"},
        {"timezone": "Europe/Moscow"},
    ],
)
async def test_nothing_is_queued_for_someone_mailerlite_must_not_hold(contact):
    queued = AsyncMock()
    with patch("backend.notifications.subscriber_fields.queue_audience_change", queued):
        await queue_lifecycle_fields(_contact(**contact), snapshot())
    queued.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_queue_failure_is_reported_not_raised():
    queued = AsyncMock(side_effect=RuntimeError("rabbit down"))
    with patch("backend.notifications.subscriber_fields.queue_audience_change", queued):
        await queue_lifecycle_fields(_contact(), snapshot())


# ---- the consumer -----------------------------------------------------------


@pytest.mark.parametrize(
    "stripe_key, group_id, expected",
    [
        # The API process: Stripe, but no MailerLite settings at all.
        ("sk_test", "", True),
        ("", "grp_checkout", False),
        ("", "", False),
    ],
)
def test_mailerlite_s_half_runs_wherever_stripe_does(
    monkeypatch, stripe_key, group_id, expected
):
    """Only the notification service holds the MailerLite settings, so the
    process that queues the fields can't ask for them; a stack without
    Stripe has no checkout openers to write."""
    settings = SimpleNamespace(
        secrets=SimpleNamespace(stripe_api_key=stripe_key),
        config=SimpleNamespace(mailerlite_checkout_group_id=group_id),
    )
    monkeypatch.setattr(lifecycle_audience, "settings", settings)
    assert mailerlite_lifecycle_enabled() is expected


@pytest.mark.asyncio
async def test_a_group_join_asks_for_one_more_sync(lifecycle_resync):
    await resync_after_group_change("user-1")
    lifecycle_resync.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_a_failed_resync_request_is_reported_not_raised(lifecycle_resync, caplog):
    lifecycle_resync.side_effect = RuntimeError("database manager down")
    await resync_after_group_change("user-1")
    assert "Could not request a lifecycle sync for user user-1" in caplog.text


@pytest.fixture
def mailerlite_configured(monkeypatch):
    fake = SimpleNamespace(
        config=SimpleNamespace(
            mailerlite_checkout_group_id="grp_checkout",
            mailerlite_trial_group_id="",
        ),
        secrets=SimpleNamespace(mailerlite_api_token="token"),
    )
    monkeypatch.setattr(mailerlite, "settings", fake)
    monkeypatch.setattr(mailerlite, "_fields_ready", True)


def _response(status: int, body: dict | None = None) -> MagicMock:
    response = MagicMock(status=status)
    response.json.return_value = body or {}
    return response


class _FakeMailerLite:
    def __init__(self, held: dict[str, dict] | None = None):
        self.subscribers = held or {}
        self.posts: list[dict] = []

    async def get(self, url: str, **_) -> MagicMock:
        held = self.subscribers.get(url.rsplit("/", 1)[-1])
        if held is None:
            return _response(404)
        return _response(200, {"data": {"id": "ml_1", "fields": dict(held)}})

    async def post(self, url: str, json: dict, **_) -> MagicMock:
        self.posts.append(json)
        self.subscribers.setdefault(json["email"], {}).update(json.get("fields") or {})
        return _response(200)


async def _consume(fields: subscriber_fields.Fields) -> None:
    event = subscriber_fields.audience_event(
        AudienceAction.LIFECYCLE_FIELDS, EMAIL, "user-1", fields
    )
    assert event is not None
    with patch(
        "backend.notifications.notifications._kept_out_since_queued",
        AsyncMock(return_value=False),
    ):
        assert await NotificationManager._process_audience_change(
            MagicMock(), event.model_dump_json()
        )


@pytest.mark.asyncio
async def test_only_the_fields_that_differ_are_written(mailerlite_configured):
    held = {
        "subscription_status": "subscribed",
        # MailerLite may hand a date back with a time part.
        "signup_date": "2026-07-01 00:00:00",
        "subscription_started_date": "2026-07-05",
        "checkout_opened_date": "2026-07-01",
    }
    ml = _FakeMailerLite({EMAIL: held})
    fields = {
        F.STATUS: "signed",
        F.SIGNUP: "2026-07-01",
        F.SUBSCRIPTION_STARTED: None,
        F.SUBSCRIPTION_ENDED: None,
    }
    with patch.object(mailerlite, "_client", return_value=ml):
        await _consume(fields)
    assert ml.posts == [
        {
            "email": EMAIL,
            "fields": {
                "subscription_status": "signed",
                "subscription_started_date": None,
            },
        }
    ]
    assert ml.subscribers[EMAIL]["checkout_opened_date"] == "2026-07-01"


@pytest.mark.asyncio
async def test_nothing_is_written_when_mailerlite_already_agrees(
    mailerlite_configured,
):
    ml = _FakeMailerLite({EMAIL: {"subscription_status": "signed"}})
    with patch.object(mailerlite, "_client", return_value=ml):
        await _consume({F.STATUS: "signed", F.SUBSCRIPTION_ENDED: None})
    assert ml.posts == []


@pytest.mark.asyncio
async def test_the_status_never_creates_a_subscriber(mailerlite_configured):
    """MailerLite holds checkout openers; a Stripe customer alone is not one."""
    ml = _FakeMailerLite()
    with patch.object(mailerlite, "_client", return_value=ml):
        await _consume({F.STATUS: "subscribed"})
    assert ml.posts == []
    assert EMAIL not in ml.subscribers
