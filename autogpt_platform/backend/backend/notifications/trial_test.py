from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import NotificationResult
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.notifications import trial as notices
from backend.notifications.renderer import EmailUrls, render


@pytest.fixture
def trial() -> TrialState:
    now = datetime.now(UTC)
    return TrialState(
        id="trial-1",
        user_id="user-1",
        customer_id="cus_1",
        offer=AcceptedTrialOffer(
            version="offer-v1",
            new_users_from=now - timedelta(days=1),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=250_000,
            weekly_cost_limit=1_000_000,
            total_cost_limit=1_000_000,
            onboarding_credit_amount=300,
            price_id="price_pro",
            unit_amount=2000,
            currency="usd",
        ),
        checkout_session_id="cs_1",
        subscription_id="sub_1",
        checkout_attempt=0,
        success_url="https://example.com",
        cancel_url="https://example.com",
        checkout_metadata={},
        status="trialing",
        card_verified_at=now,
        started_at=now,
        ends_at=now + timedelta(days=7),
        consumed_at=now,
        converted_at=None,
        cancel_at_period_end=False,
        cost_microdollars=0,
    )


@pytest.fixture
def urls() -> EmailUrls:
    return EmailUrls(
        chat="https://example.com/copilot",
        dashboard="https://example.com/library",
        settings="https://example.com/settings",
        unsubscribe="https://example.com/unsubscribe",
        attention="https://example.com/attention",
        billing="https://example.com/settings/billing",
        prefs="https://example.com/prefs",
        marketplace="https://example.com/marketplace",
        docs="https://example.com/docs",
        discord="https://example.com/discord",
    )


@pytest.mark.parametrize("cycle,cycle_noun", [("monthly", "month"), ("yearly", "year")])
@pytest.mark.parametrize(
    "currency,formatted_amount",
    [
        ("usd", "$20.00"),
        ("eur", "€20.00"),
        ("gbp", "£20.00"),
        ("cad", "20.00 CAD"),
        ("jpy", "2,000 JPY"),
    ],
)
def test_trial_plan_uses_design_labels_and_unambiguous_prices(
    trial, urls, cycle, cycle_noun, currency, formatted_amount
):
    trial.offer = trial.offer.model_copy(
        update={"billing_cycle": cycle, "currency": currency}
    )
    data = notices.trial_notice_data(trial, "ending", "Sam")

    assert data.plan.label == f"Pro · {cycle}"
    assert data.plan.price_display == f"{formatted_amount} / {cycle_noun}"
    email = render(NotificationType.TRIAL_UPDATE, data, "sam@example.com", urls)
    for body in (email.html, email.text):
        assert data.plan.label in body
        assert data.plan.price_display in body


@pytest.mark.parametrize(
    "kind",
    [
        "started",
        "ending",
        "expiring",
        "last_call",
        "canceled",
        "resumed",
        "ended",
        "converted",
        "payment_failed",
    ],
)
def test_all_trial_notices_render_exact_terms_and_billing_link(trial, urls, kind):
    data = notices.trial_notice_data(trial, kind, "Sam")
    email = render(NotificationType.TRIAL_UPDATE, data, "sam@example.com", urls)
    assert email.subject and email.preheader
    for body in (email.html, email.text):
        assert "$20.00 / month" in body
        if kind != "converted":
            assert data.ends_label in body
        else:
            assert "Trial ends" not in body
        assert urls.billing in body
    if kind == "payment_failed":
        assert "paid access is paused" in email.text
        assert "keep running normally" not in email.html
    if kind == "canceled":
        assert "will not convert" in email.text


def test_trial_email_escapes_user_supplied_name(trial, urls):
    data = notices.trial_notice_data(trial, "started", "<script>alert(1)</script>")
    email = render(NotificationType.TRIAL_UPDATE, data, "sam@example.com", urls)
    assert "<script>" not in email.html
    assert "&lt;script&gt;" in email.html


def test_trial_welcome_does_not_advertise_onboarding_credits(trial, urls):
    data = notices.trial_notice_data(trial, "started", "Sam")
    email = render(NotificationType.TRIAL_UPDATE, data, "sam@example.com", urls)
    for body in (email.html, email.text):
        assert "onboarding credits" not in body
        assert "automation credits" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["sent", "duplicate", "publish_failed", "raised"])
async def test_notice_uses_shared_notification_queue(trial, outcome):
    raw = {
        "id": trial.subscription_id,
        "customer": trial.customer_id,
        "status": "trialing",
        "trial_end": int(trial.ends_at.timestamp()),
        "metadata": {
            "trial_enrollment_id": trial.id,
            "user_id": trial.user_id,
            "trial_checkout_attempt": str(trial.checkout_attempt),
        },
    }
    queue = AsyncMock(
        return_value=NotificationResult(success=outcome != "publish_failed"),
        side_effect=RuntimeError("queue unavailable") if outcome == "raised" else None,
    )
    with (
        patch.object(notices, "stripe_call", AsyncMock(return_value=raw)),
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=trial)
            ),
        ),
        patch.object(
            notices,
            "user_db",
            return_value=MagicMock(
                get_user_by_id=AsyncMock(
                    return_value=SimpleNamespace(
                        id="user-1",
                        # What email sign-up stores: the address's local part.
                        name="sam.carter",
                        email="sam.carter@example.com",
                        marketing_opt_out_at=None,
                        timezone="America/Chicago",
                    )
                )
            ),
        ),
        patch.object(
            notices, "claim_once", AsyncMock(return_value=outcome != "duplicate")
        ),
        patch.object(notices, "release_claim", AsyncMock()) as release,
        patch.object(notices, "queue_notification_async", queue),
        patch.object(notices, "_track_billing_event") as track,
    ):
        if outcome in ("publish_failed", "raised"):
            with pytest.raises(RuntimeError):
                await notices.notify_trial(raw, "started")
        else:
            assert await notices.notify_trial(raw, "started")
    if outcome == "duplicate":
        queue.assert_not_awaited()
    else:
        queue.assert_awaited_once()
        event = queue.await_args.args[0]
        assert event.type == NotificationType.TRIAL_UPDATE
        assert event.user_id == trial.user_id
        assert event.data.notice_key == notices.trial_notice_key(trial, "started")
        assert event.data.user_name == "there"
    if outcome in ("publish_failed", "raised"):
        release.assert_awaited_once_with(notices.trial_notice_key(trial, "started"))
    else:
        release.assert_not_awaited()
    assert track.call_count == int(outcome == "sent")
    if outcome == "sent":
        assert track.call_args.args[0] == "trial_started"


def test_every_trial_notice_kind_has_the_analytics_plan_event_name():
    """Every kind maps to the analytics plan's ``trial_*`` name; a failed
    conversion charge is the plan's ``payment_failed``."""
    kinds = get_args(notices.TrialNoticeKind)
    assert set(notices.TRIAL_NOTICE_EVENTS) == set(kinds)
    for kind in kinds:
        expected = "payment_failed" if kind == "payment_failed" else f"trial_{kind}"
        assert notices.TRIAL_NOTICE_EVENTS[kind].value == expected


def test_canceled_trial_suppresses_late_ending_reminder(trial):
    raw = {
        "status": "trialing",
        "trial_end": int(trial.ends_at.timestamp()),
        "cancel_at_period_end": True,
    }
    assert not notices._notice_applies(trial, "ending", raw)
    assert notices._notice_applies(trial, "canceled", raw)


def test_ending_reminder_waits_for_current_reminder_window(trial):
    raw = {
        "status": "trialing",
        "trial_end": int((datetime.now(UTC) + timedelta(days=7)).timestamp()),
        "cancel_at_period_end": False,
    }
    assert not notices._notice_applies(trial, "ending", raw)


def test_ending_reminder_sent_just_before_three_day_mark(trial):
    """Stripe's trial_will_end lead time is three days. An event created or
    processed a little before that mark is still the reminder."""
    raw = {
        "status": "trialing",
        "trial_end": int((datetime.now(UTC) + timedelta(days=3, hours=1)).timestamp()),
        "cancel_at_period_end": False,
    }
    assert notices._notice_applies(trial, "ending", raw)


def test_ending_reminder_dropped_just_past_reminder_window(trial):
    trial_end = datetime.now(UTC) + notices.TRIAL_REMINDER_WINDOW + timedelta(hours=1)
    raw = {
        "status": "trialing",
        "trial_end": int(trial_end.timestamp()),
        "cancel_at_period_end": False,
    }
    assert notices._notice_skip_reason(trial, "ending", raw) == (
        "trial ends after the reminder window"
    )


def test_replayed_ending_reminder_for_ended_trial_names_the_real_reason(trial):
    raw = {
        "status": "canceled",
        "trial_end": int((datetime.now(UTC) + timedelta(days=7)).timestamp()),
        "cancel_at_period_end": False,
    }
    assert notices._notice_skip_reason(trial, "ending", raw) == (
        "trial is no longer running"
    )


@pytest.mark.parametrize(
    "kind, current, reason",
    [
        ("converted", {}, "subscription is unknown, not converted"),
        (
            "payment_failed",
            {"status": "active"},
            "subscription is active, not a failed conversion",
        ),
        (
            "ended",
            {"status": "trialing"},
            "subscription is trialing, not an ended trial",
        ),
        ("canceled", {"status": "trialing"}, "cancellation is not scheduled"),
    ],
)
def test_skip_reason_names_why_the_notice_no_longer_applies(
    trial, kind, current, reason
):
    current = {**current, "trial_end": int(trial.ends_at.timestamp())}
    assert notices._notice_skip_reason(trial, kind, current) == reason


@pytest.mark.asyncio
async def test_dropped_notice_is_logged_without_sending(trial, caplog):
    raw = {
        "id": trial.subscription_id,
        "customer": trial.customer_id,
        "status": "trialing",
        "trial_end": int((datetime.now(UTC) + timedelta(days=7)).timestamp()),
        "cancel_at_period_end": False,
        "metadata": {
            "trial_enrollment_id": trial.id,
            "user_id": trial.user_id,
            "trial_checkout_attempt": str(trial.checkout_attempt),
        },
    }
    queue = AsyncMock()
    with (
        patch.object(notices, "stripe_call", AsyncMock(return_value=raw)),
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=trial)
            ),
        ),
        patch.object(notices, "queue_notification_async", queue),
        patch.object(notices, "_track_billing_event") as track,
        caplog.at_level("INFO", logger=notices.logger.name),
    ):
        assert await notices.notify_trial(raw, "ending")
    queue.assert_not_awaited()
    track.assert_not_called()
    [record] = [r for r in caplog.records if r.name == notices.logger.name]
    assert record.levelname == "INFO"
    message = record.getMessage()
    assert "ending" in message
    assert trial.user_id in message
    assert "reminder window" in message


def test_failed_conversion_suppresses_paid_confirmation(trial):
    assert not notices._notice_applies(trial, "converted", {"status": "past_due"})


def test_cancellation_notice_key_changes_only_with_durable_revision(trial):
    first = notices.trial_notice_key(trial, "canceled")
    assert notices.trial_notice_key(trial, "canceled") == first
    trial.notification_revision += 1
    assert notices.trial_notice_key(trial, "canceled") != first
    assert notices.trial_notice_key(trial, "resumed") != notices.trial_notice_key(
        trial, "canceled"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invoice_id,expected", [("in_first", True), ("in_renewal", False)]
)
async def test_only_conversion_invoice_triggers_trial_confirmation(
    trial, invoice_id, expected
):
    trial = trial.model_copy(
        update={"converted_at": datetime.now(UTC), "conversion_invoice_id": "in_first"}
    )
    subscription = {
        "id": "sub_1",
        "customer": "cus_1",
        "metadata": {
            "user_id": trial.user_id,
            "trial_enrollment_id": trial.id,
            "trial_checkout_attempt": "0",
        },
    }
    invoice = {
        "id": invoice_id,
        "subscription": "sub_1",
        "subscription_details": {"metadata": subscription["metadata"]},
    }
    with (
        patch.object(
            notices.stripe.Subscription,
            "retrieve_async",
            AsyncMock(return_value=subscription),
        ),
        patch.object(notices, "sync_subscription_from_stripe", AsyncMock()),
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=trial)
            ),
        ),
        patch.object(notices, "notify_trial", AsyncMock()) as notify,
    ):
        assert await notices.on_trial_invoice(invoice, paid=True) is expected
    if expected:
        notify.assert_awaited_once_with(subscription, "converted")
    else:
        notify.assert_not_awaited()


def _pending(trial: TrialState, left: timedelta, canceled_ago=timedelta(days=2)):
    now = datetime.now(UTC)
    return {
        "status": "trialing",
        "trial_end": int((now + left).timestamp()),
        "cancel_at_period_end": True,
        "canceled_at": int((now - canceled_ago).timestamp()),
    }


@pytest.mark.parametrize("kind", ["expiring", "last_call"])
def test_not_renewing_notices_offer_resume_and_never_mention_a_charge(
    trial, urls, kind
):
    data = notices.trial_notice_data(trial, kind, "Sam")
    email = render(NotificationType.TRIAL_UPDATE, data, "sam@example.com", urls)
    assert "set not to renew" in email.preheader
    for body in (email.html, email.text):
        assert "Resume my trial" in body
        assert "Auto-renew" in body
        assert "won't be charged" in body
        assert "Cancel trial" not in body
        assert "will be charged" not in body
    if kind == "last_call":
        assert email.subject == "Your AutoGPT access ends tomorrow"
    else:
        assert email.subject == "Your AutoGPT trial ends in 3 days"
    # The plain-text part keeps each block on its own paragraph.
    assert "Auto-renew: Off\n\nIf you resume:" in email.text
    assert "added.\n\nYou received this" in email.text
    # A service message: no urgency styling or sales framing.
    assert "Last chance" not in email.subject
    assert "#FFF6E8" not in email.html  # the amber attention strip
    assert "#B45309" not in email.html  # the needs-you eyebrow


def test_three_day_reminder_follows_the_renewal_setting(trial):
    """Exactly one of the two three-day reminders applies to a trial."""
    pending = _pending(trial, timedelta(days=3))
    assert notices._notice_applies(trial, "expiring", pending)
    assert not notices._notice_applies(trial, "ending", pending)
    renewing = {**pending, "cancel_at_period_end": False, "canceled_at": None}
    assert notices._notice_applies(trial, "ending", renewing)
    assert notices._notice_skip_reason(trial, "expiring", renewing) == (
        "cancellation is not scheduled"
    )
    renewing_tomorrow = {
        **_pending(trial, timedelta(hours=24)),
        "cancel_at_period_end": False,
        "canceled_at": None,
    }
    assert notices._notice_skip_reason(trial, "last_call", renewing_tomorrow) == (
        "cancellation is not scheduled"
    )


@pytest.mark.parametrize("kind,left", [("expiring", 3), ("last_call", 1)])
def test_not_renewing_notice_skips_a_cancellation_from_the_last_day(trial, kind, left):
    raw = _pending(trial, timedelta(days=left), canceled_ago=timedelta(hours=5))
    assert notices._notice_skip_reason(trial, kind, raw) == (
        "canceled within the last day"
    )


@pytest.mark.parametrize(
    "left,reason",
    [
        (timedelta(hours=30), "trial ends after the last-call window"),
        (timedelta(hours=24, minutes=30), None),
        (timedelta(hours=19), None),
        (timedelta(hours=10), "too little of the trial is left for a last call"),
    ],
)
def test_last_call_only_goes_out_the_day_before_the_end(trial, left, reason):
    raw = _pending(trial, left)
    assert notices._notice_skip_reason(trial, "last_call", raw) == reason


def test_not_renewing_notices_are_claimed_once_per_trial_end(trial):
    keys = {
        kind: notices.trial_notice_key(trial, kind)
        for kind in ("ending", "expiring", "last_call")
    }
    assert len(set(keys.values())) == 3
    trial.notification_revision += 1
    for kind, key in keys.items():
        assert notices.trial_notice_key(trial, kind) == key
    trial.ends_at += timedelta(days=1)
    for kind, key in keys.items():
        assert notices.trial_notice_key(trial, kind) != key


@pytest.mark.asyncio
async def test_converted_trial_gets_no_not_renewing_notice(trial):
    converted = trial.model_copy(update={"converted_at": datetime.now(UTC)})
    raw = {
        "id": trial.subscription_id,
        "customer": trial.customer_id,
        "metadata": {
            "trial_enrollment_id": trial.id,
            "user_id": trial.user_id,
            "trial_checkout_attempt": str(trial.checkout_attempt),
        },
    }
    queue = AsyncMock()
    with (
        patch.object(notices, "stripe_call", AsyncMock(return_value=raw)),
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=converted)
            ),
        ),
        patch.object(notices, "queue_notification_async", queue),
    ):
        for kind in ("expiring", "last_call"):
            assert not await notices.notify_trial(raw, kind)
    queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_last_call_pass_notifies_each_due_trial_despite_one_failure(trial):
    second = trial.model_copy(
        update={"id": "trial-2", "user_id": "user-2", "subscription_id": "sub_2"}
    )
    listing = AsyncMock(return_value=[trial, second])
    notify = AsyncMock(side_effect=[RuntimeError("stripe down"), True])
    with (
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(list_trials_ending_unrenewed=listing),
        ),
        patch.object(notices, "notify_trial", notify),
        patch.object(notices, "is_claimed", AsyncMock(return_value=False)),
    ):
        assert await notices.send_trial_last_calls() == 2
    after, until = listing.await_args.args
    assert until - after == notices.LAST_CALL_WINDOW - notices.LAST_CALL_MIN_LEAD
    assert [c.args for c in notify.await_args_list] == [
        ({"id": "sub_1", "metadata": {"trial_enrollment_id": "trial-1"}}, "last_call"),
        ({"id": "sub_2", "metadata": {"trial_enrollment_id": "trial-2"}}, "last_call"),
    ]


def test_late_three_day_reminder_is_dropped_rather_than_miscounting(trial):
    """The subject says "ends in 3 days"; a trial_will_end handled after an
    outage leaves the trial to the last call instead."""
    assert notices._notice_applies(
        trial, "expiring", _pending(trial, timedelta(days=3))
    )
    late = _pending(trial, timedelta(days=2))
    assert notices._notice_skip_reason(trial, "expiring", late) == (
        "too little of the trial is left for a three-day reminder"
    )


@pytest.mark.asyncio
async def test_last_call_pass_skips_trials_it_already_notified(trial):
    second = trial.model_copy(update={"id": "trial-2", "subscription_id": "sub_2"})
    listing = AsyncMock(return_value=[trial, second])
    notify = AsyncMock(return_value=True)
    sent = notices.trial_notice_key(trial, "last_call")
    with (
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(list_trials_ending_unrenewed=listing),
        ),
        patch.object(notices, "notify_trial", notify),
        patch.object(notices, "is_claimed", AsyncMock(side_effect=lambda k: k == sent)),
    ):
        await notices.send_trial_last_calls()
    [call] = notify.await_args_list
    assert call.args[0]["id"] == "sub_2"


def test_three_day_reminder_only_goes_near_the_three_day_mark(trial):
    early = _pending(trial, timedelta(days=3, hours=20))
    assert notices._notice_skip_reason(trial, "expiring", early) == (
        "trial ends after the reminder window"
    )
