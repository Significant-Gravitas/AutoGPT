from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType

from backend.data.notifications import (
    NotificationResult,
    SubscriptionPlan,
    TrialUpdateData,
)
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


@pytest.mark.parametrize(
    "kind",
    [
        "started",
        "ending",
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
        assert "$20.00 USD / month" in body
        assert data.ends_label in body
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


@pytest.mark.asyncio
async def test_queue_failure_preserves_durable_notice_for_recovery(trial):
    raw = {
        "id": "sub_1",
        "customer": "cus_1",
        "status": "trialing",
        "metadata": {
            "trial_enrollment_id": trial.id,
            "user_id": trial.user_id,
            "trial_checkout_attempt": str(trial.checkout_attempt),
        },
        "trial_end": int(trial.ends_at.timestamp()),
    }
    persist = AsyncMock(return_value=MagicMock(id="notice-1", created=True))
    with (
        patch.object(
            notices.stripe.Subscription, "retrieve_async", AsyncMock(return_value=raw)
        ),
        patch.object(
            notices,
            "credit_db",
            return_value=MagicMock(
                get_subscription_trial=AsyncMock(return_value=trial),
                enqueue_trial_notification=persist,
            ),
        ),
        patch.object(
            notices,
            "user_db",
            return_value=MagicMock(
                get_user_by_id=AsyncMock(return_value=MagicMock(name="Sam"))
            ),
        ),
        patch.object(
            notices,
            "trial_notice_data",
            return_value=TrialUpdateData(
                user_name="Sam",
                kind="started",
                ends_label="17 Sep 2026",
                onboarding_credit_amount=300,
                offer_version="offer-v1",
                plan=SubscriptionPlan(
                    name="Pro",
                    cycle="monthly",
                    cycle_noun="month",
                    label="Pro",
                    price_display="$20.00 / month",
                ),
            ),
        ),
        patch.object(
            notices,
            "queue_trial_delivery",
            AsyncMock(return_value=NotificationResult(success=False)),
        ),
        patch.object(notices, "_track_billing_event") as track,
    ):
        assert await notices.notify_trial(raw, "started")
    persist.assert_awaited_once()
    track.assert_called_once()


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
