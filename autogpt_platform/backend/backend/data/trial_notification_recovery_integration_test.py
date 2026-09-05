from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.models import SubscriptionTrial, TrialNotificationDelivery

from backend.data import trial_notification_recovery as recovery
from backend.data import trial_notifications as outbox
from backend.data import trial_notifications_integration_test as fixtures
from backend.data.notifications import NotificationResult
from backend.data.subscription_trial import TrialState, get_subscription_trial
from backend.notifications import trial as notices
from backend.notifications import trial_recovery as worker
from backend.notifications.trial import trial_notice_data, trial_notice_key

enrollment = fixtures.enrollment
pytestmark = fixtures.pytestmark


async def activate(enrollment, **changes):
    now = datetime.now(UTC)
    row = await SubscriptionTrial.prisma().update(
        where={"id": enrollment.id},
        data={
            "stripeSubscriptionId": "sub_recovery",
            "status": "trialing",
            "consumedAt": now,
            "cardVerifiedAt": now,
            "startedAt": now,
            "endsAt": now + timedelta(days=7),
            **changes,
        },
    )
    return TrialState.from_db(row)


async def persist(trial, kind):
    return await outbox.enqueue_trial_notification(
        trial.user_id,
        trial.id,
        trial_notice_key(trial, kind),
        trial_notice_data(trial, kind, "Sam"),
    )


async def selected(trial, after_id=""):
    rows = await recovery.get_trial_notice_candidates(after_id)
    return trial.id in [row.id for row in rows]


@pytest.mark.asyncio
async def test_crash_before_intent_creation_is_discoverable(enrollment):
    trial = await activate(enrollment)
    assert await selected(trial)
    await persist(trial, "started")
    assert not await selected(trial)


@pytest.mark.asyncio
async def test_reminder_is_due_without_webhook_and_is_not_recreated(enrollment):
    trial = await activate(enrollment, endsAt=datetime.now(UTC) + timedelta(days=2))
    await persist(trial, "started")
    assert await selected(trial)
    receipt = await persist(trial, "ending")
    await TrialNotificationDelivery.prisma().update(
        where={"id": receipt.id}, data={"status": "failed"}
    )
    assert not await selected(trial)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes,kind",
    [
        ({"cancelAtPeriodEnd": True, "notificationRevision": 1}, "canceled"),
        ({"cancelAtPeriodEnd": False, "notificationRevision": 2}, "resumed"),
        ({"status": "past_due", "cardVerifiedAt": None}, "payment_failed"),
        ({"status": "unpaid", "cardVerifiedAt": None}, "payment_failed"),
        ({"status": "canceled", "cardVerifiedAt": None}, "ended"),
        ({"status": "paused", "cardVerifiedAt": None}, "ended"),
        (
            {
                "status": "active",
                "convertedAt": datetime.now(UTC),
                "stripeConversionInvoiceId": "in_conversion",
            },
            "converted",
        ),
    ],
)
async def test_each_customer_state_is_repairable(enrollment, changes, kind):
    trial = await activate(enrollment, **changes)
    await persist(trial, "started")
    assert await selected(trial)
    await persist(trial, kind)
    assert not await selected(trial)


@pytest.mark.asyncio
async def test_abandoned_checkout_does_not_receive_trial_mail(enrollment):
    assert not await selected(enrollment)


@pytest.mark.asyncio
async def test_cursor_does_not_repeat_an_unrepairable_candidate(enrollment):
    trial = await activate(enrollment)
    assert await selected(trial)
    assert not await selected(trial, after_id=trial.id)


@pytest.mark.asyncio
async def test_expired_trial_snapshot_is_rechecked_even_after_welcome(enrollment):
    trial = await activate(enrollment, endsAt=datetime.now(UTC) - timedelta(seconds=1))
    await persist(trial, "started")
    assert await selected(trial)


@pytest.mark.asyncio
async def test_recovery_persists_missing_welcome_and_reminder_during_queue_outage(
    enrollment,
):
    trial = await activate(enrollment, endsAt=datetime.now(UTC) + timedelta(days=2))
    subscription = {
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
    database = MagicMock(
        get_trial_notice_candidates=recovery.get_trial_notice_candidates,
        sync_subscription_from_stripe=AsyncMock(),
        get_subscription_trial=get_subscription_trial,
        enqueue_trial_notification=outbox.enqueue_trial_notification,
    )
    user = MagicMock()
    user.name = "Sam"
    with (
        patch.object(worker, "credit_db", return_value=database),
        patch.object(notices, "credit_db", return_value=database),
        patch.object(worker, "stripe_call", AsyncMock(return_value=subscription)),
        patch.object(notices, "stripe_call", AsyncMock(return_value=subscription)),
        patch.object(
            notices,
            "user_db",
            return_value=MagicMock(get_user_by_id=AsyncMock(return_value=user)),
        ),
        patch.object(
            notices,
            "queue_trial_delivery",
            AsyncMock(return_value=NotificationResult(success=False)),
        ),
        patch.object(notices, "_track_billing_event") as analytics,
    ):
        await worker.recover_missing_trial_notices()
        await worker.recover_missing_trial_notices()
    rows = await TrialNotificationDelivery.prisma().find_many(
        where={"trialId": trial.id}
    )
    assert {row.idempotencyKey for row in rows} == {
        trial_notice_key(trial, "started"),
        trial_notice_key(trial, "ending"),
    }
    assert all(row.status == "pending" for row in rows)
    assert analytics.call_count == 2
    database.sync_subscription_from_stripe.assert_awaited_once_with(subscription)
