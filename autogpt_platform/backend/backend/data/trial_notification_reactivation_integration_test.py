import asyncio
from datetime import UTC, datetime

import pytest
from prisma.models import SubscriptionTrial, TrialNotificationDelivery

from backend.data import trial_notification_recovery as recovery
from backend.data import trial_notifications as outbox
from backend.data import trial_notifications_integration_test as fixtures
from backend.data.trial_notification_recovery_integration_test import activate
from backend.notifications.trial import trial_notice_data, trial_notice_key

enrollment = fixtures.enrollment
payload = fixtures.payload
pytestmark = fixtures.pytestmark


async def suppress(enrollment, payload):
    receipt = await fixtures.enqueue(enrollment, payload)
    delivery = await outbox.claim_trial_notification(receipt.id)
    assert delivery is not None
    assert await outbox.finish_trial_notification(
        receipt.id, delivery.lease_token, "suppressed"
    )
    return receipt


@pytest.mark.asyncio
async def test_concurrent_reactivation_keeps_original_identity_payload_and_attempts(
    enrollment, payload
):
    receipt = await suppress(enrollment, payload)
    changed = payload.model_copy(update={"user_name": "Changed"})
    repeats = await asyncio.gather(
        *[fixtures.enqueue(enrollment, changed) for _ in range(10)]
    )
    assert all(row.id == receipt.id and not row.created for row in repeats)
    row = await TrialNotificationDelivery.prisma().find_unique_or_raise(
        where={"id": receipt.id}
    )
    assert row.status == "pending" and row.attempts == 1
    assert row.payload == payload.model_dump(mode="json")
    assert receipt.id in await outbox.get_due_trial_notifications()
    claims = await asyncio.gather(
        *[outbox.claim_trial_notification(receipt.id) for _ in range(10)]
    )
    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1 and winners[0].attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"status": "accepted", "providerMessageId": "message"},
        {"status": "failed"},
        {"status": "obsolete"},
        {"status": "pending"},
        {"status": "sending", "leaseToken": "current-owner"},
        {"status": "suppressed", "providerMessageId": "already-sent"},
        {"status": "suppressed", "acceptedAt": datetime.now(UTC)},
        {"status": "suppressed", "leaseToken": "still-owned"},
        {"status": "suppressed", "leaseExpiresAt": datetime.now(UTC)},
    ],
)
async def test_reactivation_cannot_overwrite_terminal_acceptance_or_owned_work(
    enrollment, payload, changes
):
    receipt = await suppress(enrollment, payload)
    before = await TrialNotificationDelivery.prisma().update(
        where={"id": receipt.id}, data=changes
    )
    await fixtures.enqueue(enrollment, payload)
    after = await TrialNotificationDelivery.prisma().find_unique_or_raise(
        where={"id": receipt.id}
    )
    assert after == before


@pytest.mark.asyncio
async def test_exhausted_suppression_becomes_failed_without_resetting_budget(
    enrollment, payload
):
    receipt = await suppress(enrollment, payload)
    await TrialNotificationDelivery.prisma().update(
        where={"id": receipt.id}, data={"attempts": outbox.MAX_DELIVERY_ATTEMPTS}
    )
    await fixtures.enqueue(enrollment, payload)
    row = await TrialNotificationDelivery.prisma().find_unique_or_raise(
        where={"id": receipt.id}
    )
    assert row.status == "failed" and row.attempts == outbox.MAX_DELIVERY_ATTEMPTS
    assert row.lastError == "suppression_recovery_attempts_exhausted"
    assert await outbox.claim_trial_notification(receipt.id) is None


@pytest.mark.asyncio
async def test_restored_card_makes_original_notice_repairable(enrollment):
    trial = await activate(enrollment)
    data = trial_notice_data(trial, "started", "Sam")
    data.notice_key = trial_notice_key(trial, "started")
    receipt = await suppress(enrollment, data)
    await SubscriptionTrial.prisma().update(
        where={"id": trial.id}, data={"cardVerifiedAt": None}
    )
    assert trial.id not in [
        row.id for row in await recovery.get_trial_notice_candidates()
    ]
    await SubscriptionTrial.prisma().update(
        where={"id": trial.id}, data={"cardVerifiedAt": datetime.now(UTC)}
    )
    assert trial.id in [row.id for row in await recovery.get_trial_notice_candidates()]
    await fixtures.enqueue(enrollment, data)
    assert trial.id not in [
        row.id for row in await recovery.get_trial_notice_candidates()
    ]
    assert (
        await TrialNotificationDelivery.prisma().count(where={"trialId": trial.id}) == 1
    )
    row = await TrialNotificationDelivery.prisma().find_unique_or_raise(
        where={"id": receipt.id}
    )
    assert row.status == "pending"
