import asyncio
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.models import TrialNotificationDelivery, User

from backend.data import db
from backend.data import trial_notifications as outbox
from backend.data.notifications import SubscriptionPlan, TrialUpdateData
from backend.data.subscription_trial import reserve_subscription_trial
from backend.data.subscription_trial_config import AcceptedTrialOffer

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable trial database",
)


@pytest_asyncio.fixture
async def enrollment():
    target = urlparse(db.DATABASE_URL)
    local = (target.hostname, target.port, target.path) == (
        "127.0.0.1",
        15432,
        "/trial_test",
    )
    ci = os.environ.get("GITHUB_ACTIONS") == "true" and (
        target.hostname,
        target.port,
        target.path,
    ) == ("localhost", 5432, "/postgres")
    assert (
        local or ci
    ), "Trial integration tests require an approved disposable database"
    owns_connection = not db.is_connected()
    await db.connect()
    user_id = str(uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    trial = await reserve_subscription_trial(
        user_id,
        AcceptedTrialOffer(
            version="outbox-v1",
            new_users_from=datetime.now(UTC),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=100,
            weekly_cost_limit=100,
            total_cost_limit=100,
            onboarding_credit_amount=300,
            price_id="price_test",
            unit_amount=2000,
            currency="usd",
        ),
        "cus_test",
        "https://example.com/ok",
        "https://example.com/no",
        {},
    )
    yield trial
    await User.prisma().delete(where={"id": user_id})
    if owns_connection:
        await db.disconnect()


@pytest.fixture
def payload():
    return TrialUpdateData(
        user_name="Sam",
        kind="started",
        ends_label="17 Sep 2026",
        onboarding_credit_amount=300,
        offer_version="outbox-v1",
        plan=SubscriptionPlan(
            name="Pro",
            cycle="monthly",
            cycle_noun="month",
            label="Pro",
            price_display="$20.00 / month",
        ),
    )


async def enqueue(enrollment, payload):
    return await outbox.enqueue_trial_notification(
        enrollment.user_id, enrollment.id, f"trial:{enrollment.id}:started", payload
    )


@pytest.mark.asyncio
async def test_parallel_enqueue_keeps_one_immutable_intent(enrollment, payload):
    receipts = await asyncio.gather(*[enqueue(enrollment, payload) for _ in range(10)])
    assert sum(receipt.created for receipt in receipts) == 1
    assert len({receipt.id for receipt in receipts}) == 1
    replay = await enqueue(
        enrollment, payload.model_copy(update={"user_name": "Changed"})
    )
    row = await TrialNotificationDelivery.prisma().find_unique_or_raise(
        where={"id": replay.id}
    )
    assert not replay.created
    assert TrialUpdateData.model_validate(row.payload).user_name == "Sam"


@pytest.mark.asyncio
async def test_enqueue_rejects_other_users_enrollment(enrollment, payload):
    with pytest.raises(ValueError, match="ownership"):
        await outbox.enqueue_trial_notification(
            "other-user", enrollment.id, f"trial:{enrollment.id}:started", payload
        )
    assert (
        await TrialNotificationDelivery.prisma().count(where={"trialId": enrollment.id})
        == 0
    )


@pytest.mark.asyncio
async def test_parallel_claims_allow_one_sender(enrollment, payload):
    receipt = await enqueue(enrollment, payload)
    claims = await asyncio.gather(
        *[outbox.claim_trial_notification(receipt.id) for _ in range(10)]
    )
    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1 and winners[0].attempts == 1
    restored = outbox.ClaimedTrialDelivery.model_validate_json(
        winners[0].model_dump_json()
    )
    assert restored.payload == payload
    assert not await outbox.finish_trial_notification(
        receipt.id, "not-owner", "accepted", "wrong-message"
    )
    assert await outbox.finish_trial_notification(
        receipt.id, winners[0].lease_token, "accepted", f"message-{receipt.id}"
    )
    assert await outbox.claim_trial_notification(receipt.id) is None
    assert receipt.id not in await outbox.get_due_trial_notifications()


@pytest.mark.asyncio
async def test_expired_lease_recovers_and_fences_the_previous_sender(
    enrollment, payload
):
    receipt = await enqueue(enrollment, payload)
    first = await outbox.claim_trial_notification(receipt.id)
    row = await TrialNotificationDelivery.prisma().find_unique_or_raise(
        where={"id": receipt.id}
    )
    assert first is not None, (
        row.status,
        row.attempts,
        row.nextAttemptAt,
        datetime.now(UTC),
    )
    await TrialNotificationDelivery.prisma().update(
        where={"id": receipt.id},
        data={"leaseExpiresAt": datetime.now(UTC) - timedelta(seconds=1)},
    )
    assert receipt.id in await outbox.get_due_trial_notifications()
    second = await outbox.claim_trial_notification(receipt.id)
    assert second is not None and second.lease_token != first.lease_token
    assert second.attempts == 2
    assert not await outbox.finish_trial_notification(
        receipt.id, first.lease_token, "suppressed"
    )
    await outbox.retry_trial_notification(receipt.id, first.lease_token, "late-error")
    row = await TrialNotificationDelivery.prisma().find_unique_or_raise(
        where={"id": receipt.id}
    )
    assert row.status == "sending" and row.leaseToken == second.lease_token


@pytest.mark.asyncio
async def test_queue_ack_does_not_delay_the_consumer_claim(enrollment, payload):
    receipt = await enqueue(enrollment, payload)
    await outbox.mark_trial_notification_queued(receipt.id)
    assert receipt.id not in await outbox.get_due_trial_notifications()
    assert await outbox.claim_trial_notification(receipt.id) is not None


@pytest.mark.asyncio
async def test_failed_send_is_durable_and_backed_off(enrollment, payload):
    receipt = await enqueue(enrollment, payload)
    claim = await outbox.claim_trial_notification(receipt.id)
    assert claim is not None
    await outbox.retry_trial_notification(
        receipt.id, claim.lease_token, "ConnectTimeout"
    )
    row = await TrialNotificationDelivery.prisma().find_unique_or_raise(
        where={"id": receipt.id}
    )
    assert row.status == "pending" and row.lastError == "ConnectTimeout"
    assert row.nextAttemptAt > datetime.now(UTC)
    assert await outbox.claim_trial_notification(receipt.id) is None


@pytest.mark.asyncio
async def test_last_attempt_crash_becomes_visible_failure(enrollment, payload):
    receipt = await enqueue(enrollment, payload)
    await TrialNotificationDelivery.prisma().update(
        where={"id": receipt.id},
        data={
            "status": "sending",
            "attempts": outbox.MAX_DELIVERY_ATTEMPTS,
            "leaseExpiresAt": datetime.now(UTC) - timedelta(seconds=1),
        },
    )
    assert receipt.id not in await outbox.get_due_trial_notifications()
    row = await TrialNotificationDelivery.prisma().find_unique_or_raise(
        where={"id": receipt.id}
    )
    assert (
        row.status == "failed" and row.lastError == "lease_expired_after_last_attempt"
    )
