"""Run only against a disposable database with TRIAL_TEST_DATABASE=1."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.models import (
    CreditTransaction,
    SubscriptionTrial,
    User,
    UserBalance,
    UserOnboarding,
)

from backend.data import credit, db, subscription_trial_checkout
from backend.data.credit import UserCredit
from backend.data.onboarding import _reward_user
from backend.data.onboarding_steps import OnboardingStep
from backend.data.subscription_checkout import subscription_checkout_lock
from backend.data.subscription_trial import (
    get_subscription_trial,
    record_subscription_trial_cost,
    reserve_subscription_trial,
)
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.data.subscription_trial_stripe import (
    Invoice,
    SubscriptionSnapshot,
    _save_snapshot,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires explicitly selected disposable trial database",
)


@pytest_asyncio.fixture
async def user_id():
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
    yield user_id
    await CreditTransaction.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete(where={"id": user_id})
    if owns_connection:
        await db.disconnect()


def offer(credits: int = 300) -> AcceptedTrialOffer:
    return AcceptedTrialOffer(
        version="integration-v1",
        new_users_from=datetime(2026, 9, 1, tzinfo=UTC),
        duration_days=7,
        tier="PRO",
        billing_cycle="monthly",
        daily_cost_limit=250_000,
        weekly_cost_limit=1_000_000,
        total_cost_limit=1_000_000,
        onboarding_credit_amount=credits,
        price_id="price_integration",
        unit_amount=2000,
        currency="usd",
    )


async def reserve(user_id: str, credits: int = 300):
    return await reserve_subscription_trial(
        user_id,
        offer(credits),
        "cus_integration",
        "https://example.com/success",
        "https://example.com/cancel",
        {"datafast_visitor_id": "visitor"},
    )


@pytest.mark.asyncio
async def test_parallel_enrollment_keeps_one_accepted_offer(user_id):
    results = await asyncio.gather(*[reserve(user_id, 300 + i) for i in range(12)])
    assert len({trial.id for trial in results}) == 1
    assert len({trial.offer.token for trial in results}) == 1
    assert await SubscriptionTrial.prisma().count(where={"userId": user_id}) == 1


@pytest.mark.asyncio
async def test_concurrent_checkout_is_rejected_and_lock_is_released(user_id):
    async with subscription_checkout_lock(user_id):
        with pytest.raises(ValueError, match="checkout is already"):
            async with subscription_checkout_lock(user_id):
                pytest.fail("A second checkout entered the critical section")
        async with subscription_checkout_lock(str(uuid4())):
            pass
    async with subscription_checkout_lock(user_id):
        pass


@pytest.mark.asyncio
async def test_checkout_failure_releases_lock(user_id):
    with pytest.raises(RuntimeError):
        async with subscription_checkout_lock(user_id):
            raise RuntimeError("Stripe failed")
    async with subscription_checkout_lock(user_id):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("trial_first", [True, False])
async def test_paid_and_trial_checkouts_cannot_run_together(user_id, trial_first):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def paused_checkout(*args, **kwargs):
        entered.set()
        await release.wait()
        return "https://checkout.stripe.com/test"

    async def trial_checkout():
        return await subscription_trial_checkout.create_trial_checkout(
            user_id,
            offer().token,
            "https://example.com/ok",
            "https://example.com/no",
            {},
        )

    async def paid_checkout():
        return await credit.create_subscription_checkout(
            user_id,
            credit.SubscriptionTier.PRO,
            "https://example.com/ok",
            "https://example.com/no",
        )

    first, second = (
        (trial_checkout, paid_checkout)
        if trial_first
        else (paid_checkout, trial_checkout)
    )
    with (
        patch.object(
            credit,
            "_create_subscription_checkout",
            AsyncMock(side_effect=paused_checkout),
        ),
        patch.object(
            subscription_trial_checkout,
            "_create_trial_checkout",
            AsyncMock(side_effect=paused_checkout),
        ),
    ):
        task = asyncio.create_task(first())
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            with pytest.raises(ValueError, match="checkout is already"):
                await second()
        finally:
            release.set()
            assert await task == "https://checkout.stripe.com/test"


@pytest.mark.asyncio
async def test_trial_cost_increments_are_atomic_and_do_not_reset(user_id):
    await reserve(user_id)
    await SubscriptionTrial.prisma().update(
        where={"userId": user_id}, data={"status": "trialing"}
    )
    await asyncio.gather(
        *[record_subscription_trial_cost(user_id, 100) for _ in range(30)]
    )
    trial = await get_subscription_trial(user_id)
    assert trial is not None and trial.cost_microdollars == 3000
    await reserve(user_id, 500)
    trial = await get_subscription_trial(user_id)
    assert trial is not None and trial.cost_microdollars == 3000


@pytest.mark.asyncio
async def test_conversion_invoice_survives_database_roundtrip_and_renewal(user_id):
    trial = await reserve(user_id)
    now = datetime.now(UTC)
    invoice = Invoice(
        id=f"in_first_{user_id}", status="paid", created=int(now.timestamp())
    )
    snapshot = SubscriptionSnapshot(
        id=f"sub_{user_id}",
        customer=trial.customer_id,
        status="active",
        latest_invoice=invoice,
    )
    async with db.transaction() as tx:
        await _save_snapshot(
            trial, snapshot, credit.SubscriptionTier.PRO, now, tx, True
        )
    converted = await get_subscription_trial(user_id)
    assert converted is not None and converted.converted_at is not None
    assert converted.conversion_invoice_id == f"in_first_{user_id}"
    invoice.id = f"in_renewal_{user_id}"
    async with db.transaction() as tx:
        await _save_snapshot(
            converted, snapshot, credit.SubscriptionTier.PRO, now, tx, True
        )
    renewed = await get_subscription_trial(user_id)
    assert renewed is not None
    assert renewed.conversion_invoice_id == converted.conversion_invoice_id
    assert renewed.converted_at == converted.converted_at
    assert trial.offer.onboarding_credit_amount == 300


@pytest.mark.asyncio
async def test_trial_reuses_onboarding_reward_key_and_never_grants_twice(user_id):
    await reserve(user_id, 500)
    now = datetime.now(UTC)
    await SubscriptionTrial.prisma().update(
        where={"userId": user_id},
        data={
            "status": "trialing",
            "cardVerifiedAt": now,
            "consumedAt": now,
            "endsAt": now + timedelta(days=7),
        },
    )
    onboarding = await UserOnboarding.prisma().create(data={"userId": user_id})
    with (
        patch(
            "backend.data.onboarding.get_user_credit_model",
            AsyncMock(return_value=UserCredit()),
        ),
        patch(
            "backend.executor.billing.clear_insufficient_funds_notifications",
            AsyncMock(),
        ),
    ):
        await _reward_user(user_id, onboarding, OnboardingStep.ONBOARDING_COMPLETE)
        await _reward_user(user_id, onboarding, OnboardingStep.ONBOARDING_COMPLETE)
    balance = await UserBalance.prisma().find_unique_or_raise(where={"userId": user_id})
    assert balance.balance == 500
    transactions = await CreditTransaction.prisma().find_many(where={"userId": user_id})
    assert len(transactions) == 1
    assert transactions[0].transactionKey == f"REWARD-{user_id}-ONBOARDING_COMPLETE"


@pytest.mark.asyncio
async def test_existing_onboarding_recipient_gets_no_trial_topup(user_id):
    onboarding = await UserOnboarding.prisma().create(data={"userId": user_id})
    with (
        patch(
            "backend.data.onboarding.get_user_credit_model",
            AsyncMock(return_value=UserCredit()),
        ),
        patch(
            "backend.executor.billing.clear_insufficient_funds_notifications",
            AsyncMock(),
        ),
    ):
        await _reward_user(user_id, onboarding, OnboardingStep.ONBOARDING_COMPLETE)
        await reserve(user_id, 500)
        await SubscriptionTrial.prisma().update(
            where={"userId": user_id}, data={"consumedAt": datetime.now(UTC)}
        )
        await _reward_user(user_id, onboarding, OnboardingStep.ONBOARDING_COMPLETE)
    balance = await UserBalance.prisma().find_unique_or_raise(where={"userId": user_id})
    assert balance.balance == 300
    assert await CreditTransaction.prisma().count(where={"userId": user_id}) == 1
