import asyncio
from uuid import uuid4

import pytest
from prisma.models import AuthAccount, AuthUser, SubscriptionTrialClaim, User

from backend.data import subscription_trial_integration_fixtures as fixtures
from backend.data.db import transaction
from backend.data.subscription_trial_claims import (
    claim_trial_identities,
    trial_identity_key,
)

enrollment = fixtures.enrollment
pytestmark = fixtures.pytestmark


async def claim(trial, fingerprint):
    async with transaction() as tx:
        return await claim_trial_identities(trial, fingerprint, tx)


@pytest.mark.asyncio
async def test_concurrent_accounts_cannot_share_one_intro_card(enrollment):
    fingerprint = f"fp_{uuid4()}"
    trials = [
        enrollment.model_copy(
            update={
                "id": f"{enrollment.id}-{index}",
                "user_id": str(uuid4()),
                "customer_id": f"cus_{uuid4()}",
            }
        )
        for index in range(10)
    ]
    results = await asyncio.gather(*(claim(trial, fingerprint) for trial in trials))
    assert sum(results) == 1
    winner = trials[results.index(True)]
    rows = await SubscriptionTrialClaim.prisma().find_many(
        where={"trialId": {"startsWith": enrollment.id}}
    )
    assert len(rows) == 3
    assert {row.trialId for row in rows} == {winner.id}
    assert await claim(winner, fingerprint)


@pytest.mark.asyncio
async def test_claim_survives_account_deletion_and_card_changes(enrollment):
    fingerprint = f"fp_{uuid4()}"
    assert await claim(enrollment, fingerprint)
    assert await claim(enrollment, f"fp_{uuid4()}")
    await User.prisma().delete(where={"id": enrollment.user_id})
    recreated = enrollment.model_copy(
        update={
            "id": f"{enrollment.id}-recreated",
            "user_id": str(uuid4()),
            "customer_id": f"cus_{uuid4()}",
        }
    )
    assert not await claim(recreated, fingerprint)
    assert await claim(recreated, f"fp_{uuid4()}")


@pytest.mark.asyncio
async def test_recreated_linked_login_cannot_claim_with_a_different_card(enrollment):
    subject = str(uuid4())
    other_user = str(uuid4())
    try:
        for user_id in (enrollment.user_id, other_user):
            await AuthUser.prisma().create(
                data={
                    "id": user_id,
                    "name": "Trial test",
                    "email": f"{user_id}@example.com",
                }
            )
            await AuthAccount.prisma().create(
                data={
                    "id": str(uuid4()),
                    "userId": user_id,
                    "providerId": "google",
                    "accountId": subject,
                }
            )
        assert await claim(enrollment, f"fp_{uuid4()}")
        await AuthUser.prisma().delete(where={"id": enrollment.user_id})
        recreated = enrollment.model_copy(
            update={
                "id": f"{enrollment.id}-recreated",
                "user_id": other_user,
                "customer_id": f"cus_{uuid4()}",
            }
        )
        assert not await claim(recreated, f"fp_{uuid4()}")
    finally:
        await AuthUser.prisma().delete_many(
            where={"id": {"in": [enrollment.user_id, other_user]}}
        )


@pytest.mark.asyncio
async def test_missing_fingerprint_fails_closed_without_claiming(enrollment):
    assert not await claim(enrollment, None)
    assert not await claim(enrollment, "")
    assert (
        await SubscriptionTrialClaim.prisma().count(where={"trialId": enrollment.id})
        == 0
    )


def test_claim_keys_are_domain_separated_and_do_not_store_raw_identifiers():
    key = trial_identity_key("card", "fp_example")
    assert len(key) == 64 and "fp_example" not in key
    assert key != trial_identity_key("customer", "fp_example")
