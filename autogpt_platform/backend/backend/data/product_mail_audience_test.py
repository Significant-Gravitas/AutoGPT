"""The product-mail audience against the real database.

The rule is written twice: once for the consumer, which decides on one user,
and once as the where clause the hourly briefing pass selects with. These pin
that both give the same answer for the same accounts, and that a plan or trial
ending stamps the date the win-back window counts from.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from prisma import Json
from prisma.enums import SubscriptionTier
from prisma.models import SubscriptionTrial, User

from backend.data import product_mail_audience
from backend.data.credit import set_subscription_tier
from backend.data.product_mail_audience import (
    get_product_mail_standing,
    is_product_mail_recipient,
)
from backend.data.user import get_briefing_candidates
from backend.util.settings import UserNotificationsAudience
from backend.util.test import SpinTestServer


@pytest.fixture
def subscribers(monkeypatch: pytest.MonkeyPatch) -> None:
    config = product_mail_audience.settings.config
    monkeypatch.setattr(
        config, "user_notifications_audience", UserNotificationsAudience.SUBSCRIBERS
    )
    monkeypatch.setattr(config, "user_notifications_allowlist", "")


async def _account(
    zone: str,
    tier: SubscriptionTier = SubscriptionTier.NO_TIER,
    *,
    trial_ended_days_ago: int | None = None,
    plan_ended_days_ago: int | None = None,
) -> str:
    """A user with the weekly briefing on. The time zone must be a real one: a
    check constraint on User.timezone rejects anything Postgres can't read."""
    user_id = str(uuid4())
    now = datetime.now(UTC)
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"audience-{user_id}@example.com",
            "timezone": zone,
            "subscriptionTier": tier,
            "subscriptionEndedAt": (
                now - timedelta(days=plan_ended_days_ago)
                if plan_ended_days_ago is not None
                else None
            ),
        }
    )
    if trial_ended_days_ago is not None:
        await SubscriptionTrial.prisma().create(
            data={
                "userId": user_id,
                "offer": Json({}),
                "stripeCustomerId": f"cus_{user_id}",
                "checkoutSuccessUrl": "https://example.invalid/ok",
                "checkoutCancelUrl": "https://example.invalid/no",
                "status": "payment_failed",
                "endsAt": now - timedelta(days=trial_ended_days_ago),
            }
        )
    return user_id


@pytest.mark.asyncio(loop_scope="session")
async def test_the_briefing_pass_and_the_consumer_pick_the_same_accounts(
    server: SpinTestServer, subscribers: None
):
    # A quiet zone, but other accounts in a shared test database may be in it
    # too: the pass is walked in full and only these accounts are compared.
    zone = "Antarctica/Troll"
    ids: dict[str, str] = {}
    try:
        ids["free"] = await _account(zone)
        ids["trialist"] = await _account(zone, SubscriptionTier.TRIAL)
        ids["paying"] = await _account(zone, SubscriptionTier.PRO)
        ids["trial_ended_3_days_ago"] = await _account(zone, trial_ended_days_ago=3)
        ids["trial_ended_10_days_ago"] = await _account(zone, trial_ended_days_ago=10)
        ids["plan_ended_3_days_ago"] = await _account(zone, plan_ended_days_ago=3)
        ids["plan_ended_10_days_ago"] = await _account(zone, plan_ended_days_ago=10)
        expected = {
            "trialist",
            "paying",
            "trial_ended_3_days_ago",
            "plan_ended_3_days_ago",
        }

        candidate_ids: set[str] = set()
        after_id = None
        while page := await get_briefing_candidates([zone], after_id, 100):
            candidate_ids.update(c.id for c in page)
            after_id = page[-1].id
        now = datetime.now(UTC)
        consumer = set()
        for name, user_id in ids.items():
            standing = await get_product_mail_standing(user_id)
            assert standing is not None
            if is_product_mail_recipient(standing, now):
                consumer.add(name)

        by_id = {user_id: name for name, user_id in ids.items()}
        assert {by_id[i] for i in candidate_ids if i in by_id} == expected
        assert consumer == expected
    finally:
        await User.prisma().delete_many(where={"id": {"in": list(ids.values())}})


@pytest.mark.asyncio(loop_scope="session")
async def test_a_plan_ending_starts_the_window_and_a_new_one_clears_it(
    server: SpinTestServer,
):
    user_id = await _account("UTC", SubscriptionTier.PRO)
    try:
        await set_subscription_tier(user_id, SubscriptionTier.NO_TIER)
        ended = await User.prisma().find_unique_or_raise(where={"id": user_id})
        assert ended.subscriptionEndedAt is not None

        # A repeated downgrade (a webhook replay) doesn't restart the window.
        await set_subscription_tier(user_id, SubscriptionTier.NO_TIER)
        replayed = await User.prisma().find_unique_or_raise(where={"id": user_id})
        assert replayed.subscriptionEndedAt == ended.subscriptionEndedAt

        await set_subscription_tier(user_id, SubscriptionTier.PRO)
        resubscribed = await User.prisma().find_unique_or_raise(where={"id": user_id})
        assert resubscribed.subscriptionEndedAt is None
    finally:
        await User.prisma().delete_many(where={"id": user_id})
