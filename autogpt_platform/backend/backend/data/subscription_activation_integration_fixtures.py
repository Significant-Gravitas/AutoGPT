"""Disposable PostgreSQL + real Redis Cluster activation test fixtures."""

import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse
from uuid import uuid4

import pytest_asyncio
import stripe
from prisma.enums import SubscriptionTier
from prisma.models import SubscriptionTrialClaim, User
from pydantic import BaseModel, ConfigDict

from backend.copilot.usage_activation import usage_keys
from backend.data import db
from backend.data.redis_client import (
    AsyncRedisClient,
    disconnect_async,
    get_redis_async,
)
from backend.data.subscription_activation import (
    lock_activation_user,
    publish_initial_pro_activation,
)
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import AcceptedTrialOffer
from backend.util.json import SafeJson


class ActivationCase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str
    subscription: dict
    invoices: list[dict]
    redis: AsyncRedisClient

    @property
    def invoice(self) -> dict:
        return self.invoices[0]

    async def activate(self, *, abort: bool = False) -> bool:
        async with db.transaction() as tx:
            user = await lock_activation_user(self.user_id, tx)
            row = await tx.subscriptiontrial.find_unique(where={"userId": user.id})
            allowed = await publish_initial_pro_activation(
                user,
                self.subscription,
                "price_pro",
                tx,
                TrialState.from_db(row) if row else None,
            )
            if allowed:
                await tx.user.update(
                    where={"id": user.id}, data={"subscriptionTier": "PRO"}
                )
            if abort:
                await tx.execute_raw("SELECT 1 / 0")
            return allowed

    async def add_trial(self, cost: int) -> TrialState:
        now = datetime.now(UTC)
        end = datetime.fromtimestamp(self.invoice["created"], UTC)
        offer = AcceptedTrialOffer(
            version="activation-integration-v1",
            new_users_from=now - timedelta(days=9),
            duration_days=7,
            tier="PRO",
            billing_cycle="monthly",
            daily_cost_limit=100,
            weekly_cost_limit=100,
            total_cost_limit=100,
            onboarding_credit_amount=300,
            price_id="price_pro",
            unit_amount=2000,
            currency="usd",
        )
        row = await db.prisma.subscriptiontrial.create(
            data={
                "userId": self.user_id,
                "stripeCustomerId": self.subscription["customer"],
                "stripeSubscriptionId": self.subscription["id"],
                "offer": SafeJson(offer.model_dump(mode="json")),
                "checkoutSuccessUrl": "https://example.com/chat?thread=kept",
                "checkoutCancelUrl": "https://example.com/chat?thread=kept",
                "consumedAt": now - timedelta(days=7),
                "startedAt": now - timedelta(days=7),
                "endsAt": end,
                "cardVerifiedAt": now - timedelta(days=7),
                "status": "trialing",
                "costMicrodollars": cost,
            }
        )
        await User.prisma().update(
            where={"id": self.user_id}, data={"subscriptionTier": "TRIAL"}
        )
        self.subscription["trial_end"] = int(end.timestamp())
        assert row.startedAt is not None
        self.subscription["trial_start"] = int(row.startedAt.timestamp())
        self.subscription["metadata"].update(
            {"trial_enrollment_id": row.id, "trial_checkout_attempt": "0"}
        )
        self.subscription["default_payment_method"] = {
            "id": "pm_test",
            "type": "card",
            "card": {
                "exp_month": 12,
                "exp_year": 2099,
                "fingerprint": f"fp_{self.user_id}",
            },
        }
        self.invoice["billing_reason"] = "subscription_cycle"
        self.invoice["lines"]["data"][0]["period"]["start"] = int(end.timestamp())
        return TrialState.from_db(row)

    async def counters(self, generation: str | None) -> tuple[int, int]:
        keys = usage_keys(self.user_id, generation, datetime.now(UTC))
        values = [int(await self.redis.get(key) or 0) for key in keys]
        return values[0], values[1]

    async def add_cost(self, generation: str | None, cost: int) -> None:
        for key in usage_keys(self.user_id, generation, datetime.now(UTC)):
            value = int(await self.redis.get(key) or 0) + cost
            await self.redis.set(key, value)


@pytest_asyncio.fixture
async def activation_case(mocker):
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
    ), "Requires the explicitly selected disposable activation database"
    owns_connection = not db.is_connected()
    await db.connect()
    user_id = str(uuid4())
    customer_id, subscription_id, invoice_id = (
        f"{prefix}_{user_id}" for prefix in ("cus", "sub", "in")
    )
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"{user_id}@example.com",
            "stripeCustomerId": customer_id,
            "subscriptionTier": SubscriptionTier.NO_TIER,
        }
    )
    now = int(datetime.now(UTC).timestamp())
    invoice = {
        "id": invoice_id,
        "object": "invoice",
        "customer": customer_id,
        "subscription": subscription_id,
        "status": "paid",
        "amount_remaining": 0,
        "amount_paid": 2000,
        "created": now,
        "status_transitions": {"paid_at": now},
        "billing_reason": "subscription_create",
        "lines": {
            "has_more": False,
            "data": [
                {
                    "id": f"il_{user_id}",
                    "object": "line_item",
                    "type": "subscription",
                    "subscription": subscription_id,
                    "amount": 2000,
                    "price": {"id": "price_pro"},
                    "quantity": 1,
                    "proration": False,
                    "period": {"start": now, "end": now + 30 * 86400},
                }
            ],
        },
    }
    case = ActivationCase(
        user_id=user_id,
        invoices=[invoice],
        redis=await get_redis_async(),
        subscription={
            "id": subscription_id,
            "object": "subscription",
            "customer": customer_id,
            "status": "active",
            "metadata": {"user_id": user_id},
            "latest_invoice": invoice_id,
            "trial_end": None,
            "items": {"data": [{"quantity": 1, "price": {"id": "price_pro"}}]},
        },
    )

    async def retrieve_invoice(invoice_id, **kwargs):
        return stripe.Invoice.construct_from(
            next(i for i in case.invoices if i["id"] == invoice_id), None
        )

    async def list_invoices(**kwargs):
        return stripe.ListObject.construct_from(
            {"object": "list", "data": case.invoices, "has_more": False}, None
        )

    async def retrieve_subscription(subscription_id, **kwargs):
        assert subscription_id == case.subscription["id"]
        data = deepcopy(case.subscription)
        if "latest_invoice" in kwargs.get("expand", []):
            data["latest_invoice"] = next(
                i for i in case.invoices if i["id"] == data["latest_invoice"]
            )
        return stripe.Subscription.construct_from(data, None)

    async def list_subscriptions(**kwargs):
        return stripe.ListObject.construct_from(
            {"object": "list", "data": [], "has_more": False}, None
        )

    mocker.patch("stripe.Invoice.retrieve_async", side_effect=retrieve_invoice)
    mocker.patch("stripe.Invoice.list_async", side_effect=list_invoices)
    mocker.patch(
        "stripe.Subscription.retrieve_async", side_effect=retrieve_subscription
    )
    mocker.patch("stripe.Subscription.list_async", side_effect=list_subscriptions)
    try:
        yield case
    finally:
        await db.prisma.credittransaction.delete_many(where={"userId": user_id})
        trial = await db.prisma.subscriptiontrial.find_unique(where={"userId": user_id})
        if trial:
            await SubscriptionTrialClaim.prisma().delete_many(
                where={"trialId": trial.id}
            )
        await User.prisma().delete_many(where={"id": user_id})
        async for key in case.redis.scan_iter(match=f"*{user_id}*"):
            await case.redis.delete(key)
        await disconnect_async()
        if owns_connection:
            await db.disconnect()
