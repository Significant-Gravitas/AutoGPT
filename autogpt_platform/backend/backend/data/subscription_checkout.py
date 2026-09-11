"""Coordinate paid and trial Checkout creation for the same account."""

from contextlib import asynccontextmanager
from datetime import timedelta

import stripe
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema, transaction
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_trial import get_subscription_trial


class SubscriptionCheckoutUnavailable(ValueError):
    pass


class CheckoutLock(BaseModel):
    acquired: bool


@asynccontextmanager
async def subscription_checkout_lock(user_id: str):
    async with transaction(timeout=timedelta(seconds=120)) as tx:
        locks = await query_raw_with_schema(
            "SELECT pg_try_advisory_xact_lock(hashtextextended($1, 0)) AS acquired",
            f"subscription-checkout:{user_id}",
            client=tx,
            model=CheckoutLock,
        )
        if not locks or not locks[0].acquired:
            raise SubscriptionCheckoutUnavailable(
                "Another checkout is already starting. Please retry."
            )
        yield


async def expire_other_subscription_checkouts(
    customer_id: str, keep_session_id: str | None = None
) -> None:
    sessions = await stripe_call(
        stripe.checkout.Session.list_async,
        customer=customer_id,
        status="open",
        limit=100,
    )
    async for session in stripe_list_items(sessions):
        if session.mode == "subscription" and session.id != keep_session_id:
            await stripe_call(stripe.checkout.Session.expire_async, session.id)


async def ensure_no_unconverted_trial(user_id: str, customer_id: str) -> None:
    trial = await get_subscription_trial(user_id)
    if trial is None or trial.converted_at:
        return
    subscriptions = await stripe_call(
        stripe.Subscription.list_async, customer=customer_id, status="all", limit=100
    )
    async for subscription in stripe_list_items(subscriptions):
        if (subscription.metadata or {}).get(
            "trial_enrollment_id"
        ) == trial.id and subscription.status not in ("canceled", "incomplete_expired"):
            raise SubscriptionCheckoutUnavailable(
                "This account already has a trial subscription. "
                "Manage it in billing before starting another plan."
            )
