"""Coordinate paid and trial Checkout creation for the same account."""

from contextlib import asynccontextmanager
from datetime import timedelta

import stripe
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema, transaction
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_trial import get_subscription_trial

_ENDED_STATUSES = ("canceled", "incomplete_expired")
ANOTHER_PLAN_LIVE = "Another plan is already active. Manage it in billing."


class SubscriptionCheckoutUnavailable(ValueError):
    pass


class AnotherPlanLive(SubscriptionCheckoutUnavailable):
    """A plan bought while the trial was cancel-pending has not ended it yet."""


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


async def other_plan_is_live(customer_id: str, exclude_subscription_id: str) -> bool:
    """Whether the customer has a plan besides this one that has not ended.

    A plan bought through Checkout while a trial is cancel-pending ends the
    trial only once its webhook is handled, and the stale-subscription cleanup
    can fail; until then both are live, so keeping the trial would bill twice.
    """
    subscriptions = await stripe_call(
        stripe.Subscription.list_async, customer=customer_id, status="all", limit=100
    )
    async for subscription in stripe_list_items(subscriptions):
        if (
            subscription.id != exclude_subscription_id
            and subscription.status not in _ENDED_STATUSES
        ):
            return True
    return False


async def ensure_no_unconverted_trial(user_id: str, customer_id: str) -> None:
    trial = await get_subscription_trial(user_id)
    if trial is None or trial.converted_at:
        return
    subscriptions = await stripe_call(
        stripe.Subscription.list_async, customer=customer_id, status="all", limit=100
    )
    trial_is_cancel_pending = another_plan_is_live = False
    async for subscription in stripe_list_items(subscriptions):
        live = subscription.status not in _ENDED_STATUSES
        if (subscription.metadata or {}).get("trial_enrollment_id") != trial.id:
            another_plan_is_live |= live
        elif not _ends_without_converting(subscription):
            raise SubscriptionCheckoutUnavailable(
                "This account already has a trial subscription. "
                "Manage it in billing before starting another plan."
            )
        else:
            trial_is_cancel_pending |= live
    # A plan bought while the trial is cancel-pending ends the trial only once
    # its webhook is handled; another Checkout before then would bill twice.
    if trial_is_cancel_pending and another_plan_is_live:
        raise AnotherPlanLive(ANOTHER_PLAN_LIVE)


def _ends_without_converting(subscription: stripe.Subscription) -> bool:
    """A cancel-pending trial never bills, and the stale-subscription cleanup
    ends it as soon as the new plan's subscription is active."""
    if subscription.status in _ENDED_STATUSES:
        return True
    return subscription.status == "trialing" and bool(
        subscription.get("cancel_at_period_end")
    )
