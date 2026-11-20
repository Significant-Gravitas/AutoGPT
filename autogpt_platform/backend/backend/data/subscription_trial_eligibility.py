import stripe

from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_checkout import SubscriptionCheckoutUnavailable
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_config import TrialOffer
from backend.data.user import get_user_by_id


async def verify_trial_eligibility(
    user_id: str,
    customer_id: str,
    offer: TrialOffer,
    *,
    trial: TrialState | None = None,
    session: stripe.checkout.Session | None = None,
) -> None:
    user = await get_user_by_id(user_id)
    if trial and session and not _owned_checkout(trial, session):
        raise SubscriptionCheckoutUnavailable("Checkout does not match this trial")
    result = await stripe_call(
        stripe.Subscription.list_async, customer=customer_id, status="all", limit=100
    )
    subscriptions = [sub async for sub in stripe_list_items(result)]
    allowed = await _unfinished_subscriptions(trial, session) if subscriptions else {}
    has_history = any(
        sub.status not in allowed.get(sub.id, ()) for sub in subscriptions
    )
    if not offer.is_eligible(
        created_at=user.created_at,
        current_tier=user.subscription_tier.value,
        has_subscription_history=has_history,
    ):
        raise SubscriptionCheckoutUnavailable(
            "This account is not eligible for a trial"
        )


async def _unfinished_subscriptions(
    trial: TrialState | None, current: stripe.checkout.Session | None
) -> dict[str, tuple[str, ...]]:
    if trial is None or trial.consumed_at is not None:
        return {}
    allowed: dict[str, tuple[str, ...]] = {}
    if current and current.status == "open" and _owned_checkout(trial, current):
        sub_id = current.subscription or trial.subscription_id
        if sub_id:
            allowed[str(sub_id)] = ("trialing", "incomplete")
    sessions = await stripe_call(
        stripe.checkout.Session.list_async, customer=trial.customer_id, limit=100
    )
    async for session in stripe_list_items(sessions):
        if session.status != "expired" or not _owned_checkout(
            trial, session, previous=True
        ):
            continue
        if session.subscription:
            allowed[str(session.subscription)] = ("canceled", "incomplete_expired")
    return allowed


def _owned_checkout(
    trial: TrialState, session: stripe.checkout.Session, *, previous: bool = False
) -> bool:
    metadata = session.metadata or {}
    attempt = metadata.get("trial_checkout_attempt", "")
    try:
        attempt_number = int(attempt)
    except ValueError:
        return False
    return bool(
        session.customer == trial.customer_id
        and session.mode == "subscription"
        and metadata.get("user_id") == trial.user_id
        and metadata.get("trial_enrollment_id") == trial.id
        and metadata.get("trial_offer_version") == trial.offer.version
        and str(attempt_number) == attempt
        and (
            0 <= attempt_number <= trial.checkout_attempt
            if previous
            else attempt_number == trial.checkout_attempt
        )
    )
