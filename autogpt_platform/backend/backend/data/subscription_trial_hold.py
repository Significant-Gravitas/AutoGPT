"""A small authorization on the trial's card when the trial starts, released
at once and never captured: a card that cannot hold it cannot pay the plan at
the trial's end either."""

import logging

import stripe

from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial import TrialState

logger = logging.getLogger(__name__)

# In the offer currency's minor unit: $1 on a USD offer.
CARD_HOLD_AMOUNT = 100
_PURPOSE_KEY = "autogpt_purpose"
_PURPOSE = "trial_card_hold"


async def card_holds_funds(trial: TrialState, payment_method_id: str) -> bool:
    """Whether the card authorized the hold.

    A bank asking to authenticate the off-session hold says nothing about
    funds, and the card was authenticated in Checkout moments ago, so that
    passes too.
    """
    try:
        intent = await stripe_call(
            stripe.PaymentIntent.create_async,
            amount=CARD_HOLD_AMOUNT,
            currency=trial.offer.currency,
            customer=trial.customer_id,
            payment_method=payment_method_id,
            payment_method_types=["card"],
            capture_method="manual",
            confirm=True,
            off_session=True,
            description="Card check for your AutoGPT trial",
            metadata={
                _PURPOSE_KEY: _PURPOSE,
                "trial_enrollment_id": trial.id,
                "user_id": trial.user_id,
            },
            idempotency_key=(
                f"trial-card-hold-{trial.id}-{trial.checkout_attempt}"
                f"-{payment_method_id}"
            ),
        )
    except stripe.CardError as exc:
        decline_code = exc.error.decline_code if exc.error else None
        if "authentication_required" in (exc.code, decline_code):
            logger.info(f"Trial {trial.id} card hold needs authentication; passing")
            return True
        logger.info(f"Trial {trial.id} card hold declined: {exc.code} {decline_code}")
        return False
    if intent.status == "requires_capture":
        await stripe_call(
            stripe.PaymentIntent.cancel_async,
            intent.id,
            idempotency_key=f"trial-card-hold-release-{intent.id}",
        )
    return True


def is_trial_card_hold(payment_intent: stripe.PaymentIntent) -> bool:
    return (payment_intent.get("metadata") or {}).get(_PURPOSE_KEY) == _PURPOSE
