"""The open invoice a trial's failed first charge leaves behind."""

import logging

import stripe
from pydantic import BaseModel, ValidationError

from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_payment import SubscriptionSnapshot

logger = logging.getLogger(__name__)

PAYMENT_DUE_STATUSES = frozenset({"past_due", "unpaid"})


class TrialPaymentDue(BaseModel):
    amount_due: int
    currency: str
    payment_url: str


async def get_trial_payment_due(trial: TrialState) -> TrialPaymentDue | None:
    """Paying this invoice on Stripe's hosted page converts the trial through
    the usual invoice-paid path. Checkout refuses a second plan while it is
    open, so it is the one way back to the plan besides Stripe's own retries.
    """
    if (
        trial.converted_at is not None
        or trial.subscription_id is None
        or trial.status not in PAYMENT_DUE_STATUSES
    ):
        return None
    try:
        snapshot = SubscriptionSnapshot.model_validate(
            await stripe_call(
                stripe.Subscription.retrieve_async,
                trial.subscription_id,
                expand=["latest_invoice"],
            )
        )
    except (stripe.StripeError, ValidationError):
        logger.warning(
            f"Could not read the open invoice for trial {trial.id}", exc_info=True
        )
        return None
    invoice = snapshot.latest_invoice
    if (
        snapshot.customer != trial.customer_id
        or snapshot.metadata.get("trial_enrollment_id") != trial.id
        or invoice is None
        or invoice.status != "open"
        or invoice.customer != trial.customer_id
        or invoice.amount_due <= 0
        or not invoice.currency
        or not invoice.hosted_invoice_url
    ):
        return None
    return TrialPaymentDue(
        amount_due=invoice.amount_due,
        currency=invoice.currency,
        payment_url=invoice.hosted_invoice_url,
    )
