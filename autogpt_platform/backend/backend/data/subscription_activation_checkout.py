"""Explicit trial conversion, durable confirmation, and read-only payment recovery."""

import logging
from datetime import UTC, datetime, timedelta

import stripe
from prisma.models import User

from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_activation import reconcile_paid_activation
from backend.data.subscription_activation_attempt import (
    get_attempt,
    save_confirmation,
    save_quote,
)
from backend.data.subscription_activation_models import (
    ActivationAttempt,
    ActivationConfirmRequest,
    ActivationNotFound,
    ActivationResponse,
    ActivationUnavailable,
)
from backend.data.subscription_activation_return import activation_return_to
from backend.data.subscription_activation_stripe import (
    BillingSubscription,
    conversion_trial,
    invoice_payment_intent,
    owned_subscription,
    quote_terms,
)
from backend.data.subscription_checkout import subscription_checkout_lock

logger = logging.getLogger(__name__)


async def preview_activation(user_id: str, return_to: str) -> ActivationResponse:
    async with subscription_checkout_lock(user_id):
        existing = await get_attempt(user_id)
        if existing and existing.confirmed_at:
            return await activation_status(existing)
        trial, sub = await conversion_trial(user_id)
        terms = await quote_terms(trial, sub)
        attempt = await save_quote(user_id, sub.id, sub.customer, terms, return_to)
        return attempt.response("confirmation_required")


async def confirm_activation(
    user_id: str,
    attempt_id: str,
    body: ActivationConfirmRequest,
) -> ActivationResponse:
    async with subscription_checkout_lock(user_id):
        attempt = await _require_attempt(user_id, attempt_id)
        if body.terms_token != attempt.terms.token:
            raise ActivationUnavailable("The terms changed. Review a fresh preview.")
        if attempt.confirmed_at is None:
            sub = await owned_subscription(user_id, attempt.subscription_id)
            if sub.status != "trialing":
                return await activation_status(attempt)
            attempt = await _confirm_terms(attempt)
        return await _submit_confirmed(attempt)


async def get_activation(user_id: str, attempt_id: str) -> ActivationResponse:
    return await activation_status(await _require_attempt(user_id, attempt_id))


async def current_activation(user_id: str) -> ActivationResponse:
    attempt = await get_attempt(user_id)
    previous = None
    if attempt:
        previous = await activation_status(attempt)
        if previous.status != "failed":
            return previous
    user = await User.prisma().find_unique_or_raise(where={"id": user_id})
    if not user.stripeCustomerId:
        raise ActivationNotFound("No activation is available")
    subscriptions = await stripe_call(
        stripe.Subscription.list_async,
        customer=user.stripeCustomerId,
        status="all",
        limit=100,
    )
    async for subscription in stripe_list_items(subscriptions):
        if subscription.status in ("active", "incomplete", "past_due"):
            sub = await owned_subscription(user_id, subscription.id)
            response = ActivationResponse(status="processing", retry_after_seconds=3)
            response.return_to = await activation_return_to(sub)
            return await _payment_status(user_id, sub, response)
    if previous:
        return previous
    raise ActivationNotFound("No activation is available")


async def activation_status(attempt: ActivationAttempt) -> ActivationResponse:
    response = attempt.response("processing", retry_after_seconds=3)
    try:
        sub = await owned_subscription(attempt.user_id, attempt.subscription_id)
        if sub.customer != attempt.customer_id:
            raise ActivationUnavailable("Subscription ownership changed")
        if sub.status == "trialing" and attempt.confirmed_at is None:
            return attempt.response("confirmation_required")
        return await _payment_status(attempt.user_id, sub, response)
    except Exception:
        logger.exception(
            "Activation status is pending reconciliation for %s", attempt.id
        )
        return response.model_copy(
            update={
                "status": "processing",
                "retry_after_seconds": 3,
                "hosted_invoice_url": None,
            }
        )


async def _confirm_terms(attempt: ActivationAttempt) -> ActivationAttempt:
    if attempt.terms.expires_at <= datetime.now(UTC):
        raise ActivationUnavailable("The preview expired. Review a fresh preview.")
    trial, sub = await conversion_trial(attempt.user_id)
    if sub.id != attempt.subscription_id:
        raise ActivationUnavailable("The trial subscription changed")
    if not attempt.terms.same_charge_as(await quote_terms(trial, sub)):
        raise ActivationUnavailable("The charge changed. Review a fresh preview.")
    return await save_confirmation(attempt)


async def _submit_confirmed(attempt: ActivationAttempt) -> ActivationResponse:
    # Persist intent before Stripe; even an unknown network outcome retains the
    # same operation. Only this explicitly confirmed POST can initiate payment.
    try:
        sub = await owned_subscription(attempt.user_id, attempt.subscription_id)
        if sub.status == "trialing":
            if not attempt.confirmed_at or (
                datetime.now(UTC) - attempt.confirmed_at >= timedelta(hours=23)
            ):
                return attempt.response("processing", error_code="recovery_required")
            if (
                sub.trial_end is None
                or sub.trial_end <= datetime.now(UTC).timestamp() + 300
            ):
                return attempt.response("processing", retry_after_seconds=3)
            trial, sub = await conversion_trial(attempt.user_id)
            if not attempt.terms.same_charge_as(await quote_terms(trial, sub)):
                return attempt.response("processing", error_code="terms_changed")
            await stripe_call(
                stripe.Subscription.modify_async,
                attempt.subscription_id,
                trial_end="now",
                proration_behavior="none",
                payment_behavior="allow_incomplete",
                metadata={"pro_activation_attempt_id": attempt.id},
                idempotency_key=f"pro-activation:{attempt.id}",
            )
        return await activation_status(attempt)
    except Exception:
        logger.exception("Confirmed activation needs recovery for %s", attempt.id)
        return attempt.response("processing", retry_after_seconds=3)


async def _payment_status(
    user_id: str,
    sub: BillingSubscription,
    response: ActivationResponse,
) -> ActivationResponse:
    invoice = sub.latest_invoice
    if invoice:
        response.invoice_id = invoice.id
    if sub.status in ("canceled", "incomplete_expired") or (
        invoice and invoice.status in ("void", "uncollectible")
    ):
        response.status = "failed"
        response.error_code = "payment_canceled"
        response.retry_after_seconds = None
        return response
    if invoice and invoice.status == "paid":
        if await reconcile_paid_activation(user_id, sub.id):
            response.status = "ready"
            response.retry_after_seconds = None
        return response
    if invoice and invoice.status == "open":
        response.hosted_invoice_url = invoice.hosted_invoice_url
        response.status = "payment_required"
        response.retry_after_seconds = None
        payment_intent_id = await invoice_payment_intent(invoice)
        if payment_intent_id:
            intent = await stripe_call(
                stripe.PaymentIntent.retrieve_async,
                payment_intent_id,
            )
            if intent.customer != sub.customer:
                raise ActivationUnavailable(
                    "Payment ownership could not be established"
                )
            if intent.status == "requires_action":
                response.status = "action_required"
            elif intent.status in ("processing", "succeeded"):
                response.status = "processing"
                response.retry_after_seconds = 3
        return response
    return response


async def _require_attempt(user_id: str, attempt_id: str) -> ActivationAttempt:
    attempt = await get_attempt(user_id, attempt_id)
    if attempt is None:
        raise ActivationNotFound("Activation was not found")
    return attempt
