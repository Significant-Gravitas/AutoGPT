"""The account and billing emails, triggered from Stripe.

Each handler listens for exactly one signal and does one thing. The rules that
keep them from misfiring — first subscription only, first failed charge only,
the false→true flip rather than every subscription update — live here rather
than in the webhook router, so the webhook stays a dispatcher.

There is deliberately no trial handler: the platform does not offer a trial, so
`customer.subscription.trial_will_end` is not listened for.
"""

import logging
from datetime import datetime, timezone

import stripe
from prisma.enums import NotificationType

from backend.data.notifications import (
    AudienceAction,
    AudienceEventModel,
    NotificationEventModel,
    PaymentFailedData,
    PaymentFinalNoticeData,
    SubscriptionCancelledData,
    SubscriptionEndedData,
    SubscriptionResumedData,
    SubscriptionWelcomeData,
)
from backend.data.user import BillingEmailRecipient
from backend.notifications.dedupe import claim_once, release_claim
from backend.notifications.lifecycle_plan import (
    card_from_invoice,
    format_amount,
    format_date,
    plan_from_invoice,
    plan_from_subscription,
)
from backend.notifications.queue import queue_audience_change, queue_notification_async
from backend.util.clients import get_database_manager_async_client
from backend.util.logging import TruncatedLogger
from backend.util.settings import Settings

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[Lifecycle]")
settings = Settings()


def _db():
    """These handlers run in two processes — the REST API on a Stripe webhook,
    and the notification service when the welcome is picked up off the work
    queue. Only one of them owns a Prisma connection, so both go via the RPC."""
    return get_database_manager_async_client()


async def _publish(event, claim_key: str | None = None) -> None:
    """Publish a billing email, or give its claim back and fail loudly.

    Every one of these is claimed before it is published so a Stripe replay
    cannot double-send. Ignoring a failed publish turns that safety into a
    liability: the key is spent, the webhook still returns 200, Stripe never
    retries, and the customer simply never hears that their payment failed.

    Raising instead lets the webhook dispatcher release its own event claim and
    return 5xx, and releasing this key first means the retry actually gets
    through rather than being deduped.
    """
    result = await queue_notification_async(event)
    if result.success:
        return
    if claim_key:
        await release_claim(claim_key)
    raise RuntimeError(f"Could not queue {event.type.value}: {result.message}")


async def send_welcome_for_session(session_id: str) -> None:
    """Re-read the checkout from Stripe and send the welcome.

    Called from the work queue rather than the webhook, so the Stripe round
    trip and the email are covered by retry-with-backoff and a dead-letter
    queue. Re-reading rather than carrying the payload means a message that
    waited acts on current state.

    Raises on failure so the consumer retries; `on_checkout_completed` is
    idempotent via the `welcomeEmailSentAt` claim.
    """
    session = dict(await stripe.checkout.Session.retrieve_async(session_id))
    subscription_id = session.get("subscription")
    if not subscription_id:
        logger.info(f"Checkout {session_id} has no subscription; nothing to welcome")
        return
    subscription = dict(await stripe.Subscription.retrieve_async(str(subscription_id)))
    await on_checkout_completed(session, subscription)


async def on_checkout_completed(session: dict, subscription: dict) -> None:
    """First subscription → welcome email and the onboarding tour. A returning
    customer is not greeted like a stranger: they go straight into the
    changelog audience instead."""
    user = await _user_for(session.get("customer"))
    if user is None:
        return

    if user.welcome_email_sent_at is not None:
        await queue_audience_change(
            AudienceEventModel(
                action=AudienceAction.ADD_CHANGELOG, email=user.email, user_id=user.id
            )
        )
        return

    # Claim before queueing: Stripe retries webhooks, and two welcomes is worse
    # than none.
    if not await _claim_welcome(user):
        return

    plan = await plan_from_subscription(subscription)
    try:
        await _publish(
            NotificationEventModel[SubscriptionWelcomeData](
                user_id=user.id,
                type=NotificationType.SUBSCRIPTION_WELCOME,
                data=SubscriptionWelcomeData(
                    user_name=_greeting_name(user),
                    plan=plan,
                    renews_label=format_date(subscription.get("current_period_end")),
                ),
            )
        )
    except Exception:
        # The claim is a database flag, not a Redis key with a TTL, so a failed
        # publish would otherwise mark this customer as welcomed forever and
        # every retry would take the returning-customer branch instead.
        await _release_welcome(user)
        raise

    # Must not propagate: the welcome is already out and the claim is durable,
    # so a Stripe retry would take the returning-customer branch and enrol them
    # in the changelog instead of the tour. Report it rather than fail.
    enrolled = await queue_audience_change(
        AudienceEventModel(
            action=AudienceAction.ENROLL_TOUR, email=user.email, user_id=user.id
        )
    )
    if not enrolled.success:
        logger.error(
            f"Welcomed user {user.id} but could not queue the onboarding tour: "
            f"{enrolled.message}"
        )


async def on_payment_failed(invoice: dict) -> None:
    """First failed charge → a calm heads-up. Retries exhausted → the final
    notice. Nothing at all during Stripe's automatic retries, which re-fire
    this same event."""
    user = await _user_for(invoice.get("customer"))
    if user is None:
        return

    invoice_id = str(invoice.get("id") or "")
    plan = await plan_from_invoice(invoice)
    amount = format_amount(invoice.get("amount_due"), invoice.get("currency", "usd"))

    if invoice.get("next_payment_attempt"):
        if int(invoice.get("attempt_count") or 0) != 1:
            # Automatic retries stay silent: four "payment failed" emails in
            # two weeks reads as a billing crisis.
            return
        claim_key = f"payment_failed:{invoice_id}"
        if not await claim_once(claim_key):
            return
        await _publish(
            NotificationEventModel[PaymentFailedData](
                user_id=user.id,
                type=NotificationType.PAYMENT_FAILED,
                data=PaymentFailedData(
                    user_name=_greeting_name(user),
                    plan=plan,
                    amount_display=amount,
                    card=card_from_invoice(invoice),
                    next_retry_label=format_date(invoice.get("next_payment_attempt")),
                ),
            ),
            claim_key,
        )
        return

    claim_key = f"final_notice:{invoice_id}"
    if not await claim_once(claim_key):
        return
    await _publish(
        NotificationEventModel[PaymentFinalNoticeData](
            user_id=user.id,
            type=NotificationType.PAYMENT_FINAL_NOTICE,
            data=PaymentFinalNoticeData(
                user_name=_greeting_name(user),
                plan=plan,
                amount_display=amount,
                # Matches Stripe's automatic-collection configuration; see the
                # deployment notes for the setting this must agree with.
                pauses_label=format_date(invoice.get("period_end")),
            ),
        ),
        claim_key,
    )


async def on_subscription_updated(subscription: dict, previous: dict) -> None:
    """Only the cancel_at_period_end flip matters here; this event fires for
    many unrelated changes."""
    if "cancel_at_period_end" not in previous:
        return
    user = await _user_for(subscription.get("customer"))
    if user is None:
        return

    was_cancelling = bool(previous.get("cancel_at_period_end"))
    is_cancelling = bool(subscription.get("cancel_at_period_end"))
    if was_cancelling == is_cancelling:
        return

    sub_id = str(subscription.get("id") or "")
    period_end = subscription.get("current_period_end")
    plan = await plan_from_subscription(subscription)

    # The cancellation episode, not the billing period: cancel → resume →
    # cancel inside one period is two real cancellations. Stripe stamps
    # `canceled_at` on request and clears it on resume, so it names the episode.
    episode = (
        subscription.get("canceled_at")
        if is_cancelling
        else previous.get("canceled_at")
    ) or period_end

    if is_cancelling:
        claim_key = f"cancelled:{sub_id}:{episode}"
        if not await claim_once(claim_key):
            return
        await _publish(
            NotificationEventModel[SubscriptionCancelledData](
                user_id=user.id,
                type=NotificationType.SUBSCRIPTION_CANCELLED,
                data=SubscriptionCancelledData(
                    user_name=_greeting_name(user),
                    plan=plan,
                    access_until_label=format_date(period_end),
                ),
            ),
            claim_key,
        )
        return

    claim_key = f"resumed:{sub_id}:{episode}"
    if not await claim_once(claim_key):
        return
    await _publish(
        NotificationEventModel[SubscriptionResumedData](
            user_id=user.id,
            type=NotificationType.SUBSCRIPTION_RESUMED,
            data=SubscriptionResumedData(
                user_name=_greeting_name(user),
                plan=plan,
                renews_label=format_date(period_end),
            ),
        ),
        claim_key,
    )


async def on_subscription_deleted(subscription: dict) -> None:
    """Two roads lead here — a cancellation reaching period end, and dunning
    exhaustion — so the copy branches on which one the customer took."""
    user = await _user_for(subscription.get("customer"))
    if user is None:
        return

    sub_id = str(subscription.get("id") or "")
    claim_key = f"ended:{sub_id}"
    if not await claim_once(claim_key):
        return

    reason = (subscription.get("cancellation_details") or {}).get("reason")
    plan = await plan_from_subscription(subscription)
    await _publish(
        NotificationEventModel[SubscriptionEndedData](
            user_id=user.id,
            type=NotificationType.SUBSCRIPTION_ENDED,
            data=SubscriptionEndedData(
                user_name=_greeting_name(user),
                plan=plan,
                ended_label=format_date(subscription.get("ended_at")),
                due_to_payment=reason == "payment_failed",
            ),
        ),
        claim_key,
    )
    # Churned users get win-back only, never the monthly update.
    await queue_audience_change(
        AudienceEventModel(
            action=AudienceAction.REMOVE_CHANGELOG, email=user.email, user_id=user.id
        )
    )


async def _user_for(customer_id: object) -> BillingEmailRecipient | None:
    """Skip deleted or unknown accounts rather than emailing into the void."""
    if not customer_id or not isinstance(customer_id, str):
        return None
    user = await _db().get_billing_email_recipient(customer_id)
    if user is None:
        logger.warning(f"No user for Stripe customer {customer_id}; skipping email")
    return user


async def _claim_welcome(user: BillingEmailRecipient) -> bool:
    """Set the "welcome sent" flag, and only send if this call is the one that
    set it."""
    return await _db().claim_welcome_email(user.id)


async def _release_welcome(user: BillingEmailRecipient) -> None:
    """Give the welcome claim back so a retry can actually send."""
    await _db().release_welcome_email(user.id)


def _greeting_name(user: BillingEmailRecipient) -> str:
    if user.name and user.name.strip():
        return user.name.strip().split()[0]
    return user.email.split("@")[0]


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)
