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

from prisma.enums import NotificationType
from prisma.models import User

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
from backend.notifications.lifecycle_dedupe import claim_once
from backend.notifications.lifecycle_plan import (
    card_from_invoice,
    format_amount,
    format_date,
    plan_from_invoice,
    plan_from_subscription,
)
from backend.notifications.queue import queue_audience_change, queue_notification_async
from backend.util.logging import TruncatedLogger
from backend.util.settings import Settings

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[Lifecycle]")
settings = Settings()


async def on_checkout_completed(session: dict, subscription: dict) -> None:
    """First subscription → welcome email and the onboarding tour. A returning
    customer is not greeted like a stranger: they go straight into the
    changelog audience instead."""
    user = await _user_for(session.get("customer"))
    if user is None:
        return

    if user.welcomeEmailSentAt is not None:
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
    await queue_notification_async(
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
    await queue_audience_change(
        AudienceEventModel(
            action=AudienceAction.ENROLL_TOUR, email=user.email, user_id=user.id
        )
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
        if not await claim_once(f"payment_failed:{invoice_id}"):
            return
        await queue_notification_async(
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
            )
        )
        return

    if not await claim_once(f"final_notice:{invoice_id}"):
        return
    await queue_notification_async(
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
        )
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

    if is_cancelling:
        if not await claim_once(f"cancelled:{sub_id}:{period_end}"):
            return
        await queue_notification_async(
            NotificationEventModel[SubscriptionCancelledData](
                user_id=user.id,
                type=NotificationType.SUBSCRIPTION_CANCELLED,
                data=SubscriptionCancelledData(
                    user_name=_greeting_name(user),
                    plan=plan,
                    access_until_label=format_date(period_end),
                ),
            )
        )
        return

    if not await claim_once(f"resumed:{sub_id}:{period_end}"):
        return
    await queue_notification_async(
        NotificationEventModel[SubscriptionResumedData](
            user_id=user.id,
            type=NotificationType.SUBSCRIPTION_RESUMED,
            data=SubscriptionResumedData(
                user_name=_greeting_name(user),
                plan=plan,
                renews_label=format_date(period_end),
            ),
        )
    )


async def on_subscription_deleted(subscription: dict) -> None:
    """Two roads lead here — a cancellation reaching period end, and dunning
    exhaustion — so the copy branches on which one the customer took."""
    user = await _user_for(subscription.get("customer"))
    if user is None:
        return

    sub_id = str(subscription.get("id") or "")
    if not await claim_once(f"ended:{sub_id}"):
        return

    reason = (subscription.get("cancellation_details") or {}).get("reason")
    plan = await plan_from_subscription(subscription)
    await queue_notification_async(
        NotificationEventModel[SubscriptionEndedData](
            user_id=user.id,
            type=NotificationType.SUBSCRIPTION_ENDED,
            data=SubscriptionEndedData(
                user_name=_greeting_name(user),
                plan=plan,
                ended_label=format_date(subscription.get("ended_at")),
                due_to_payment=reason == "payment_failed",
            ),
        )
    )
    # Churned users get win-back only, never the monthly update.
    await queue_audience_change(
        AudienceEventModel(
            action=AudienceAction.REMOVE_CHANGELOG, email=user.email, user_id=user.id
        )
    )


async def _user_for(customer_id: object) -> User | None:
    """Skip deleted or unknown accounts rather than emailing into the void."""
    if not customer_id or not isinstance(customer_id, str):
        return None
    user = await User.prisma().find_first(where={"stripeCustomerId": customer_id})
    if user is None:
        logger.warning("No user for Stripe customer %s; skipping email", customer_id)
    return user


async def _claim_welcome(user: User) -> bool:
    """Set the "welcome sent" flag, and only send if this call is the one that
    set it."""
    claimed = await User.prisma().update_many(
        where={"id": user.id, "welcomeEmailSentAt": None},
        data={"welcomeEmailSentAt": _now()},
    )
    return claimed > 0


def _greeting_name(user: User) -> str:
    if user.name and user.name.strip():
        return user.name.strip().split()[0]
    return user.email.split("@")[0]


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)
