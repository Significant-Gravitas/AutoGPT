"""Trial notices use accepted terms and current subscription state."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

import stripe

from backend.data.credit import (
    _invoice_subscription_id,
    _invoice_subscription_metadata,
    _track_billing_event,
    sync_subscription_from_stripe,
)
from backend.data.db_accessors import credit_db, user_db
from backend.data.notifications import SubscriptionPlan, TrialUpdateData
from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial import TrialState
from backend.notifications.lifecycle_plan import format_amount
from backend.notifications.queue import queue_trial_delivery

logger = logging.getLogger(__name__)
TRIAL_REMINDER_WINDOW = timedelta(days=3)

TrialNoticeKind = Literal[
    "started", "ending", "canceled", "resumed", "ended", "converted", "payment_failed"
]


async def notify_trial(subscription: dict, kind: TrialNoticeKind) -> bool:
    metadata = subscription.get("metadata") or {}
    if not metadata.get("trial_enrollment_id"):
        return False
    current = dict(
        await stripe_call(stripe.Subscription.retrieve_async, subscription["id"])
    )
    user_id = (current.get("metadata") or {}).get("user_id")
    if not user_id:
        raise ValueError("Trial notification has no user identity")
    trial = await credit_db().get_subscription_trial(user_id)
    if trial is None:
        raise ValueError("Trial notification has no matching enrollment")
    if not _owns_subscription(trial, current):
        raise ValueError("Trial notification ownership does not match the enrollment")
    attempt = (current.get("metadata") or {}).get("trial_checkout_attempt", "")
    if attempt.isdecimal() and int(attempt) < trial.checkout_attempt:
        return True
    if attempt != str(trial.checkout_attempt) or trial.subscription_id != current["id"]:
        raise ValueError(
            "Trial notification does not match the current checkout attempt"
        )
    if trial.consumed_at is None:
        return True
    if trial.converted_at is not None and kind in (
        "ended",
        "canceled",
        "resumed",
        "payment_failed",
    ):
        return False
    if not _notice_applies(trial, kind, current):
        return True
    user = await user_db().get_user_by_id(user_id)
    data = trial_notice_data(trial, kind, user.name or "there")
    claim = trial_notice_key(trial, kind)
    data.notice_key = claim
    receipt = await credit_db().enqueue_trial_notification(
        user_id, trial.id, claim, data
    )
    result = await queue_trial_delivery(receipt.id)
    if result.success:
        await credit_db().mark_trial_notification_queued(receipt.id)
    else:
        logger.warning(
            "Trial notice %s is durable and awaits queue recovery", receipt.id
        )
    if not receipt.created:
        return True
    _track_billing_event(
        f"subscription_trial_{kind}",
        user_id,
        {
            "trial_id": trial.id,
            "trial_offer_version": trial.offer.version,
            "subscription_tier": trial.offer.tier,
            "billing_cycle": trial.offer.billing_cycle,
            "trial_duration_days": trial.offer.duration_days,
        },
    )
    return True


async def on_trial_invoice(invoice: dict, *, paid: bool) -> bool:
    if not _invoice_subscription_metadata(invoice).get("trial_enrollment_id"):
        return False
    sub_id = _invoice_subscription_id(invoice)
    if not sub_id:
        return False
    subscription = dict(await stripe_call(stripe.Subscription.retrieve_async, sub_id))
    if not (subscription.get("metadata") or {}).get("trial_enrollment_id"):
        return False
    await sync_subscription_from_stripe(subscription)
    user_id = (subscription.get("metadata") or {}).get("user_id")
    if not user_id:
        raise ValueError("Trial invoice has no user identity")
    trial = await credit_db().get_subscription_trial(user_id)
    if trial is None:
        raise ValueError("Trial invoice has no matching enrollment")
    if paid and trial.converted_at:
        if invoice.get("id") != trial.conversion_invoice_id:
            return False
        await notify_trial(subscription, "converted")
        return True
    if not paid and trial.converted_at is None:
        await notify_trial(subscription, "payment_failed")
        return True
    return False


async def trial_notice_is_current(user_id: str, data: TrialUpdateData) -> bool:
    return await trial_notice_disposition(user_id, data) == "current"


async def trial_notice_disposition(
    user_id: str, data: TrialUpdateData
) -> Literal["current", "suppressed", "obsolete"]:
    if not data.notice_key:
        return "obsolete"
    trial = await credit_db().get_subscription_trial(user_id)
    if trial is None or trial.subscription_id is None:
        return "obsolete"
    current = dict(
        await stripe_call(stripe.Subscription.retrieve_async, trial.subscription_id)
    )
    if (
        not _owns_subscription(trial, current)
        or current.get("id") != trial.subscription_id
    ):
        return "obsolete"
    await credit_db().sync_subscription_from_stripe(current)
    trial = await credit_db().get_subscription_trial(user_id)
    if (
        trial is None
        or trial.consumed_at is None
        or trial.ends_at is None
        or trial.subscription_id != current.get("id")
        or (current.get("metadata") or {}).get("trial_checkout_attempt")
        != str(trial.checkout_attempt)
    ):
        return "obsolete"
    if data.notice_key != trial_notice_key(trial, data.kind):
        return "obsolete"
    expected = trial_notice_data(trial, data.kind, data.user_name)
    if data.model_copy(update={"notice_key": None}) != expected:
        return "obsolete"
    if trial.status != current.get("status") or trial.cancel_at_period_end != bool(
        current.get("cancel_at_period_end")
    ):
        return "suppressed"
    return "current" if _notice_applies(trial, data.kind, current) else "suppressed"


def _owns_subscription(trial: TrialState, current: dict) -> bool:
    metadata = current.get("metadata") or {}
    return (
        trial.customer_id == current.get("customer")
        and trial.id == metadata.get("trial_enrollment_id")
        and trial.user_id == metadata.get("user_id")
    )


async def on_trial_subscription_updated(subscription: dict, previous: dict) -> bool:
    if not (subscription.get("metadata") or {}).get("trial_enrollment_id"):
        return False
    user_id = (subscription.get("metadata") or {}).get("user_id")
    if not user_id:
        raise ValueError("Trial update has no user identity")
    trial = await credit_db().get_subscription_trial(user_id)
    if trial is None or trial.converted_at is not None:
        return False
    if "cancel_at_period_end" in previous:
        kind = "canceled" if trial.cancel_at_period_end else "resumed"
        await notify_trial(subscription, kind)
    return True


def trial_notice_key(trial: TrialState, kind: TrialNoticeKind) -> str:
    key = f"trial:{trial.id}:{kind}"
    if kind in ("canceled", "resumed"):
        return f"{key}:{trial.notification_revision}"
    if kind == "ending":
        if trial.ends_at is None:
            raise ValueError("Trial reminder requires an end date")
        return f"{key}:{int(trial.ends_at.timestamp())}"
    if kind == "converted":
        if not trial.conversion_invoice_id:
            raise ValueError("Trial conversion notice requires its invoice")
        return f"{key}:{trial.conversion_invoice_id}"
    return key


def trial_notice_data(
    trial: TrialState, kind: TrialNoticeKind, name: str
) -> TrialUpdateData:
    if trial.ends_at is None:
        raise ValueError("Cannot send a trial notice without the accepted end date")
    offer = trial.offer
    cycle_noun = "month" if offer.billing_cycle == "monthly" else "year"
    return TrialUpdateData(
        user_name=name,
        kind=kind,
        plan=SubscriptionPlan(
            name=offer.tier.title(),
            cycle=offer.billing_cycle,
            cycle_noun=cycle_noun,
            label=f"{offer.tier.title()} — {offer.billing_cycle}",
            price_display=f"{format_amount(offer.unit_amount, offer.currency)} {offer.currency.upper()} / {cycle_noun}",
        ),
        ends_label=trial.ends_at.astimezone(UTC).strftime("%d %b %Y at %H:%M UTC"),
        onboarding_credit_amount=offer.onboarding_credit_amount,
        offer_version=offer.version,
    )


def _notice_applies(trial: TrialState, kind: TrialNoticeKind, current: dict) -> bool:
    status = current.get("status")
    trial_end = current.get("trial_end") or 0
    now = datetime.now(UTC).timestamp()
    in_trial = status == "trialing" and trial_end > now
    if kind == "ending" and trial_end > now + TRIAL_REMINDER_WINDOW.total_seconds():
        return False
    if kind in ("started", "ending", "resumed"):
        return (
            in_trial
            and trial.card_verified_at is not None
            and not current.get("cancel_at_period_end")
        )
    if kind == "canceled":
        return in_trial and bool(current.get("cancel_at_period_end"))
    if kind == "converted":
        return status == "active" and trial.converted_at is not None
    if kind == "payment_failed":
        return (
            status in ("past_due", "unpaid", "incomplete")
            and trial.converted_at is None
        )
    return status in ("canceled", "unpaid", "paused") and trial.converted_at is None
