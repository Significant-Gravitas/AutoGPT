"""Trial notices use accepted terms and current subscription state."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

import stripe
from prisma.enums import NotificationType

from backend.data.credit import (
    _invoice_subscription_id,
    _invoice_subscription_metadata,
    _track_billing_event,
    sync_subscription_from_stripe,
)
from backend.data.db_accessors import credit_db, user_db
from backend.data.model import User
from backend.data.notifications import (
    NotificationEventModel,
    SubscriptionPlan,
    TrialUpdateData,
)
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_trial import TrialState
from backend.notifications.dedupe import claim_once, is_claimed, release_claim
from backend.notifications.lifecycle_plan import currency_symbol, format_amount
from backend.notifications.queue import queue_notification_async
from backend.notifications.recipient import first_name
from backend.notifications.trial_audience import (
    join_paying_audience,
    leave_trial_group,
    queue_trial_audience_change,
)
from backend.notifications.trial_plan import notice_plan
from backend.util.posthog_events import PostHogEvent

logger = logging.getLogger(__name__)
# Stripe sends customer.subscription.trial_will_end three days before
# trial_end. The extra day absorbs an event created or handled a little early;
# a delayed or replayed event for a trial since extended further still drops.
TRIAL_REMINDER_WINDOW = timedelta(days=4)
# The last-call notice says access ends tomorrow, so it goes out only while
# the trial has between these two lead times left. The hourly pass opens the
# window; a notice still unsent at the near edge is dropped, not sent late.
LAST_CALL_WINDOW = timedelta(hours=25)
# The not-renewing three-day reminder says "ends in 3 days", so it goes out
# only within half a day of that. A trial_will_end handled later (after an
# outage, say) is dropped and the last call covers the trial instead.
EXPIRING_MIN_LEAD = timedelta(days=2, hours=12)
EXPIRING_MAX_LEAD = timedelta(days=3, hours=12)
LAST_CALL_MIN_LEAD = timedelta(hours=18)
# Someone who canceled this recently has just had the cancellation notice,
# which gives the same end date and the same way back.
RECENT_CANCEL = timedelta(days=1)

# "expiring" is the three-day reminder for a trial set not to renew, and
# "last_call" the one-day reminder; "ending" is the three-day reminder for a
# trial that will convert.
TrialNoticeKind = Literal[
    "started",
    "ending",
    "expiring",
    "last_call",
    "canceled",
    "resumed",
    "ended",
    "converted",
    "payment_failed",
]
# Notices that only describe a trial set not to renew.
CANCEL_PENDING_KINDS = ("canceled", "expiring", "last_call")
# The reminders for a trial set not to renew.
NOT_RENEWING_KINDS = ("expiring", "last_call")

TRIAL_NOTICE_EVENTS: dict[TrialNoticeKind, PostHogEvent] = {
    "started": PostHogEvent.TRIAL_STARTED,
    "ending": PostHogEvent.TRIAL_ENDING,
    "expiring": PostHogEvent.TRIAL_EXPIRING,
    "last_call": PostHogEvent.TRIAL_LAST_CALL,
    "canceled": PostHogEvent.TRIAL_CANCELED,
    "resumed": PostHogEvent.TRIAL_RESUMED,
    "ended": PostHogEvent.TRIAL_ENDED,
    "converted": PostHogEvent.TRIAL_CONVERTED,
    "payment_failed": PostHogEvent.PAYMENT_FAILED,
}


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
    if kind == "ending" and current.get("cancel_at_period_end"):
        # Stripe's trial_will_end also fires for a trial set not to renew.
        kind = "expiring"
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
        "expiring",
        "last_call",
        "resumed",
        "payment_failed",
    ):
        return False
    if skip := _notice_skip_reason(trial, kind, current):
        logger.info(f"Dropped trial {kind} notice for user {user_id}: {skip}")
        return True
    queue = _queue_not_renewing if kind in NOT_RENEWING_KINDS else _queue_notice
    user = await queue(trial, kind, current, user_id)
    if user is None:
        return True
    if kind == "converted":
        await join_paying_audience(user)
    plan = notice_plan(trial, kind, current)
    _track_billing_event(
        TRIAL_NOTICE_EVENTS[kind],
        user_id,
        {
            "trial_id": trial.id,
            "trial_offer_version": trial.offer.version,
            "subscription_tier": plan.tier,
            "billing_cycle": plan.cycle,
            "trial_duration_days": trial.offer.duration_days,
        },
    )
    return True


async def send_trial_last_calls() -> int:
    """Queue the last-call notice for each trial set not to renew whose end
    is inside the last-call window. Runs hourly; each notice is claimed once
    per trial end, so a later pass skips it. Returns how many trials were
    checked."""
    now = datetime.now(UTC)
    trials = await credit_db().list_trials_ending_unrenewed(
        now + LAST_CALL_MIN_LEAD, now + LAST_CALL_WINDOW
    )
    for trial in trials:
        # Most due trials were notified on an earlier run: skip their Stripe
        # retrieve. notify_trial's own claim still decides.
        if await is_claimed(trial_notice_key(trial, "last_call")):
            continue
        subscription = {
            "id": trial.subscription_id,
            "metadata": {"trial_enrollment_id": trial.id},
        }
        try:
            await notify_trial(subscription, "last_call")
        except Exception:
            # One broken trial must not hold back everyone else's notice;
            # the next hourly pass tries it again.
            logger.exception(f"Trial last call for user {trial.user_id} failed")
    return len(trials)


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
    if data.kind in NOT_RENEWING_KINDS and await _another_plan_outlives(trial, current):
        return "obsolete"
    expected = trial_notice_data(trial, data.kind, data.user_name, current)
    if not _notice_matches_current_terms(data, expected, trial, current):
        return "obsolete"
    if trial.status != current.get("status") or trial.cancel_at_period_end != bool(
        current.get("cancel_at_period_end")
    ):
        return "suppressed"
    return "current" if _notice_applies(trial, data.kind, current) else "suppressed"


def _notice_matches_current_terms(
    data: TrialUpdateData, expected: TrialUpdateData, trial: TrialState, current: dict
) -> bool:
    # Transitional shim: remove once old-format queued notices drain, per the
    # rollout plan in docs/engineering/transactional-email-design.md.
    terms = notice_plan(trial, data.kind, current)
    plan = data.plan.model_copy()
    legacy_label = f"{expected.plan.name} — {expected.plan.cycle}"
    legacy_amount = format_amount(terms.unit_amount, terms.currency)
    legacy_price = (
        f"{legacy_amount} {terms.currency.upper()} / {expected.plan.cycle_noun}"
    )
    # Accept only the old presentation of these same terms, so a deploy does
    # not discard queued notices or conceal a changed amount, plan or end date.
    if plan.label == legacy_label:
        plan.label = expected.plan.label
    if plan.price_display == legacy_price:
        plan.price_display = expected.plan.price_display
    return (
        data.model_copy(
            update={
                "plan": plan,
                "notice_key": None,
                "started_label": (
                    expected.started_label
                    if data.started_label is None
                    else data.started_label
                ),
            }
        )
        == expected
    )


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
    if trial is None:
        return False
    if trial.converted_at is not None:
        # A payload still trialing is a trial-era cancel or resume handled
        # after the conversion. Subscribe now on a cancel-pending trial ends
        # the trial and clears the cancellation in one update. The trial's own
        # notices cover both; only later flips are paid news.
        if subscription.get("status") == "trialing":
            return True
        was_trialing = previous.get("status") == "trialing"
        return was_trialing and "cancel_at_period_end" in previous
    if "cancel_at_period_end" in previous:
        kind = "canceled" if trial.cancel_at_period_end else "resumed"
        await notify_trial(subscription, kind)
    return True


def trial_notice_key(trial: TrialState, kind: TrialNoticeKind) -> str:
    key = f"trial:{trial.id}:{kind}"
    if kind in ("canceled", "resumed"):
        return f"{key}:{trial.notification_revision}"
    if kind in ("ending", "expiring", "last_call"):
        if trial.ends_at is None:
            raise ValueError("Trial reminder requires an end date")
        return f"{key}:{int(trial.ends_at.timestamp())}"
    if kind == "converted":
        if not trial.conversion_invoice_id:
            raise ValueError("Trial conversion notice requires its invoice")
        return f"{key}:{trial.conversion_invoice_id}"
    return key


def trial_notice_data(
    trial: TrialState,
    kind: TrialNoticeKind,
    name: str,
    current: dict | None = None,
) -> TrialUpdateData:
    if trial.ends_at is None:
        raise ValueError("Cannot send a trial notice without the accepted end date")
    plan = notice_plan(trial, kind, current)
    title = plan.tier.title()
    cycle_noun = "month" if plan.cycle == "monthly" else "year"
    amount = format_amount(plan.unit_amount, plan.currency)
    if plan.unit_amount is not None and not currency_symbol(plan.currency):
        amount = f"{amount} {plan.currency.upper()}"
    return TrialUpdateData(
        user_name=name,
        kind=kind,
        plan=SubscriptionPlan(
            name=title,
            cycle=plan.cycle,
            cycle_noun=cycle_noun,
            label=f"{title} · {plan.cycle}",
            price_display=f"{amount} / {cycle_noun}",
        ),
        ends_label=trial.ends_at.astimezone(UTC).strftime("%d %b %Y at %H:%M UTC"),
        started_label=(
            trial.converted_at.astimezone(UTC).strftime("%d %b %Y at %H:%M UTC")
            if kind == "converted" and trial.converted_at
            else None
        ),
        onboarding_credit_amount=trial.offer.onboarding_credit_amount,
        offer_version=trial.offer.version,
    )


async def _queue_not_renewing(
    trial: TrialState, kind: TrialNoticeKind, current: dict, user_id: str
) -> User | None:
    """Queue an expiring or last_call notice; None when it was not queued.

    Claimed before the plan listing: the hourly pass sees a due trial on
    several runs, and one already notified should cost no more Stripe calls.
    For the same reason the last call's claim fails closed: open, a Redis
    outage would send it on every run."""
    claim = trial_notice_key(trial, kind)
    if not await claim_once(claim, fail_open=kind != "last_call"):
        return None
    try:
        if await _another_plan_outlives(trial, current):
            # A newer plan keeps their access past the trial's end, so the
            # notice would be wrong.
            logger.info(f"Dropped trial {kind} notice for user {user_id}: another plan")
            await release_claim(claim)
            return None
        user = await user_db().get_user_by_id(user_id)
        await _publish_notice(trial, kind, current, user, claim)
    except Exception:
        await release_claim(claim)
        raise
    return user


async def _queue_notice(
    trial: TrialState, kind: TrialNoticeKind, current: dict, user_id: str
) -> User | None:
    """Queue any other trial notice; None when it was not queued."""
    user = await user_db().get_user_by_id(user_id)
    if kind == "ended" and await _another_plan_is_live(trial, current):
        await leave_trial_group(user)
        return None
    claim = trial_notice_key(trial, kind)
    if not await claim_once(claim):
        return None
    try:
        await _publish_notice(trial, kind, current, user, claim)
    except Exception:
        await release_claim(claim)
        raise
    return user


async def _publish_notice(
    trial: TrialState, kind: TrialNoticeKind, current: dict, user: User, claim: str
) -> None:
    """Queue the notice and its trial-group change. Raises when either
    cannot be queued, so the caller gives the claim back."""
    data = trial_notice_data(
        trial, kind, first_name(user.name, user.email) or "there", current
    )
    data.notice_key = claim
    await queue_trial_audience_change(kind, user)
    result = await queue_notification_async(
        NotificationEventModel[TrialUpdateData](
            user_id=trial.user_id, type=NotificationType.TRIAL_UPDATE, data=data
        )
    )
    if not result.success:
        raise RuntimeError(f"Could not queue trial notice: {result.message}")


def _notice_applies(trial: TrialState, kind: TrialNoticeKind, current: dict) -> bool:
    return _notice_skip_reason(trial, kind, current) is None


def _notice_skip_reason(
    trial: TrialState, kind: TrialNoticeKind, current: dict
) -> str | None:
    """Why the notice no longer describes the subscription, or None."""
    status = current.get("status") or "unknown"
    if kind == "converted":
        if status == "active" and trial.converted_at is not None:
            return None
        return f"subscription is {status}, not converted"
    if kind == "payment_failed":
        if status in ("past_due", "unpaid", "incomplete") and not trial.converted_at:
            return None
        return f"subscription is {status}, not a failed conversion"
    if kind == "ended":
        if status in ("canceled", "unpaid", "paused") and not trial.converted_at:
            return None
        return f"subscription is {status}, not an ended trial"
    return _running_trial_skip_reason(trial, kind, current)


def _running_trial_skip_reason(
    trial: TrialState, kind: TrialNoticeKind, current: dict
) -> str | None:
    trial_end = current.get("trial_end") or 0
    now = datetime.now(UTC).timestamp()
    if current.get("status") != "trialing" or trial_end <= now:
        return "trial is no longer running"
    if (
        kind in ("ending", "expiring")
        and trial_end > now + TRIAL_REMINDER_WINDOW.total_seconds()
    ):
        return "trial ends after the reminder window"
    if kind == "expiring":
        if trial_end > now + EXPIRING_MAX_LEAD.total_seconds():
            return "trial ends after the reminder window"
        if trial_end < now + EXPIRING_MIN_LEAD.total_seconds():
            return "too little of the trial is left for a three-day reminder"
    if kind == "last_call":
        if trial_end > now + LAST_CALL_WINDOW.total_seconds():
            return "trial ends after the last-call window"
        if trial_end < now + LAST_CALL_MIN_LEAD.total_seconds():
            return "too little of the trial is left for a last call"
    # A canceled notice promises access until trial_end, which only a verified
    # card gives.
    if trial.card_verified_at is None:
        return "card is not verified"
    cancel_pending = bool(current.get("cancel_at_period_end"))
    if cancel_pending and kind not in CANCEL_PENDING_KINDS:
        return "cancellation is scheduled"
    if not cancel_pending and kind in CANCEL_PENDING_KINDS:
        return "cancellation is not scheduled"
    # Stripe sets canceled_at when the cancellation is requested.
    canceled_at = current.get("canceled_at") or 0
    if kind in ("expiring", "last_call") and (
        canceled_at > now - RECENT_CANCEL.total_seconds()
    ):
        return "canceled within the last day"
    return None


async def _another_plan_is_live(trial: TrialState, current: dict) -> bool:
    """A trial ended by buying another plan is not news, and its trial_canceled
    status must not overwrite the new plan's. The person still leaves the
    trial group."""
    return await _another_plan(trial, current, until=None)


async def _another_plan_outlives(trial: TrialState, current: dict) -> bool:
    """Another live plan keeps access past the trial's end, so a notice that
    says access ends then would be wrong. A plan set to end first does not."""
    return await _another_plan(trial, current, until=current.get("trial_end") or 0)


async def _another_plan(trial: TrialState, current: dict, until: int | None) -> bool:
    for status in ("active", "trialing"):
        page = await stripe_call(
            stripe.Subscription.list_async,
            customer=trial.customer_id,
            status=status,
            limit=100,
        )
        async for other in stripe_list_items(page):
            if other.id == current.get("id"):
                continue
            # Stripe sets cancel_at on a plan scheduled to end.
            if until is None or other.cancel_at is None or other.cancel_at > until:
                return True
    return False
