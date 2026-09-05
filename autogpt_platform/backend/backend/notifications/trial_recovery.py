"""Repair missing notice intents after a saved state change or a missed webhook."""

import logging
from datetime import UTC, datetime, timedelta

import stripe

from backend.data.db_accessors import credit_db
from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial import TrialState
from backend.data.trial_notification_recovery import TrialNoticeCandidate
from backend.notifications.trial import TrialNoticeKind, notify_trial

logger = logging.getLogger(__name__)


async def recover_missing_trial_notices() -> None:
    after_id = ""
    while candidates := await credit_db().get_trial_notice_candidates(after_id):
        for candidate in candidates:
            try:
                await _repair_candidate(candidate)
            except Exception:
                logger.exception("Could not repair notices for trial %s", candidate.id)
        after_id = candidates[-1].id


async def _repair_candidate(candidate: TrialNoticeCandidate) -> None:
    subscription = dict(
        await stripe_call(stripe.Subscription.retrieve_async, candidate.subscription_id)
    )
    if subscription.get("id") != candidate.subscription_id:
        raise ValueError("Trial notice repair returned a different subscription")
    await credit_db().sync_subscription_from_stripe(subscription)
    trial = await credit_db().get_subscription_trial(candidate.user_id)
    if (
        trial is None
        or trial.id != candidate.id
        or trial.subscription_id != candidate.subscription_id
    ):
        raise ValueError("Trial notice repair has no matching enrollment")
    for kind in due_trial_notices(trial, datetime.now(UTC)):
        await notify_trial(subscription, kind)


def due_trial_notices(trial: TrialState, now: datetime) -> list[TrialNoticeKind]:
    if trial.consumed_at is None or trial.ends_at is None:
        return []
    if trial.converted_at is not None:
        return (
            ["converted"]
            if trial.status == "active" and trial.conversion_invoice_id
            else []
        )
    if trial.status in ("past_due", "unpaid", "incomplete"):
        return ["payment_failed"]
    if trial.status in ("canceled", "paused"):
        return ["ended"]
    if trial.status != "trialing" or trial.ends_at <= now:
        return []
    if trial.cancel_at_period_end:
        return ["canceled"]
    if trial.card_verified_at is None:
        return []
    kinds: list[TrialNoticeKind] = ["started"]
    if trial.notification_revision > 0:
        kinds.append("resumed")
    if trial.ends_at <= now + timedelta(days=3):
        kinds.append("ending")
    return kinds
