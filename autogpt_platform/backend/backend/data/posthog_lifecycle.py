"""Subscription status and lifecycle dates for the PostHog person (SECRT-2778).

This module is the mapping only: pure, no I/O. ``posthog_lifecycle_sync``
reads the inputs and sends the result.

The status is worked out from the current state every time (the user row,
the trial row and all of the customer's Stripe subscriptions), never from the
event that triggered a sync, so an out-of-order webhook can't leave it wrong.

Person properties, set on distinct id = platform user id (dates are ISO-8601
UTC strings of the real lifecycle moment, never the sync time; a date that
doesn't apply is ``$unset``, never sent as null):

- ``subscription_status``: one of ``signed``, ``in_trial``,
  ``trial_canceled``, ``subscribed``, ``subscription_canceled`` (canceled,
  still active until the period ends), ``payment_failed`` (renewal payment
  failed; Stripe ``past_due`` or ``unpaid``, access lost),
  ``subscription_ended``. Rules in
  ``compute_lifecycle_snapshot``.
- ``signup_at``: ``User.createdAt``.
- ``trial_started_at``: ``SubscriptionTrial.startedAt``, for a trial whose
  checkout completed and wasn't rejected; for a Stripe-native trial (one
  created outside our trial flow, e.g. in the dashboard), its ``trial_start``.
- ``subscription_started_at``: start of the current or most recent paid
  subscription (Stripe ``start_date``); for a converted trial, the conversion
  (``SubscriptionTrial.convertedAt``).
- ``subscription_canceled_at``: Stripe ``canceled_at``, while
  ``subscription_canceled`` or ``subscription_ended``.
- ``subscription_ended_at``: Stripe ``ended_at``, while ``subscription_ended``.
"""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from prisma.enums import SubscriptionTier
from pydantic import BaseModel, Field

from backend.data.subscription_trial import TrialState

LifecycleStatus = Literal[
    "signed",
    "in_trial",
    "trial_canceled",
    "subscribed",
    "subscription_canceled",
    "payment_failed",
    "subscription_ended",
]

# Stripe statuses in which a paid subscriber still has a subscription.
_LIVE_STATUSES = frozenset({"active"})
# A renewal payment failed. The backend drops the user to NO_TIER, so they have
# lost access, but they didn't choose to leave (SECRT-2778, question 3).
_PAYMENT_FAILED_STATUSES = frozenset({"past_due", "unpaid"})
# A subscription in one of these was never paid for.
_NEVER_PAID_STATUSES = frozenset({"trialing", "incomplete", "incomplete_expired"})
_EPOCH = datetime.min.replace(tzinfo=UTC)
_NEVER = datetime.max.replace(tzinfo=UTC)


class StripeSubscriptionFacts(BaseModel):
    """The fields of a Stripe subscription the mapping reads.

    Validates straight from a Stripe object or dict; Stripe's unix
    timestamps become UTC datetimes.
    """

    id: str
    customer: str | None = None
    status: str
    cancel_at_period_end: bool = False
    start_date: datetime | None = None
    created: datetime | None = None
    canceled_at: datetime | None = None
    ended_at: datetime | None = None
    trial_start: datetime | None = None
    trial_end: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def started_at(self) -> datetime | None:
        return self.start_date or self.created

    @property
    def trial_enrollment_id(self) -> str | None:
        return self.metadata.get("trial_enrollment_id") or None


class LifecycleUser(BaseModel):
    id: str
    created_at: datetime | None = None
    stripe_customer_id: str | None = None
    subscription_tier: SubscriptionTier | None = None


class LifecycleSnapshot(BaseModel):
    subscription_status: LifecycleStatus
    signup_at: datetime | None = None
    trial_started_at: datetime | None = None
    subscription_started_at: datetime | None = None
    subscription_canceled_at: datetime | None = None
    subscription_ended_at: datetime | None = None

    def person_update(self) -> tuple[dict[str, str], list[str]]:
        """The ``$set`` and ``$unset`` for this snapshot.

        A date that doesn't apply is unset rather than sent as null, so a
        value left from an earlier state (say, the end date of a subscription
        the user has since replaced) can't linger on the person.
        """
        dates: dict[str, datetime | None] = self.model_dump(
            exclude={"subscription_status"}
        )
        to_set = {"subscription_status": self.subscription_status}
        to_set.update({name: _iso(value) for name, value in dates.items() if value})
        return to_set, [name for name, value in dates.items() if value is None]


def compute_lifecycle_snapshot(
    *,
    user: LifecycleUser,
    trial: TrialState | None,
    subscriptions: Iterable[StripeSubscriptionFacts],
    now: datetime,
) -> LifecycleSnapshot:
    """Map the current state to ``subscription_status`` and its dates.

    First match wins:

    0. ENTERPRISE (admin-managed, never touched by Stripe) -> ``subscribed``.
       A live Stripe subscription alongside it still supplies the start date.
       A self-serve paid tier with no Stripe subscription is not treated as a
       grant: the column defaulted to PRO from 2026-03-26 to 2026-05-01, so
       most such rows never paid. They follow the rules below.
    1. Live paid subscription set to cancel at period end
       -> ``subscription_canceled``
    2. Live paid subscription (``active``) -> ``subscribed``
    3. In trial and set to cancel -> ``trial_canceled``
    4. In trial -> ``in_trial``. Our own trial first (including the hour
       between the trial ending and its first invoice being charged, when
       Stripe already says ``active`` but the trial hasn't converted), else a
       Stripe-native ``trialing`` subscription with no trial enrollment.
    5. The most recent paid subscription is ``past_due`` or ``unpaid``
       -> ``payment_failed``
    6. A paid subscription that has ended -> ``subscription_ended``
    7. A trial that ended without converting -> ``trial_canceled``
    8. Otherwise -> ``signed``

    A trial's own Stripe subscription counts as paid only once the trial has
    converted, and its paid start is then the conversion, not the trial start.
    The subscription dates belong to the subscription statuses; they are
    unset in every other status.
    """
    tier = SubscriptionTier(user.subscription_tier or SubscriptionTier.NO_TIER)
    started = _trial_started(trial)
    all_subs = list(subscriptions)
    common: dict[str, Any] = {
        "signup_at": user.created_at,
        "trial_started_at": (
            trial.started_at if trial and started else _native_trial_start(all_subs)
        ),
    }
    paid = [sub for sub in all_subs if _is_paid(sub, trial)]
    live = _current_live(paid)

    if tier == SubscriptionTier.ENTERPRISE:
        return LifecycleSnapshot(
            subscription_status="subscribed",
            subscription_started_at=_paid_start(live, trial) if live else None,
            **common,
        )
    if live is not None:
        canceling = live.cancel_at_period_end
        return LifecycleSnapshot(
            subscription_status="subscription_canceled" if canceling else "subscribed",
            subscription_started_at=_paid_start(live, trial),
            subscription_canceled_at=live.canceled_at if canceling else None,
            **common,
        )
    if (
        trial
        and started
        and (_in_trial(trial, now) or _awaiting_conversion(trial, all_subs))
    ):
        status = "trial_canceled" if trial.cancel_at_period_end else "in_trial"
        return LifecycleSnapshot(subscription_status=status, **common)
    if trial and started and _first_charge_failed(trial, all_subs):
        return LifecycleSnapshot(subscription_status="payment_failed", **common)
    native = _native_trial(all_subs, now)
    if native is not None:
        status = "trial_canceled" if native.cancel_at_period_end else "in_trial"
        return LifecycleSnapshot(subscription_status=status, **common)
    ended_native = any(_is_ended_native_trial(sub, trial) for sub in all_subs)
    return _after_access_ended(paid, trial, started or ended_native, common)


def _is_ended_native_trial(
    sub: StripeSubscriptionFacts, trial: TrialState | None
) -> bool:
    return (
        not sub.trial_enrollment_id
        and sub.trial_start is not None
        and sub.status != "trialing"
        and not _is_paid(sub, trial)
    )


def _after_access_ended(
    paid: Sequence[StripeSubscriptionFacts],
    trial: TrialState | None,
    trial_started: bool,
    common: dict[str, Any],
) -> LifecycleSnapshot:
    """Rules 5-8: nothing live, no trial running."""
    ended = max(paid, key=_ended_order, default=None)
    if ended is not None and ended.status in _PAYMENT_FAILED_STATUSES:
        return LifecycleSnapshot(
            subscription_status="payment_failed",
            subscription_started_at=_paid_start(ended, trial),
            **common,
        )
    if ended is not None:
        return LifecycleSnapshot(
            subscription_status="subscription_ended",
            subscription_started_at=_paid_start(ended, trial),
            subscription_canceled_at=ended.canceled_at,
            subscription_ended_at=ended.ended_at,
            **common,
        )
    if trial and trial.converted_at is not None:
        # The converted subscription is missing from Stripe's list, but the
        # conversion alone proves a paid subscription existed and has ended.
        return LifecycleSnapshot(
            subscription_status="subscription_ended",
            subscription_started_at=trial.converted_at,
            **common,
        )
    if trial_started:
        return LifecycleSnapshot(subscription_status="trial_canceled", **common)
    return LifecycleSnapshot(subscription_status="signed", **common)


def _trial_started(trial: TrialState | None) -> bool:
    """Checkout completed and the offer was not rejected (card reuse etc.)."""
    return (
        trial is not None
        and trial.consumed_at is not None
        and trial.rejection_reason is None
    )


def _in_trial(trial: TrialState, now: datetime) -> bool:
    # Not ``TrialState.active``: that is the entitlement check and also needs
    # a currently verified card. A lapsed card suspends the entitlement, but
    # the person is still in their trial.
    return (
        trial.status == "trialing"
        and trial.converted_at is None
        and trial.ends_at is not None
        and _utc(trial.ends_at) > now
    )


def _awaiting_conversion(
    trial: TrialState, subscriptions: Sequence[StripeSubscriptionFacts]
) -> bool:
    """The trial ended and Stripe made its subscription ``active``, but the
    first invoice is still a draft (Stripe charges it about an hour later),
    so the trial hasn't converted yet. A failed charge moves the subscription
    to ``past_due`` and ends this."""
    return trial.converted_at is None and any(
        sub.id == trial.subscription_id
        and sub.trial_enrollment_id == trial.id
        and sub.status in _LIVE_STATUSES
        for sub in subscriptions
    )


def _first_charge_failed(
    trial: TrialState, subscriptions: Sequence[StripeSubscriptionFacts]
) -> bool:
    """The trial's first charge failed and Stripe is still retrying it: the
    same state the backend reports as ``payment_failed``."""
    return trial.converted_at is None and any(
        sub.id == trial.subscription_id
        and sub.trial_enrollment_id == trial.id
        and sub.status in _PAYMENT_FAILED_STATUSES
        for sub in subscriptions
    )


def _is_paid(sub: StripeSubscriptionFacts, trial: TrialState | None) -> bool:
    if sub.trial_enrollment_id:
        return (
            trial is not None
            and trial.id == sub.trial_enrollment_id
            and trial.subscription_id == sub.id
            and trial.converted_at is not None
        )
    if sub.status in _NEVER_PAID_STATUSES:
        return False
    return not _ended_during_native_trial(sub)


def _ended_during_native_trial(sub: StripeSubscriptionFacts) -> bool:
    """A Stripe-native trial canceled before its first charge was never paid."""
    ended = sub.ended_at or sub.canceled_at
    return (
        sub.trial_end is not None
        and sub.status not in _LIVE_STATUSES
        and ended is not None
        and _utc(ended) <= _utc(sub.trial_end)
    )


def _current_live(
    paid: Sequence[StripeSubscriptionFacts],
) -> StripeSubscriptionFacts | None:
    # Normally at most one. If there are two, the one not being canceled is
    # the subscription the user is keeping.
    return max(
        (sub for sub in paid if sub.status in _LIVE_STATUSES),
        key=lambda sub: (not sub.cancel_at_period_end, _utc(sub.started_at)),
        default=None,
    )


def _native_trial(
    subscriptions: Sequence[StripeSubscriptionFacts], now: datetime
) -> StripeSubscriptionFacts | None:
    """A running Stripe trial that didn't come through our trial flow."""
    return max(
        (
            sub
            for sub in subscriptions
            if sub.status == "trialing"
            and not sub.trial_enrollment_id
            and (sub.trial_end is None or _utc(sub.trial_end) > now)
        ),
        key=lambda sub: _utc(sub.started_at),
        default=None,
    )


def _ended_order(sub: StripeSubscriptionFacts) -> tuple[datetime, datetime]:
    # The most recently started subscription is the one the user is on. Not
    # the end date: after an upgrade the old subscription is canceled after
    # the new one started. Ties go to the one that ended last, and a failed
    # renewal hasn't ended. Known gap: an older subscription that outlived a
    # newer one loses to it; Stripe data alone can't tell which one the user
    # kept using.
    failed = sub.status in _PAYMENT_FAILED_STATUSES
    return (
        _utc(sub.started_at),
        _NEVER if failed else _utc(sub.ended_at or sub.canceled_at),
    )


def _native_trial_start(
    subscriptions: Sequence[StripeSubscriptionFacts],
) -> datetime | None:
    """Start of the newest Stripe-native trial (one outside our trial flow),
    kept after it converts or ends, as our own trial's start is."""
    native = max(
        (s for s in subscriptions if not s.trial_enrollment_id and s.trial_start),
        key=lambda s: _utc(s.trial_start),
        default=None,
    )
    return native.trial_start if native else None


def _paid_start(
    sub: StripeSubscriptionFacts, trial: TrialState | None
) -> datetime | None:
    if sub.trial_enrollment_id and trial is not None and trial.converted_at:
        return trial.converted_at
    return sub.started_at


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return _EPOCH
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _iso(value: datetime) -> str:
    return _utc(value).astimezone(UTC).isoformat(timespec="seconds")
