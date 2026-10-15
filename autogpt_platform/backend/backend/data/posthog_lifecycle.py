"""Subscription status and lifecycle dates (SECRT-2778, SECRT-2795).

The one mapping both sinks use: the PostHog person properties, and the
MailerLite subscriber fields of the same names (``subscription_status`` and
the ``*_date`` fields, see ``notifications/lifecycle_audience.py``). This
module is the mapping only: pure, no I/O. ``posthog_lifecycle_sync`` reads the
inputs and sends the result to both.

The status is worked out from the current state every time (the trial row and
all of the customer's Stripe subscriptions), never from the event that
triggered a sync, so an out-of-order webhook can't leave it wrong. The
subscription tier is not read: only a payment makes a subscriber, so an
admin-granted plan (ENTERPRISE or any other) with no paid Stripe subscription
is ``signed``.

Person properties, set on distinct id = platform user id (dates are ISO-8601
UTC strings of the real lifecycle moment, never the sync time; a date that
doesn't apply is ``$unset``, never sent as null):

- ``subscription_status``: one of ``signed``, ``in_trial``,
  ``trial_canceled`` (the user canceled their trial), ``trial_expired`` (the
  trial ended without a payment and without the user canceling it),
  ``subscribed``, ``subscription_canceled`` (canceled, still active until the
  period ends), ``payment_failed`` (a payment failed; Stripe ``past_due`` or
  ``unpaid``, access lost), ``subscription_ended``. Rules in
  ``compute_lifecycle_snapshot``.
- ``signup_at``: ``User.createdAt``.
- ``trial_started_at``: ``SubscriptionTrial.startedAt``, for a trial whose
  checkout completed and wasn't rejected; for a Stripe-native trial (one
  created outside our trial flow, e.g. in the dashboard), its ``trial_start``.
- ``subscription_started_at``: when the person first became a paying
  subscriber: the earliest start of any paid subscription (Stripe
  ``start_date``), kept across a resubscribe after a gap. For a converted
  trial, the conversion (``SubscriptionTrial.convertedAt``), not the trial
  start. Kept too for a former payer now on a free subscription, who is
  ``signed``.
- ``subscription_canceled_at``: Stripe ``canceled_at`` of the current or most
  recent paid subscription, while ``subscription_canceled`` or
  ``subscription_ended``.
- ``subscription_ended_at``: its Stripe ``ended_at``, while
  ``subscription_ended``.

A paid subscription is one that charges something: a subscription whose
discounts take the whole price (a 100% coupon, or an amount off at least the
price, on the subscription or the customer) is treated like a trial that never
converted, and never makes anyone ``subscribed``, ``subscription_canceled`` or
``subscription_ended``. So is one whose only invoice so far charged nothing
(a 100% coupon for the first period, which Stripe takes off the subscription
once that invoice is made). A partial discount is still paid, and so is a
subscription whose full discount has ended.
"""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel

from backend.data.stripe_subscription_facts import StripeSubscriptionFacts, utc
from backend.data.subscription_trial import TrialState

LifecycleStatus = Literal[
    "signed",
    "in_trial",
    "trial_canceled",
    "trial_expired",
    "subscribed",
    "subscription_canceled",
    "payment_failed",
    "subscription_ended",
]

# Stripe statuses in which a paid subscriber still has a subscription.
_LIVE_STATUSES = frozenset({"active"})
# A payment failed. The backend drops the user to NO_TIER, so they have lost
# access, but they didn't choose to leave (SECRT-2778, question 3).
_PAYMENT_FAILED_STATUSES = frozenset({"past_due", "unpaid"})
# A subscription in one of these was never paid for.
_NEVER_PAID_STATUSES = frozenset({"trialing", "incomplete", "incomplete_expired"})
_NEVER = datetime.max.replace(tzinfo=UTC)
# A coupon applied at checkout can start a moment after the subscription;
# one that started later than this left at least the first period charged.
_FIRST_CHARGE_GRACE = timedelta(days=1)


class LifecycleUser(BaseModel):
    id: str
    created_at: datetime | None = None
    stripe_customer_id: str | None = None


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

    1. Live paid subscription set to cancel at period end
       -> ``subscription_canceled``
    2. Live paid subscription (``active``) -> ``subscribed``
    3. In trial and set to cancel -> ``trial_canceled``
    4. In trial -> ``in_trial``. Our own trial first (including the hour
       between the trial ending and its first invoice being charged, when
       Stripe already says ``active`` but the trial hasn't converted), else a
       Stripe-native ``trialing`` subscription with no trial enrollment.
    5. A trial's first charge failed and Stripe is still retrying it
       -> ``payment_failed``
    6. Has paid before and now has a live free subscription (a customer
       given a 100% coupon) -> ``signed``, keeping the first paid start
    7. The most recent paid subscription is ``past_due`` or ``unpaid``
       -> ``payment_failed``
    8. A paid subscription that has ended -> ``subscription_ended``
    9. A trial that ended without converting -> ``trial_canceled`` if the
       user canceled it (set it to cancel, or asked for the cancel), else
       ``trial_expired`` (it ran out, or its first charge never cleared)
    10. Otherwise -> ``signed``

    Paid means charging something (see the module docstring). A trial's own
    Stripe subscription counts as paid only once the trial has converted, and
    its paid start is then the conversion, not the trial start. The
    subscription dates belong to the subscription statuses; they are unset in
    every other status, except the first paid start of rule 6.
    """
    started = _trial_started(trial)
    all_subs = list(subscriptions)
    common: dict[str, Any] = {
        "signup_at": user.created_at,
        "trial_started_at": (
            trial.started_at if trial and started else _native_trial_start(all_subs)
        ),
    }
    paid = [sub for sub in all_subs if _is_paid(sub, trial, now)]
    charged = [sub for sub in all_subs if _charged_before_free(sub, trial)]
    first_paid = _first_paid_start([*paid, *charged], trial, all_subs)
    live = _current_live(paid)

    if live is not None:
        canceling = live.cancel_at_period_end
        return LifecycleSnapshot(
            subscription_status="subscription_canceled" if canceling else "subscribed",
            subscription_started_at=first_paid,
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
    if first_paid is not None and _has_live_free(all_subs, now):
        # Comped: not paying now, but not churned either.
        return LifecycleSnapshot(
            subscription_status="signed", subscription_started_at=first_paid, **common
        )
    ended = _after_access_ended(paid, first_paid, common)
    if ended is not None:
        return ended
    trial_end = _ended_trial_status(trial, started, all_subs, now)
    return LifecycleSnapshot(subscription_status=trial_end or "signed", **common)


def _has_live_free(
    subscriptions: Sequence[StripeSubscriptionFacts], now: datetime
) -> bool:
    """A live subscription that charges nothing (see the module docstring)."""
    return any(
        sub.status in _LIVE_STATUSES and sub.free_at(now) for sub in subscriptions
    )


def _after_access_ended(
    paid: Sequence[StripeSubscriptionFacts],
    first_paid: datetime | None,
    common: dict[str, Any],
) -> LifecycleSnapshot | None:
    """Rules 5-6: nothing live, no trial running, and they have paid."""
    ended = max(paid, key=_ended_order, default=None)
    if ended is not None and ended.status in _PAYMENT_FAILED_STATUSES:
        return LifecycleSnapshot(
            subscription_status="payment_failed",
            subscription_started_at=first_paid,
            **common,
        )
    if ended is not None:
        return LifecycleSnapshot(
            subscription_status="subscription_ended",
            subscription_started_at=first_paid,
            subscription_canceled_at=ended.canceled_at,
            subscription_ended_at=ended.ended_at,
            **common,
        )
    if first_paid is not None:
        # The converted subscription is missing from Stripe's list, but the
        # conversion alone proves a paid subscription existed and has ended.
        return LifecycleSnapshot(
            subscription_status="subscription_ended",
            subscription_started_at=first_paid,
            **common,
        )
    return None


def _ended_trial_status(
    trial: TrialState | None,
    started: bool,
    subscriptions: Sequence[StripeSubscriptionFacts],
    now: datetime,
) -> LifecycleStatus | None:
    """Rule 7: a trial that ended unpaid, canceled by the user or not."""
    if trial and started:
        canceled = trial.cancel_at_period_end or any(
            sub.id == trial.subscription_id and sub.cancel_requested
            for sub in subscriptions
        )
        return "trial_canceled" if canceled else "trial_expired"
    native = max(
        (sub for sub in subscriptions if _is_ended_native_trial(sub, trial, now)),
        key=lambda sub: utc(sub.trial_start),
        default=None,
    )
    if native is None:
        return None
    return "trial_canceled" if native.cancel_requested else "trial_expired"


def _is_ended_native_trial(
    sub: StripeSubscriptionFacts, trial: TrialState | None, now: datetime
) -> bool:
    return (
        not sub.trial_enrollment_id
        and sub.trial_start is not None
        and sub.status != "trialing"
        and not _is_paid(sub, trial, now)
    )


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
        and utc(trial.ends_at) > now
    )


def _awaiting_conversion(
    trial: TrialState, subscriptions: Sequence[StripeSubscriptionFacts]
) -> bool:
    """The trial ended and Stripe made its subscription ``active``, but the
    first invoice is still a draft (Stripe charges it about an hour later),
    so the trial hasn't converted yet. A failed charge moves the subscription
    to ``past_due`` and ends this."""
    return trial.converted_at is None and any(
        _is_trial_subscription(sub, trial) and sub.status in _LIVE_STATUSES
        for sub in subscriptions
    )


def _first_charge_failed(
    trial: TrialState, subscriptions: Sequence[StripeSubscriptionFacts]
) -> bool:
    """The trial's first charge failed and Stripe is still retrying it: the
    same state the backend reports as ``payment_failed``."""
    return trial.converted_at is None and any(
        _is_trial_subscription(sub, trial) and sub.status in _PAYMENT_FAILED_STATUSES
        for sub in subscriptions
    )


def _is_trial_subscription(sub: StripeSubscriptionFacts, trial: TrialState) -> bool:
    return sub.id == trial.subscription_id and sub.trial_enrollment_id == trial.id


def _is_paid(
    sub: StripeSubscriptionFacts, trial: TrialState | None, now: datetime
) -> bool:
    if sub.free_at(utc(sub.ended_at) if sub.ended_at else now):
        return False
    return _charges(sub, trial)


def _charged_before_free(
    sub: StripeSubscriptionFacts, trial: TrialState | None
) -> bool:
    """Free now (or when it ended), but it charged first: the full discount
    started after its first paid period began. Its paid start still counts
    as the first paid start."""
    start = _paid_start(sub, trial)
    return (
        start is not None
        and _charges(sub, trial)
        and not sub.free_at(utc(start) + _FIRST_CHARGE_GRACE)
    )


def _charges(sub: StripeSubscriptionFacts, trial: TrialState | None) -> bool:
    """Everything that makes a subscription paid except its discounts."""
    if sub.trial_enrollment_id:
        return (
            trial is not None
            and _is_trial_subscription(sub, trial)
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
        and utc(ended) <= utc(sub.trial_end)
    )


def _first_paid_start(
    paid: Sequence[StripeSubscriptionFacts],
    trial: TrialState | None,
    subscriptions: Sequence[StripeSubscriptionFacts],
) -> datetime | None:
    """When they first became a paying subscriber. A converted trial whose
    subscription is missing from Stripe's list still proves a payment at its
    conversion; one that is listed but free does not."""
    starts = [_paid_start(sub, trial) for sub in paid]
    if (
        trial is not None
        and trial.converted_at is not None
        and not any(sub.id == trial.subscription_id for sub in subscriptions)
    ):
        starts.append(trial.converted_at)
    return min((s for s in starts if s), key=utc, default=None)


def _current_live(
    paid: Sequence[StripeSubscriptionFacts],
) -> StripeSubscriptionFacts | None:
    # Normally at most one. If there are two, the one not being canceled is
    # the subscription the user is keeping.
    return max(
        (sub for sub in paid if sub.status in _LIVE_STATUSES),
        key=lambda sub: (not sub.cancel_at_period_end, utc(sub.started_at)),
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
            and (sub.trial_end is None or utc(sub.trial_end) > now)
        ),
        key=lambda sub: utc(sub.started_at),
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
        utc(sub.started_at),
        _NEVER if failed else utc(sub.ended_at or sub.canceled_at),
    )


def _native_trial_start(
    subscriptions: Sequence[StripeSubscriptionFacts],
) -> datetime | None:
    """Start of the newest Stripe-native trial (one outside our trial flow),
    kept after it converts or ends, as our own trial's start is."""
    native = max(
        (s for s in subscriptions if not s.trial_enrollment_id and s.trial_start),
        key=lambda s: utc(s.trial_start),
        default=None,
    )
    return native.trial_start if native else None


def _paid_start(
    sub: StripeSubscriptionFacts, trial: TrialState | None
) -> datetime | None:
    if sub.trial_enrollment_id and trial is not None and trial.converted_at:
        return trial.converted_at
    return sub.started_at


def _iso(value: datetime) -> str:
    return utc(value).astimezone(UTC).isoformat(timespec="seconds")
