"""Give accounts the MailerLite status and dates the live code would have.

Signups were never synced and the lifecycle handlers only react to new
events, so this works out each account's fields from scratch: the signup date
from our database, the rest from Stripe, by the rules in `subscriber_fields.py`.
Only accounts with a Stripe customer are given (see `cli/mailerlite_backfill`):
MailerLite holds checkout openers, not every signup. An account that opted out
of marketing is never written, since a field write creates the subscriber
(`consent.py`).

Resumable by construction: the fields MailerLite already holds are read first
and only the difference is written, so an interrupted or repeated run picks up
where the last one stopped.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from urllib.parse import urlencode

from pydantic import BaseModel, EmailStr, TypeAdapter, ValidationError

from backend.data.notifications import SubscriberField, SubscriptionStatus
from backend.notifications.consent import marketing_allowed
from backend.notifications.mailerlite import (
    API_BASE,
    MailerLiteError,
    _client,
    _headers,
    _require_token,
    field_type,
)
from backend.notifications.mailerlite_backfill import (
    BATCH_SIZE,
    MEMBER_STATUSES,
    PAGE_SIZE,
    UPSERT_BATCH_INTERVAL_SECONDS,
    Subscription,
    _refusal,
    _send_batch,
    next_cursor,
)
from backend.notifications.subscriber_fields import Fields, mailerlite_date

logger = logging.getLogger(__name__)

_EMAIL = TypeAdapter(EmailStr)
_NEVER_STARTED = ("incomplete", "incomplete_expired", "trialing")

# Email → our fields as MailerLite holds them.
Current = dict[str, dict[str, str | None]]


class Person(BaseModel):
    """A platform account and its Stripe subscriptions, if any."""

    user_id: str
    email: str
    created_at: datetime
    subscriptions: list[Subscription] = []
    stripe_customer_id: str | None = None
    # The browser's IANA timezone, for the checkout opener's country.
    timezone: str | None = None
    # Set when they refused marketing: they must never enter MailerLite.
    marketing_opt_out_at: datetime | None = None


class FieldChange(BaseModel):
    person: Person
    status: SubscriptionStatus
    # Only the fields that differ from MailerLite's.
    fields: Fields
    # Not a MailerLite subscriber yet: writing creates one.
    new: bool


class FieldPlan(BaseModel):
    statuses: dict[SubscriptionStatus, int]
    changes: list[FieldChange]
    # Addresses MailerLite would refuse, such as a reserved domain.
    invalid: int
    # People who refused marketing, left out of the plan and the statuses.
    opted_out: int


def standing(subscriptions: list[Subscription]) -> tuple[SubscriptionStatus, Fields]:
    """The status and every date but signup. A paid subscription outranks a
    trial, and a live one outranks one that ended."""
    trials = [s for s in subscriptions if s.from_trial]
    trial_start = max((s.trial_start for s in trials if s.trial_start), default=None)
    dates: Fields = {
        SubscriberField.TRIAL_STARTED: _day(trial_start),
        SubscriberField.SUBSCRIPTION_STARTED: None,
        SubscriberField.SUBSCRIPTION_CANCELED: None,
        SubscriberField.SUBSCRIPTION_ENDED: None,
    }
    paid = [s for s in subscriptions if _ever_paid(s)]
    live = [s for s in paid if s.status in ("active", "past_due")]
    if live:
        current = _latest(live)
        dates[SubscriberField.SUBSCRIPTION_STARTED] = _day(_started(current))
        if current.cancel_at_period_end:
            dates[SubscriberField.SUBSCRIPTION_CANCELED] = _day(current.canceled_at)
            return SubscriptionStatus.SUBSCRIPTION_CANCELED, dates
        return SubscriptionStatus.SUBSCRIBED, dates
    trialing = [s for s in subscriptions if s.status == "trialing"]
    if any(not s.cancel_at_period_end for s in trialing):
        return SubscriptionStatus.IN_TRIAL, dates
    if trialing:
        return SubscriptionStatus.TRIAL_CANCELED, dates
    if paid:
        last = _latest(paid)
        dates[SubscriberField.SUBSCRIPTION_STARTED] = _day(_started(last))
        if last.cancel_at_period_end:
            dates[SubscriberField.SUBSCRIPTION_CANCELED] = _day(last.canceled_at)
        dates[SubscriberField.SUBSCRIPTION_ENDED] = _day(last.ended_at)
        return SubscriptionStatus.SUBSCRIPTION_ENDED, dates
    if trials:
        # Ended, or failed its first payment, without ever converting.
        return SubscriptionStatus.TRIAL_CANCELED, dates
    return SubscriptionStatus.SIGNED, dates


def desired(person: Person) -> tuple[SubscriptionStatus, Fields]:
    """All six fields, so a run leaves nothing stale behind."""
    status, dates = standing(person.subscriptions)
    return status, {
        SubscriberField.STATUS: status.value,
        SubscriberField.SIGNUP: mailerlite_date(person.created_at),
        **dates,
    }


def plan(people: list[Person], current: Current, *, create: bool = True) -> FieldPlan:
    """Each person's fields that differ from MailerLite's. With create=False,
    someone MailerLite does not hold is left out: only the checkout openers
    backfill brings new people in, since a Stripe customer alone does not
    mean they opened checkout (the billing portal creates one too)."""
    result = FieldPlan(
        statuses={s: 0 for s in SubscriptionStatus},
        changes=[],
        invalid=0,
        opted_out=0,
    )
    for person in people:
        if not marketing_allowed(person):
            result.opted_out += 1
            continue
        if not _valid(person.email):
            result.invalid += 1
            continue
        status, fields = desired(person)
        result.statuses[status] += 1
        held = current.get(person.email.strip().lower())
        if held is None and not create:
            continue
        differ = {
            key: value
            for key, value in fields.items()
            if held is None or _normalise(key, held.get(key.value)) != value
        }
        if differ:
            result.changes.append(
                FieldChange(
                    person=person, status=status, fields=differ, new=held is None
                )
            )
    return result


async def read_current() -> Current:
    """Our fields for every subscriber, in every status: an unsubscribed
    person still gets their fields, and must not be counted as new."""
    _require_token()
    keys = [f.value for f in SubscriberField]
    current: Current = {}
    for status in MEMBER_STATUSES:
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            params = {"limit": str(PAGE_SIZE), "filter[status]": status}
            if cursor:
                params["cursor"] = cursor
            page = await _get_subscribers(params)
            for row in page.get("data") or []:
                fields = row.get("fields") or {}
                current[str(row["email"]).strip().lower()] = {
                    k: fields.get(k) for k in keys
                }
            cursor = next_cursor(page, seen)
            if not cursor:
                break
    return current


async def apply(
    changes: list[FieldChange],
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Write each change as a subscriber upsert, fifty to a batch, paced for
    MailerLite's import limit. Returns (succeeded, failed); a failure is left
    for the next run."""
    succeeded = failed = 0
    for start in range(0, len(changes), BATCH_SIZE):
        if start:
            await asyncio.sleep(UPSERT_BATCH_INTERVAL_SECONDS)
        chunk = changes[start : start + BATCH_SIZE]
        answers = await _send_batch([_upsert(c) for c in chunk])
        for change, answer in zip(chunk, answers):
            if answer.code in (200, 201, 202, 204):
                succeeded += 1
                continue
            failed += 1
            logger.warning(
                f"Field update failed for {_refusal(change.person.email, answer)}; "
                "the next run retries it"
            )
        if on_progress:
            on_progress(start + len(chunk), len(changes))
    return succeeded, failed


def _upsert(change: FieldChange) -> dict:
    return {
        "method": "POST",
        "path": "api/subscribers",
        "body": {
            "email": change.person.email,
            "fields": {key.value: value for key, value in change.fields.items()},
        },
    }


def _ever_paid(subscription: Subscription) -> bool:
    if subscription.from_trial:
        return subscription.converted or subscription.status == "active"
    return subscription.status not in _NEVER_STARTED


def _started(subscription: Subscription) -> int | None:
    """A converted trial's paid subscription began when its trial ended."""
    if subscription.from_trial:
        return subscription.trial_end
    return subscription.start_date


def _latest(subscriptions: list[Subscription]) -> Subscription:
    return max(subscriptions, key=lambda s: _started(s) or 0)


def _day(timestamp: int | None) -> str | None:
    """Unlike a live event, a missing timestamp here is no reason to write
    today's date."""
    return mailerlite_date(timestamp) if timestamp is not None else None


def _normalise(field: SubscriberField, value: object) -> str | None:
    if value is None or value == "":
        return None
    text = str(value)
    # A date may come back with a time part.
    return text[:10] if field_type(field) == "date" else text


def _valid(email: str) -> bool:
    try:
        _EMAIL.validate_python(email)
    except ValidationError:
        return False
    return True


async def _get_subscribers(params: dict[str, str]) -> dict:
    response = await _client().get(
        f"{API_BASE}/subscribers?{urlencode(params)}", headers=_headers()
    )
    if response.status != 200:
        raise MailerLiteError(
            f"Reading MailerLite subscribers failed with {response.status}"
        )
    return response.json() or {}
