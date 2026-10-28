"""Give accounts the MailerLite status and dates the live code would have.

Signups were never synced and the live writes only react to new events, so
this works out each account's fields from scratch, from our database (the
signup and the trial row) and Stripe, by the same mapping PostHog's status
comes from (`data/posthog_lifecycle.py`, via `lifecycle_audience`).
Only accounts with a Stripe customer are given (see `cli/mailerlite_backfill`):
MailerLite holds checkout openers, not every signup. An account that opted out
of marketing, or that a signal places in Iran or Russia, is never written,
since a field write creates the subscriber (`consent.py`).

Resumable by construction: the fields MailerLite already holds are read first
and only the difference is written, so an interrupted or repeated run picks up
where the last one stopped.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlencode

from pydantic import BaseModel, EmailStr, TypeAdapter, ValidationError

from backend.data.notifications import SubscriberField, SubscriptionStatus
from backend.data.posthog_lifecycle import LifecycleUser, compute_lifecycle_snapshot
from backend.data.stripe_subscription_facts import StripeSubscriptionFacts
from backend.data.subscription_trial import TrialState
from backend.notifications.audience_enrichment import points_at_excluded_country
from backend.notifications.consent import KeptOut, marketing_allowed
from backend.notifications.lifecycle_audience import lifecycle_fields
from backend.notifications.mailerlite import (
    API_BASE,
    MailerLiteError,
    _client,
    _headers,
    _require_token,
    held_value,
)
from backend.notifications.mailerlite_backfill import (
    BATCH_SIZE,
    MEMBER_STATUSES,
    PAGE_SIZE,
    UPSERT_BATCH_INTERVAL_SECONDS,
    BatchAnswer,
    _refusal,
    _send_batch,
    next_cursor,
)
from backend.notifications.subscriber_fields import Fields

logger = logging.getLogger(__name__)

_EMAIL = TypeAdapter(EmailStr)

# Email → our fields as MailerLite holds them.
Current = dict[str, dict[str, str | None]]


class Person(BaseModel):
    """A platform account, its Stripe subscriptions and its trial, if any."""

    user_id: str
    email: str
    created_at: datetime
    # Listed with the discounts expanded (`stripe_subscription_facts`).
    subscriptions: list[StripeSubscriptionFacts] = []
    trial: TrialState | None = None
    stripe_customer_id: str | None = None
    # The browser's IANA timezone, for the checkout opener's country.
    timezone: str | None = None
    # Set when they refused marketing: they must never enter MailerLite.
    marketing_opt_out_at: datetime | None = None
    # The Stripe customer's billing address country, when it has one.
    billing_country: str | None = None
    # The Iranian or Russian country a checkout recorded (`consent.py`).
    excluded_country: str | None = None

    def placed_in_excluded_country(self, *countries: str | None) -> bool:
        """Whether any signal, `countries` included, places them in Iran or
        Russia, so they must never enter MailerLite (`consent.py`)."""
        return points_at_excluded_country(
            email=self.email,
            timezone=self.timezone,
            countries=(self.billing_country, self.excluded_country, *countries),
        )


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
    # People a signal places in Iran or Russia, left out the same way.
    excluded_country: int
    # Without create, people MailerLite doesn't hold: left out the same way.
    not_held: int = 0


def desired(
    person: Person, now: datetime | None = None
) -> tuple[SubscriptionStatus, Fields]:
    """All six fields, so a run leaves nothing stale behind: the status and
    dates of the lifecycle snapshot, as the live sync writes them."""
    snapshot = compute_lifecycle_snapshot(
        user=LifecycleUser(
            id=person.user_id,
            created_at=person.created_at,
            stripe_customer_id=person.stripe_customer_id,
        ),
        trial=person.trial,
        subscriptions=person.subscriptions,
        now=now or datetime.now(UTC),
    )
    return SubscriptionStatus(snapshot.subscription_status), lifecycle_fields(snapshot)


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
        excluded_country=0,
    )
    for person in people:
        if not marketing_allowed(person):
            result.opted_out += 1
            continue
        if person.placed_in_excluded_country():
            result.excluded_country += 1
            continue
        if not _valid(person.email):
            result.invalid += 1
            continue
        held = current.get(person.email.strip().lower())
        if held is None and not create:
            result.not_held += 1
            continue
        status, fields = desired(person)
        result.statuses[status] += 1
        differ = {
            key: value
            for key, value in fields.items()
            if held is None or held_value(key, held.get(key.value)) != value
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
    *,
    kept_out: KeptOut | None = None,
) -> tuple[int, int, int]:
    """Write each change as a subscriber upsert, fifty to a batch, paced for
    MailerLite's import limit. With `kept_out`, each batch first drops anyone
    who may no longer be written: they opted out or were seen in Iran or
    Russia since the plan. Returns (succeeded, failed, skipped); a failure,
    an account that can't be read then included, is left for the next run."""
    succeeded = failed = skipped = 0
    for start in range(0, len(changes), BATCH_SIZE):
        if start:
            await asyncio.sleep(UPSERT_BATCH_INTERVAL_SECONDS)
        chunk = changes[start : start + BATCH_SIZE]
        done = start + len(chunk)
        if kept_out:
            chunk, dropped, unreadable = await _writable(chunk, kept_out)
            skipped += dropped
            failed += unreadable
        answers = await _send_batch([_upsert(c) for c in chunk]) if chunk else []
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
            on_progress(done, len(changes))
    return succeeded, failed, skipped


async def _writable(
    chunk: list[FieldChange], kept_out: KeptOut
) -> tuple[list[FieldChange], int, int]:
    """The batch without anyone `kept_out` rejects or whose account it can't
    read, and how many of each were dropped."""
    allowed: list[FieldChange] = []
    rejected = unreadable = 0
    for change in chunk:
        try:
            if await kept_out(change.person.user_id):
                rejected += 1
                continue
        except Exception:
            unreadable += 1
            logger.warning(
                f"Re-reading the account of "
                f"{_refusal(change.person.email, BatchAnswer(code=0))} failed; "
                "the next run retries it"
            )
            continue
        allowed.append(change)
    return allowed, rejected, unreadable


def _upsert(change: FieldChange) -> dict:
    return {
        "method": "POST",
        "path": "api/subscribers",
        "body": {
            "email": change.person.email,
            "fields": {key.value: value for key, value in change.fields.items()},
        },
    }


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
