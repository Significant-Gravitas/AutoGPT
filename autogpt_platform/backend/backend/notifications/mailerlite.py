"""MailerLite audience management.

Two emails live entirely in MailerLite — the six-email White Glove Tour and the
monthly changelog — and neither needs a deploy to change. The platform's only
job is managing who is in each audience, and there is exactly one owner per
transition:

1. ENTER · tour finishers → changelog: owned by MailerLite. The onboarding
   automation's final step moves the subscriber across. That handoff is what
   implements the suppression rule — nobody gets the monthly update mid-tour,
   because mid-tour users simply are not in the group.
2. ENTER · resubscribers and pre-tour users → changelog: owned here.
3. LEAVE · churn: owned here. Churned users get win-back only, never the
   monthly update.
4. ENTER and LEAVE · the trial group: both owned here. It holds exactly the
   customers currently on a trial, so any MailerLite automation on it may send
   mail but must never move people in or out.
5. ENTER · the checkout openers group: owned here. Everyone who opened Stripe
   checkout joins it, and nobody else enters MailerLite through us: a signup
   alone does not create a subscriber. GTM segments it for outreach.

Someone who opted out of marketing enters none of these. Every write below
upserts the subscriber, a removal or a field update included, so nothing is
queued for them at all (`consent.py`), the backfills leave them out, and the
consumer re-reads the opt-out right before each write, which drops a change
queued just before the refusal. The one exception is `unsubscribe`, which
carries the refusal itself to someone MailerLite already has, and never
creates a subscriber.

Subscriber fields (`SubscriberField`) are the backend's alone: every write
comes from here, and MailerLite automations only read them.

Only the notification service holds the MailerLite settings. Whatever queues an
audience change does not ask whether MailerLite is configured; the consumer
does (`configured()`).

If both sides managed the same edge we would double-add or fight over
removals, so nothing in this module touches the tour → changelog handoff.
"""

import hashlib
import logging
from collections.abc import Mapping

from backend.data.notifications import SubscriberField
from backend.notifications.audience_enrichment import merge_with_held
from backend.util.request import Requests
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


def pseudonym(email: str) -> str:
    """A stable, non-reversible handle for logs.

    A subscriber's address is personal data; putting it in a log line or an
    exception message copies it into log retention and every downstream sink.
    The digest is enough to correlate one subscriber's records without storing
    the address itself.
    """
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:12]


settings = Settings()

API_BASE = settings.config.mailerlite_api_url.rstrip("/")
_OK_STATUSES = (200, 201, 202, 204)
# Subscriber statuses MailerLite sends nothing to, so an unsubscribe has
# nothing left to do. The API cannot set them back to active either.
_NOT_MAILED_STATUSES = ("unsubscribed", "bounced", "junk")

# The type MailerLite stores each of our custom fields as, which is also what
# `ensure_fields` creates. A date is written YYYY-MM-DD.
FIELD_TYPES: dict[SubscriberField, str] = {
    SubscriberField.STATUS: "text",
    SubscriberField.SIGNUP: "date",
    SubscriberField.TRIAL_STARTED: "date",
    SubscriberField.SUBSCRIPTION_STARTED: "date",
    SubscriberField.SUBSCRIPTION_CANCELED: "date",
    SubscriberField.SUBSCRIPTION_ENDED: "date",
    SubscriberField.CHECKOUT_OPENED: "date",
    SubscriberField.EMAIL_TYPE: "text",
    SubscriberField.SIGNIN_METHOD: "text",
    SubscriberField.COUNTRY_CODE: "text",
    SubscriberField.COUNTRY_SOURCE: "text",
    SubscriberField.EXCLUDE_DE_AT: "text",
}

# MailerLite's own fields: written like ours, but never created.
BUILT_IN_FIELD_TYPES: dict[SubscriberField, str] = {SubscriberField.COUNTRY: "text"}


def field_type(field: SubscriberField) -> str:
    return FIELD_TYPES.get(field) or BUILT_IN_FIELD_TYPES[field]


Fields = Mapping[SubscriberField, str | None]

# Set once this process has seen every field exist, so it is checked once,
# not on every write.
_fields_ready = False
# Logged once: without a checkout group, checkout openers are dropped.
_checkout_off_logged = False


class MailerLiteNotConfigured(RuntimeError):
    """A setting the change needs is missing. The consumer dead-letters it at
    once rather than silently reporting success, so it can be replayed once the
    setting is in place."""


def configured() -> bool:
    """A stack without a token, such as a self-hosted one, has no MailerLite:
    its audience changes are acknowledged and dropped, never retried."""
    return bool(settings.secrets.mailerlite_api_token)


class MailerLiteError(RuntimeError):
    """A MailerLite call failed. Raised so the queued job retries with backoff
    — a MailerLite outage must never fail payment processing, but it must not
    silently drop an enrolment either."""


async def enroll_in_onboarding(email: str, fields: Fields | None = None) -> None:
    """Add a first-time subscriber to the tour group. Joining the group is the
    automation's trigger; MailerLite sends the six emails from there."""
    await _add_to_group(
        email, settings.config.mailerlite_onboarding_group_id, "onboarding tour", fields
    )


async def add_to_changelog(email: str, fields: Fields | None = None) -> None:
    """Returning customers and anyone who predates the tour."""
    await _add_to_group(
        email, settings.config.mailerlite_changelog_group_id, "changelog", fields
    )


async def remove_from_changelog(email: str, fields: Fields | None = None) -> None:
    """The day a plan ends."""
    await _remove_from_group(
        email, settings.config.mailerlite_changelog_group_id, "changelog", fields
    )


async def add_to_trial(email: str, fields: Fields | None = None) -> None:
    """A trial started, or a cancelled one was resumed. Without a trial group
    only the fields are written: the group is optional, so waiting for it
    would only retry into the dead-letter queue."""
    group_id = settings.config.mailerlite_trial_group_id
    if not group_id:
        await update_fields(email, fields)
        return
    await _add_to_group(email, group_id, "trial", fields)


async def remove_from_trial(email: str, fields: Fields | None = None) -> None:
    """The trial was cancelled, converted, or ended unpaid. Without a trial
    group only the fields are written, as for `add_to_trial`."""
    group_id = settings.config.mailerlite_trial_group_id
    if not group_id:
        await update_fields(email, fields)
        return
    await _remove_from_group(email, group_id, "trial", fields)


async def update_fields(email: str, fields: Fields | None = None) -> None:
    """Write a subscriber's fields, creating the subscriber if MailerLite has
    none. It never changes their subscription status, so an unsubscribed
    person stays unsubscribed."""
    if not fields:
        return
    _require_token()
    await ensure_fields()
    await _upsert(email, {"fields": _payload(fields)}, "field update")


async def record_signup(email: str, fields: Fields | None = None) -> None:
    """A signup queued before signups stopped being sent here. It only fills
    in someone MailerLite already has: a signup alone must not create a
    subscriber, since only checkout openers belong in MailerLite. `signed` is
    where every subscriber starts, so it never replaces a status MailerLite
    already holds. The audience queue is worked one message at a time, so
    nothing of ours writes between the read and the write."""
    if not fields:
        return
    _require_token()
    subscriber = await _find_subscriber(email)
    if subscriber is None:
        return
    if (subscriber.get("fields") or {}).get(SubscriberField.STATUS.value):
        fields = {k: v for k, v in fields.items() if k != SubscriberField.STATUS}
    await update_fields(email, fields)


async def record_checkout_opened(email: str, fields: Fields | None = None) -> None:
    """Someone opened Stripe checkout: into the checkout openers group, with
    the fields GTM segments on. Whatever MailerLite already holds that is
    better than ours stays (`audience_enrichment.merge_with_held`): the audience
    queue is worked one message at a time, so nothing of ours writes between
    the read and the write.

    Without a checkout group the change is dropped rather than dead-lettered:
    the group is how this is switched on, and the backfill brings in anyone
    opened while it was off."""
    group_id = settings.config.mailerlite_checkout_group_id
    if not group_id:
        global _checkout_off_logged
        if not _checkout_off_logged:
            logger.warning(
                "MAILERLITE_CHECKOUT_GROUP_ID is not set; dropping checkout openers"
            )
            _checkout_off_logged = True
        return
    _require_token()
    held = (await _find_subscriber(email) or {}).get("fields") or {}
    await _add_to_group(
        email, group_id, "checkout openers", merge_with_held(fields or {}, held)
    )


async def unsubscribe(email: str, fields: Fields | None = None) -> None:
    """The account refused marketing: mark the subscriber unsubscribed so no
    campaign or automation reaches them. Someone MailerLite does not have is
    left alone, since creating a subscriber is exactly what the refusal rules
    out, and so is one MailerLite already does not mail. `fields` is ignored:
    nothing else is written for someone who refused.

    An update by subscriber ID (`PUT /subscribers/{id}`), not the upsert the
    other writes use, so it can never create anyone."""
    _require_token()
    subscriber = await _find_subscriber(email)
    if subscriber is None or not subscriber.get("id"):
        logger.info(
            f"No MailerLite subscriber for {pseudonym(email)}; nothing to unsubscribe"
        )
        return
    if subscriber.get("status") in _NOT_MAILED_STATUSES:
        return
    response = await _client().put(
        f"{API_BASE}/subscribers/{subscriber['id']}",
        headers=_headers(),
        json={"status": "unsubscribed"},
    )
    # 404 means the subscriber was deleted since the lookup: nobody to mail.
    if response.status not in _OK_STATUSES and response.status != 404:
        raise MailerLiteError(
            f"Unsubscribing MailerLite subscriber {pseudonym(email)} failed with "
            f"{response.status}"
        )
    logger.info(f"Unsubscribed {pseudonym(email)} in MailerLite")


async def ensure_fields() -> list[SubscriberField]:
    """Create any of our fields MailerLite does not have yet, and return the
    ones created. Idempotent, and checked once per process."""
    global _fields_ready
    if _fields_ready:
        return []
    _require_token()
    existing = await read_fields()
    created = []
    for field, kind in FIELD_TYPES.items():
        if field.value in existing:
            continue
        response = await _client().post(
            f"{API_BASE}/fields",
            headers=_headers(),
            json={"name": field.value, "type": kind},
        )
        if response.status not in _OK_STATUSES:
            raise MailerLiteError(
                f"Creating the MailerLite field {field.value} failed with "
                f"{response.status}"
            )
        key = ((response.json() or {}).get("data") or {}).get("key")
        if key != field.value:
            # Writes go by key, so a field under another key would silently
            # never be filled.
            raise MailerLiteError(
                f"MailerLite created the field {field.value} as {key!r}"
            )
        created.append(field)
        logger.info(f"Created the MailerLite field {field.value}")
    _fields_ready = True
    return created


# An account holds far fewer fields than this many pages of 100; more means
# MailerLite's last_page keeps moving.
_MAX_FIELD_PAGES = 50


async def read_fields() -> dict[str, str]:
    """Every custom field MailerLite has, as key → type. An empty page ends the
    read whatever last_page says."""
    fields: dict[str, str] = {}
    for page in range(1, _MAX_FIELD_PAGES + 1):
        response = await _client().get(
            f"{API_BASE}/fields?limit=100&page={page}", headers=_headers()
        )
        if response.status != 200:
            raise MailerLiteError(
                f"Reading MailerLite fields failed with {response.status}"
            )
        body = response.json() or {}
        rows = body.get("data") or []
        for row in rows:
            fields[str(row["key"])] = str(row.get("type") or "")
        if not rows or page >= int((body.get("meta") or {}).get("last_page") or 1):
            return fields
    raise MailerLiteError(f"MailerLite still had fields after {_MAX_FIELD_PAGES} pages")


async def _remove_from_group(
    email: str, group_id: str, description: str, fields: Fields | None
) -> None:
    # Fields first: they are written even when the group change cannot be.
    await update_fields(email, fields)
    _require_config(group_id, description)

    subscriber_id = await _find_subscriber_id(email)
    if subscriber_id is None:
        logger.info(
            "No MailerLite subscriber for %s; nothing to remove", pseudonym(email)
        )
        return

    response = await _client().delete(
        f"{API_BASE}/subscribers/{subscriber_id}/groups/{group_id}",
        headers=_headers(),
    )
    # 404 means they are already out of the group, which is the desired state.
    if response.status not in _OK_STATUSES and response.status != 404:
        raise MailerLiteError(
            f"Removing subscriber {pseudonym(email)} from the {description} group "
            f"failed with {response.status}"
        )
    logger.info(f"Removed {pseudonym(email)} from the MailerLite {description} group")


async def _add_to_group(
    email: str, group_id: str, description: str, fields: Fields | None
) -> None:
    if not group_id:
        # The fields still land; the group change is then dead-lettered.
        await update_fields(email, fields)
    _require_config(group_id, description)
    body: dict = {"groups": [group_id]}
    if fields:
        await ensure_fields()
        body["fields"] = _payload(fields)
    await _upsert(email, body, f"{description} group")
    logger.info("Added %s to the MailerLite %s group", pseudonym(email), description)


async def _upsert(email: str, body: dict, description: str) -> None:
    response = await _client().post(
        f"{API_BASE}/subscribers", headers=_headers(), json={"email": email, **body}
    )
    if response.status not in _OK_STATUSES:
        raise MailerLiteError(
            f"MailerLite {description} for subscriber {pseudonym(email)} "
            f"failed with {response.status}"
        )


def _payload(fields: Fields) -> dict[str, str | None]:
    return {SubscriberField(key).value: value for key, value in fields.items()}


async def _find_subscriber_id(email: str) -> str | None:
    return (await _find_subscriber(email) or {}).get("id")


async def _find_subscriber(email: str) -> dict | None:
    response = await _client().get(
        f"{API_BASE}/subscribers/{email}", headers=_headers()
    )
    if response.status == 404:
        return None
    if response.status not in _OK_STATUSES:
        raise MailerLiteError(
            f"Looking up MailerLite subscriber {pseudonym(email)} failed with "
            f"{response.status}"
        )
    return (response.json() or {}).get("data") or {}


def _client() -> Requests:
    # Statuses are inspected rather than raised on, because "already gone" is a
    # success for a removal.
    return Requests(trusted_origins=[API_BASE], raise_for_status=False)


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.secrets.mailerlite_api_token}",
        "Accept": "application/json",
    }


def _require_token() -> None:
    if not settings.secrets.mailerlite_api_token:
        raise MailerLiteNotConfigured("MAILERLITE_API_TOKEN is not set")


def _require_config(group_id: str, description: str) -> None:
    _require_token()
    if not group_id:
        raise MailerLiteNotConfigured(
            f"The MailerLite {description} group ID is not configured"
        )
