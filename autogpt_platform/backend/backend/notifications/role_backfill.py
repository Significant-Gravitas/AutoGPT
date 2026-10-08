"""Give accounts from before SECRT-2852 the onboarding role the live code sends.

Since SECRT-2852 the profile step sends the role to MailerLite and PostHog
from the pick the wizard keeps (`data/onboarding_role.py`). Accounts that got
there earlier have no kept pick, only the copy in their business
understanding, which AutoPilot rewrites. An exact option ID there can only
have come from the wizard, so it is reliable; anything else is skipped rather
than guessed (`OnboardingRole.from_understanding`).

Both sides only fill gaps, so an older pick never replaces a newer one:
PostHog gets a `$set_once`, and a MailerLite subscriber is written only while
it has no role. Nobody is created in MailerLite, since only checkout openers
belong there, and nobody is written there who opted out of marketing or whom
a signal places in Iran or Russia (`consent.py`). PostHog is analytics, not
marketing, so it gets every reliable pick.

Each MailerLite write is one update by subscriber ID, one at a time and paced
like `checkout_backfill`, after reading the subscriber again: the live code
keeps writing while a run lasts, and a role it wrote meanwhile stays.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel

from backend.data.notifications import SubscriberField
from backend.data.onboarding_role import OnboardingRole
from backend.notifications.audience_enrichment import (
    points_at_excluded_country,
    role_fields,
)
from backend.notifications.consent import marketing_allowed
from backend.notifications.mailerlite import (
    API_BASE,
    _client,
    _find_subscriber,
    _headers,
    _payload,
)
from backend.notifications.mailerlite_backfill import BatchAnswer, _refusal
from backend.notifications.mailerlite_field_backfill import Current
from backend.util.posthog_client import get_posthog_client
from backend.util.product_analytics import set_onboarding_role

logger = logging.getLogger(__name__)

# One person every two seconds, a read and a write each: half MailerLite's
# 120 calls a minute, so the live consumer keeps the rest.
WRITE_INTERVAL_SECONDS = 2.0
# The PostHog client drops events once its in-memory queue is full.
POSTHOG_FLUSH_EVERY = 500


class RoleRecord(BaseModel):
    """An account with a role on record, and what MailerLite's rules need."""

    user_id: str
    email: str
    timezone: str | None = None
    marketing_opt_out_at: datetime | None = None
    # The pick the wizard kept, if it kept one.
    choice: str | None = None
    other: str | None = None
    # The copy in the business understanding.
    understanding_role: str | None = None

    @property
    def kept(self) -> bool:
        return bool(self.choice)

    def reliable_role(self) -> OnboardingRole | None:
        if self.choice:
            return OnboardingRole(choice=self.choice, other=self.other)
        return OnboardingRole.from_understanding(self.understanding_role)


class RoleAssignment(BaseModel):
    user_id: str
    email: str
    role: OnboardingRole


class RolePlan(BaseModel):
    accounts: int
    # Reliable: the pick the wizard kept, or an exact option ID.
    kept: int
    exact: int
    # Neither: Other's text or a rewrite, left unset.
    skipped: int
    # Reliable picks by label.
    roles: dict[str, int]
    posthog: list[RoleAssignment]
    # MailerLite subscribers without a role.
    mailerlite: list[RoleAssignment]
    # MailerLite subscribers that already have one, left alone.
    already_set: int
    # Not MailerLite subscribers, never created here.
    not_subscribers: int
    opted_out: int
    excluded_country: int


def plan(records: list[RoleRecord], current: Current) -> RolePlan:
    result = RolePlan(
        accounts=len(records),
        kept=0,
        exact=0,
        skipped=0,
        roles={},
        posthog=[],
        mailerlite=[],
        already_set=0,
        not_subscribers=0,
        opted_out=0,
        excluded_country=0,
    )
    roles: Counter[str] = Counter()
    for record in records:
        role = record.reliable_role()
        if role is None:
            result.skipped += 1
            continue
        if record.kept:
            result.kept += 1
        else:
            result.exact += 1
        roles[role.label] += 1
        assignment = RoleAssignment(
            user_id=record.user_id, email=record.email, role=role
        )
        result.posthog.append(assignment)
        _plan_mailerlite(result, record, assignment, current)
    result.roles = dict(roles)
    return result


def _plan_mailerlite(
    result: RolePlan, record: RoleRecord, assignment: RoleAssignment, current: Current
) -> None:
    if not marketing_allowed(record):
        result.opted_out += 1
        return
    if points_at_excluded_country(email=record.email, timezone=record.timezone):
        result.excluded_country += 1
        return
    held = current.get(record.email.strip().lower())
    if held is None:
        result.not_subscribers += 1
    elif held.get(SubscriberField.ROLE.value):
        result.already_set += 1
    else:
        result.mailerlite.append(assignment)


async def send_to_posthog(assignments: list[RoleAssignment]) -> None:
    """A `$set_once` per person, flushed as it goes."""
    client = get_posthog_client()
    if client is None:
        raise RuntimeError("PostHog is not configured")
    for index, assignment in enumerate(assignments, start=1):
        set_onboarding_role(
            user_id=assignment.user_id, role=assignment.role, keep_existing=True
        )
        if index % POSTHOG_FLUSH_EVERY == 0:
            await asyncio.to_thread(client.flush)
    await asyncio.to_thread(client.flush)


async def apply(
    changes: list[RoleAssignment],
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int, int]:
    """Returns (succeeded, failed, skipped). A failure is logged with
    MailerLite's reason and left for the next run; someone who has gone, or
    has a role by now, is skipped."""
    succeeded = failed = skipped = 0
    for index, change in enumerate(changes):
        if index:
            await asyncio.sleep(WRITE_INTERVAL_SECONDS)
        if on_progress and index and index % 100 == 0:
            on_progress(index, len(changes))
        outcome = await _write(change)
        if outcome is None:
            skipped += 1
        elif outcome:
            succeeded += 1
        else:
            failed += 1
    if on_progress:
        on_progress(len(changes), len(changes))
    return succeeded, failed, skipped


async def _write(change: RoleAssignment) -> bool | None:
    """True once written, False on a failure, None when there is nothing to
    write any more."""
    try:
        subscriber = await _find_subscriber(change.email)
        if subscriber is None or not subscriber.get("id"):
            return None
        if (subscriber.get("fields") or {}).get(SubscriberField.ROLE.value):
            return None
        response = await _client().put(
            f"{API_BASE}/subscribers/{subscriber['id']}",
            headers=_headers(),
            json={"fields": _payload(role_fields(change.role))},
        )
    except Exception:
        logger.warning(
            "Role write failed for %s; the next run retries it",
            _refusal(change.email, BatchAnswer(code=0)),
        )
        return False
    if response.status == 404:
        return None
    if response.status in (200, 201, 202, 204):
        return True
    try:
        body = response.json()
    except Exception:
        body = None
    logger.warning(
        "Role write failed for %s; the next run retries it",
        _refusal(change.email, BatchAnswer(code=response.status, body=body)),
    )
    return False
