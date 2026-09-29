"""Put existing Stripe customers where the live audience code would have.

The lifecycle handlers only react to new Stripe events, so everyone who
subscribed before they shipped sits in no MailerLite group. This works out,
per customer, the membership those handlers would have produced, and applies
only the difference.

The one-owner-per-transition rule in `mailerlite.py` holds here too. Anyone in
the tour group is left alone: they are either mid-tour, and must not get the
changelog yet, or they finished it and MailerLite's automation has already
moved them across.

Idempotent by construction: current membership is read first, so a second run
finds nothing to do and a failed call is simply picked up by the next run.
"""

import asyncio
import logging
from enum import Enum
from urllib.parse import urlencode

from pydantic import BaseModel

from backend.notifications.mailerlite import (
    API_BASE,
    MailerLiteError,
    _client,
    _headers,
    _pseudonym,
    _require_config,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# MailerLite accepts at most 50 calls per batch. A batch made only of
# subscriber upserts is processed as an import, which has its own limit of 5
# per minute on top of the global 120 per minute.
BATCH_SIZE = 50
UPSERT_BATCH_INTERVAL_SECONDS = 13.0
OTHER_BATCH_INTERVAL_SECONDS = 1.0
PAGE_SIZE = 1000
MEMBER_STATUSES = ("active", "unsubscribed", "unconfirmed", "bounced", "junk")

_PAYING = {"active", "past_due"}
_ENDED = {"canceled", "incomplete_expired"}


class Standing(str, Enum):
    PAYING = "paying"
    TRIALING = "trialing"
    CHURNED = "churned"
    # incomplete, unpaid, paused: no lifecycle event has settled these yet.
    UNSETTLED = "unsettled"


class Decision(str, Enum):
    ADD_CHANGELOG = "add_changelog"
    ADD_TRIAL = "add_trial"
    REMOVE_CHANGELOG = "remove_changelog"
    REMOVE_TRIAL = "remove_trial"
    SKIP_IN_TOUR = "skip_in_tour"
    SKIP_NO_TRIAL_GROUP = "skip_no_trial_group"
    SKIP_UNSETTLED = "skip_unsettled"
    ALREADY_CORRECT = "already_correct"


CHANGES = (
    Decision.ADD_CHANGELOG,
    Decision.ADD_TRIAL,
    Decision.REMOVE_CHANGELOG,
    Decision.REMOVE_TRIAL,
)


class Customer(BaseModel):
    """A platform account and the statuses of all its Stripe subscriptions."""

    user_id: str
    email: str
    statuses: list[str]


class Audience(BaseModel):
    """Current membership of each group, keyed by lowercased email, valued by
    MailerLite subscriber ID."""

    tour: dict[str, str]
    changelog: dict[str, str]
    trial: dict[str, str]


class PlannedChange(BaseModel):
    customer: Customer
    standing: Standing
    decisions: list[Decision]


class ApplyResult(BaseModel):
    succeeded: dict[Decision, int]
    failed: dict[Decision, int]


def classify(statuses: list[str]) -> Standing:
    """A customer who pays on any subscription is a subscriber, whatever the
    state of their others."""
    found = set(statuses)
    if found & _PAYING:
        return Standing.PAYING
    if "trialing" in found:
        return Standing.TRIALING
    if found and found <= _ENDED:
        return Standing.CHURNED
    return Standing.UNSETTLED


def decide(customer: Customer, audience: Audience, trial_enabled: bool) -> PlannedChange:
    standing = classify(customer.statuses)
    email = customer.email.strip().lower()
    in_tour = email in audience.tour
    in_changelog = email in audience.changelog
    in_trial = email in audience.trial

    decisions: list[Decision] = []
    if standing is not Standing.TRIALING and in_trial:
        decisions.append(Decision.REMOVE_TRIAL)
    if standing is Standing.PAYING and not in_changelog:
        decisions.append(Decision.SKIP_IN_TOUR if in_tour else Decision.ADD_CHANGELOG)
    elif standing is Standing.TRIALING and not in_trial:
        decisions.append(
            Decision.ADD_TRIAL if trial_enabled else Decision.SKIP_NO_TRIAL_GROUP
        )
    elif standing is Standing.CHURNED and in_changelog:
        decisions.append(Decision.REMOVE_CHANGELOG)
    elif standing is Standing.UNSETTLED:
        decisions.append(Decision.SKIP_UNSETTLED)

    return PlannedChange(
        customer=customer,
        standing=standing,
        decisions=decisions or [Decision.ALREADY_CORRECT],
    )


def plan(customers: list[Customer], audience: Audience) -> list[PlannedChange]:
    trial_enabled = bool(settings.config.mailerlite_trial_group_id)
    return [decide(c, audience, trial_enabled) for c in customers]


async def read_audience() -> Audience:
    """Read every group this touches, in every subscriber status: an
    unsubscribed member is still a member, and must not be re-added."""
    config = settings.config
    _require_config(config.mailerlite_onboarding_group_id, "onboarding tour")
    _require_config(config.mailerlite_changelog_group_id, "changelog")
    trial_group = config.mailerlite_trial_group_id
    return Audience(
        tour=await _read_group(config.mailerlite_onboarding_group_id),
        changelog=await _read_group(config.mailerlite_changelog_group_id),
        trial=await _read_group(trial_group) if trial_group else {},
    )


async def apply(changes: list[PlannedChange], audience: Audience) -> ApplyResult:
    result = ApplyResult(
        succeeded={d: 0 for d in CHANGES}, failed={d: 0 for d in CHANGES}
    )
    batches = [
        (decision, due[start : start + BATCH_SIZE])
        for decision in CHANGES
        for due in [[c for c in changes if decision in c.decisions]]
        for start in range(0, len(due), BATCH_SIZE)
    ]
    for index, (decision, chunk) in enumerate(batches):
        if index:
            await asyncio.sleep(_interval_before(decision))
        codes = await _send_batch(
            [_call_for(decision, c.customer.email, audience) for c in chunk]
        )
        for change, code in zip(chunk, codes):
            _record(result, decision, change, code)
    return result


def _interval_before(decision: Decision) -> float:
    if decision in (Decision.ADD_CHANGELOG, Decision.ADD_TRIAL):
        return UPSERT_BATCH_INTERVAL_SECONDS
    return OTHER_BATCH_INTERVAL_SECONDS


def _call_for(decision: Decision, email: str, audience: Audience) -> dict:
    """The same call the live handler makes for this change."""
    config = settings.config
    if decision is Decision.ADD_CHANGELOG:
        group = config.mailerlite_changelog_group_id
        return {"method": "POST", "path": "api/subscribers", "body": _upsert(email, group)}
    if decision is Decision.ADD_TRIAL:
        group = config.mailerlite_trial_group_id
        return {"method": "POST", "path": "api/subscribers", "body": _upsert(email, group)}
    key = email.strip().lower()
    if decision is Decision.REMOVE_CHANGELOG:
        subscriber, group = audience.changelog[key], config.mailerlite_changelog_group_id
    else:
        subscriber, group = audience.trial[key], config.mailerlite_trial_group_id
    return {"method": "DELETE", "path": f"api/subscribers/{subscriber}/groups/{group}"}


def _upsert(email: str, group_id: str) -> dict:
    return {"email": email, "groups": [group_id]}


def _record(
    result: ApplyResult, decision: Decision, change: PlannedChange, code: int
) -> None:
    # 404 on a removal means they already left, which is the desired state.
    ok = code in (200, 201, 202, 204) or (
        code == 404 and decision in (Decision.REMOVE_CHANGELOG, Decision.REMOVE_TRIAL)
    )
    if ok:
        result.succeeded[decision] += 1
        return
    result.failed[decision] += 1
    logger.warning(
        f"{decision.value} failed for {_pseudonym(change.customer.email)} "
        f"with {code}; the next run retries it"
    )


async def _send_batch(requests: list[dict]) -> list[int]:
    """One /batch call. Requests retries a 429 with backoff before this sees
    it, so a status here is final."""
    response = await _client().post(
        f"{API_BASE}/batch", headers=_headers(), json={"requests": requests}
    )
    if response.status != 200:
        raise MailerLiteError(f"MailerLite batch failed with {response.status}")
    responses = (response.json() or {}).get("responses") or []
    if len(responses) != len(requests):
        raise MailerLiteError(
            f"MailerLite answered {len(responses)} of {len(requests)} batched calls"
        )
    return [int(r.get("code") or 0) for r in responses]


async def _read_group(group_id: str) -> dict[str, str]:
    members: dict[str, str] = {}
    for status in MEMBER_STATUSES:
        cursor: str | None = None
        while True:
            params = {"limit": str(PAGE_SIZE), "filter[status]": status}
            if cursor:
                params["cursor"] = cursor
            page = await _get_page(group_id, params)
            for row in page.get("data") or []:
                members[str(row["email"]).strip().lower()] = str(row["id"])
            cursor = (page.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
    return members


async def _get_page(group_id: str, params: dict[str, str]) -> dict:
    response = await _client().get(
        f"{API_BASE}/groups/{group_id}/subscribers?{urlencode(params)}",
        headers=_headers(),
    )
    if response.status != 200:
        raise MailerLiteError(
            f"Reading MailerLite group {group_id} failed with {response.status}"
        )
    return response.json() or {}
