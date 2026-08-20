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

If both sides managed the same edge we would double-add or fight over
removals, so nothing in this module touches the tour → changelog handoff.
"""

import logging

from backend.util.request import Requests
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

API_BASE = "https://connect.mailerlite.com/api"
_OK_STATUSES = (200, 201, 202, 204)


class MailerLiteNotConfigured(RuntimeError):
    """Raised so a queued job retries once credentials are in place, rather
    than silently reporting success."""


class MailerLiteError(RuntimeError):
    """A MailerLite call failed. Raised so the queued job retries with backoff
    — a MailerLite outage must never fail payment processing, but it must not
    silently drop an enrolment either."""


async def enroll_in_onboarding(email: str) -> None:
    """Add a first-time subscriber to the tour group. Joining the group is the
    automation's trigger; MailerLite sends the six emails from there."""
    await _add_to_group(
        email, settings.config.mailerlite_onboarding_group_id, "onboarding tour"
    )


async def add_to_changelog(email: str) -> None:
    """Returning customers and anyone who predates the tour."""
    await _add_to_group(
        email, settings.config.mailerlite_changelog_group_id, "changelog"
    )


async def remove_from_changelog(email: str) -> None:
    """The day a plan ends."""
    group_id = settings.config.mailerlite_changelog_group_id
    _require_config(group_id, "changelog")

    subscriber_id = await _find_subscriber_id(email)
    if subscriber_id is None:
        logger.info("No MailerLite subscriber for %s; nothing to remove", email)
        return

    response = await _client().delete(
        f"{API_BASE}/subscribers/{subscriber_id}/groups/{group_id}",
        headers=_headers(),
    )
    # 404 means they are already out of the group, which is the desired state.
    if response.status not in _OK_STATUSES and response.status != 404:
        raise MailerLiteError(
            f"Removing {email} from the changelog group failed with "
            f"{response.status}"
        )
    logger.info("Removed %s from the MailerLite changelog group", email)


async def _add_to_group(email: str, group_id: str, description: str) -> None:
    _require_config(group_id, description)
    response = await _client().post(
        f"{API_BASE}/subscribers",
        headers=_headers(),
        json={"email": email, "groups": [group_id]},
    )
    if response.status not in _OK_STATUSES:
        raise MailerLiteError(
            f"Adding {email} to the {description} group failed with {response.status}"
        )
    logger.info("Added %s to the MailerLite %s group", email, description)


async def _find_subscriber_id(email: str) -> str | None:
    response = await _client().get(
        f"{API_BASE}/subscribers/{email}", headers=_headers()
    )
    if response.status == 404:
        return None
    if response.status not in _OK_STATUSES:
        raise MailerLiteError(
            f"Looking up MailerLite subscriber {email} failed with {response.status}"
        )
    return ((response.json() or {}).get("data") or {}).get("id")


def _client() -> Requests:
    # Statuses are inspected rather than raised on, because "already gone" is a
    # success for a removal.
    return Requests(trusted_origins=[API_BASE], raise_for_status=False)


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.secrets.mailerlite_api_token}",
        "Accept": "application/json",
    }


def _require_config(group_id: str, description: str) -> None:
    if not settings.secrets.mailerlite_api_token:
        raise MailerLiteNotConfigured("MAILERLITE_API_TOKEN is not set")
    if not group_id:
        raise MailerLiteNotConfigured(
            f"The MailerLite {description} group ID is not configured"
        )
