"""The account outside our own systems: its PostHog person and its MailerLite
subscriber.

Stripe is left alone on purpose: the customer, invoices and payments are the
billing records accounting keeps. Langfuse traces expire on their own after
90 days, so nothing is sent there.
"""

import logging
from urllib.parse import urlencode

from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.notifications import mailerlite
from backend.util.request import Requests
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_OK_STATUSES = (200, 201, 202, 204)


async def delete_posthog_person(subject: Subject, *, dry_run: bool) -> StepReport:
    """Delete the person whose distinct ID is the user ID, with their events
    and recordings. PostHog queues the event deletion for its weekly run."""
    secrets = Settings().secrets
    if not (secrets.posthog_personal_api_key and secrets.posthog_project_id):
        return StepReport(
            name="posthog",
            status=StepStatus.SKIPPED,
            note="POSTHOG_PERSONAL_API_KEY or POSTHOG_PROJECT_ID is not set; "
            "delete the person by hand",
        )
    persons = await _count_posthog_persons(subject.user_id)
    if not dry_run and persons:
        await _bulk_delete_posthog_person(subject.user_id)
    return StepReport(
        name="posthog",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"persons": persons},
    )


async def forget_mailerlite_subscriber(
    subject: Subject, *, dry_run: bool
) -> StepReport:
    if not mailerlite.configured():
        return StepReport(
            name="mailerlite",
            status=StepStatus.SKIPPED,
            note="MAILERLITE_API_TOKEN is not set",
        )
    if not subject.email:
        return StepReport(
            name="mailerlite",
            status=StepStatus.SKIPPED,
            note="no email on record for this account; look the subscriber up "
            "by hand",
        )
    if dry_run:
        found = await mailerlite.find_subscriber_id(subject.email) is not None
    else:
        found = await mailerlite.forget_subscriber(subject.email)
    return StepReport(
        name="mailerlite",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"subscribers": int(found)},
    )


async def _count_posthog_persons(user_id: str) -> int:
    response = await _posthog().get(
        f"{_posthog_project_url()}/persons/?{urlencode({'distinct_id': user_id})}",
        headers=_posthog_headers(),
    )
    _raise_for_posthog(response.status, "looking up")
    return len((response.json() or {}).get("results") or [])


async def _bulk_delete_posthog_person(user_id: str) -> None:
    response = await _posthog().post(
        f"{_posthog_project_url()}/persons/bulk_delete/",
        json={
            "distinct_ids": [user_id],
            "delete_events": True,
            "delete_recordings": True,
        },
        headers=_posthog_headers(),
    )
    _raise_for_posthog(response.status, "deleting")
    logger.info(f"Deleted the PostHog person of {user_id}")


def _raise_for_posthog(status: int, action: str) -> None:
    if status not in _OK_STATUSES:
        raise RuntimeError(f"PostHog failed {action} the person with {status}")


def _posthog() -> Requests:
    return Requests(
        trusted_origins=[Settings().secrets.posthog_app_host],
        raise_for_status=False,
    )


def _posthog_project_url() -> str:
    secrets = Settings().secrets
    host = secrets.posthog_app_host.rstrip("/")
    return f"{host}/api/projects/{secrets.posthog_project_id}"


def _posthog_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {Settings().secrets.posthog_personal_api_key}"}
