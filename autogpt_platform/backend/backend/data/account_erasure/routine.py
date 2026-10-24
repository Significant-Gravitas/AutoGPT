"""Erase a closed account and everything it holds (SECRT-2854).

Closing comes first, and is not done here: Settings and the paywall close an
account (`account_deletion.service.delete_account`, SECRT-2877), and so does
the CLI for a request made by email (`request.close_for_email_request`).
Closing cancels the account's Stripe plan, disconnects its apps, stops its
work and email, bans its sign-in and writes its `AccountDeletionRequest`.

`erase_account` then removes the data: it reruns the work stop, sends the
confirmation closing could not, cancels the subscriptions of the orgs that go
with the account, hands the agents other accounts still use to the "deleted
user" account, removes what lives outside the account's rows (marketplace,
files, memory, PostHog, MailerLite) and its rows last, then marks the request
done. The rows go last because they are how every earlier step finds its
data, so a step that fails leaves the account in place for a rerun to
finish; a rerun of a finished erasure finds nothing and changes nothing.
"""

import logging
from collections.abc import Awaitable, Callable

from backend.data.account_erasure.database import delete_rows
from backend.data.account_erasure.marketplace import delete_marketplace
from backend.data.account_erasure.memory import delete_memory
from backend.data.account_erasure.models import (
    AccountErasureBlocked,
    Blocker,
    ErasureReport,
    StepReport,
    StepStatus,
    Subject,
)
from backend.data.account_erasure.preflight import find_blockers, load_subject
from backend.data.account_erasure.request import complete_request, send_confirmation
from backend.data.account_erasure.shared_agents import (
    plan_transfer,
    transfer_shared_agents,
)
from backend.data.account_erasure.storage import delete_files
from backend.data.account_erasure.subscriptions import cancel_org_subscriptions
from backend.data.account_erasure.third_parties import (
    delete_posthog_person,
    forget_mailerlite_subscriber,
)
from backend.data.account_erasure.work import stop_work

logger = logging.getLogger(__name__)

Step = Callable[..., Awaitable[StepReport]]


async def erase_account(
    user_id: str,
    *,
    dry_run: bool = True,
    email: str | None = None,
    closing: bool = False,
    allow_banned: bool = False,
) -> ErasureReport:
    """Erase the account, or with `dry_run` (the default) only count what
    would go. `email` finds the MailerLite subscriber of an account whose rows
    are already gone. `closing` says the caller closes the account first (a
    request made by email), so a dry run is not refused for having no
    request yet, and `allow_banned` lets it close an account banned for
    another reason.

    Raises `AccountErasureBlocked` before deleting anything when a blocker
    needs a decision; a dry run reports blockers instead. Returns the report;
    `report.complete` says whether everything is gone."""
    subject = await load_subject(user_id, email)
    plan = await plan_transfer(user_id)
    subject.shared_graph_ids = plan.graph_ids
    subject.kept_listing_ids = plan.listing_ids
    blockers = await find_blockers(subject, closing=closing, allow_banned=allow_banned)
    return await _run(subject, _erasure_steps(), blockers, dry_run)


async def _run(
    subject: Subject,
    steps: list[tuple[str, Step]],
    blockers: list[Blocker],
    dry_run: bool,
) -> ErasureReport:
    report = ErasureReport(
        user_id=subject.user_id,
        dry_run=dry_run,
        account_found=subject.has_user_row or subject.has_auth_identity,
        blockers=blockers,
    )
    if report.blockers and not dry_run:
        raise AccountErasureBlocked(report)
    for name, step in steps:
        result = await _run_step(name, step, subject, dry_run)
        report.steps.append(result)
        if result.status == StepStatus.FAILED and not dry_run:
            break
    logger.info(
        f"Account erasure of {subject.user_id}: dry_run={dry_run}, "
        f"complete={report.complete}"
    )
    return report


def _erasure_steps() -> list[tuple[str, Step]]:
    return [
        ("stop_work", stop_work),
        ("confirmation", send_confirmation),
        ("org_subscriptions", cancel_org_subscriptions),
        ("shared_agents", transfer_shared_agents),
        ("marketplace", delete_marketplace),
        ("files", delete_files),
        ("memory", delete_memory),
        ("posthog", delete_posthog_person),
        ("mailerlite", forget_mailerlite_subscriber),
        ("database", delete_rows),
        ("request", complete_request),
    ]


async def _run_step(
    name: str, step: Step, subject: Subject, dry_run: bool
) -> StepReport:
    """One step. A failure is reported by its exception type only: messages
    can carry an email or a path, and the report is printed and logged."""
    try:
        return await step(subject, dry_run=dry_run)
    except Exception as e:
        logger.exception(f"Account erasure step {name} failed for {subject.user_id}")
        return StepReport(
            name=name,
            status=StepStatus.FAILED,
            note=f"{type(e).__name__}; the log has the details. Run again to "
            "retry from here.",
        )
