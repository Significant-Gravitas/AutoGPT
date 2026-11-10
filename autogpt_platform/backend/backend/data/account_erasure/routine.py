"""Erase a closed account and everything it holds (SECRT-2854).

Closing comes first, and is not done here: Settings and the paywall close an
account (`account_deletion.service.delete_account`, SECRT-2877), and so does
the CLI for a request made by email (`request.close_for_email_request`).
Closing cancels the account's Stripe plan, disconnects its apps, stops its
work and email and writes its `AccountDeletionRequest`. It does not ban the
sign-in: for 14 days the owner can sign in and keep the account.

Once those 14 days are over, `erase_account` claims the request
(`request.claim_request`), so a late Keep is refused, and then removes the
data: it reruns the work stop and removes the memory crons, sends the
confirmation closing could not, cancels the subscriptions of the orgs that go
with the account, hands the agents other accounts still use to the "deleted
user" account, removes what lives outside the account's rows (marketplace,
files, memory, PostHog, MailerLite), keeps a ban record if the account is
banned, and deletes its rows last, then marks the request done. The rows go last because they are how every earlier step finds its
data, so a step that fails leaves the account in place for a rerun to
finish; a rerun of a finished erasure finds nothing and changes nothing.
"""

import logging
import traceback
from collections.abc import Awaitable, Callable
from functools import partial

from backend.data.account_erasure.ban_record import keep_ban_record
from backend.data.account_erasure.database import delete_rows
from backend.data.account_erasure.lease import (
    acquire_erasure_lease,
    keep_erasure_lease,
    release_erasure_lease,
)
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
from backend.data.account_erasure.request import (
    claim_request,
    complete_request,
    send_confirmation,
)
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
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

Step = Callable[..., Awaitable[StepReport]]

# The outside service each step needs. A step skips only when its service is
# not configured; an apply then stops there, before the rows go (they hold the
# address MailerLite needs and the expert IDs that name the memory graphs), so
# the request stays open until someone makes sure that data is gone. A stack
# that doesn't use a service says so in `account_erasure_unused_services`.
_SERVICE_OF_STEP = {
    "org_subscriptions": "stripe",
    "files": "buckets",
    "memory": "falkordb",
    "posthog": "posthog",
    "mailerlite": "mailerlite",
}

KeepAlive = Callable[[], Awaitable[bool]]

# Steps whose exception messages can't quote the account's address, so their
# tracebacks are logged as text. The rest (the confirmation email, the
# MailerLite lookup by address) log the exception type only. No step logs with
# exc_info: Sentry would attach the frame locals, and `subject` holds the
# address.
_TRACEBACK_STEPS = {
    "stop_work",
    "org_subscriptions",
    "shared_agents",
    "marketplace",
    "files",
    "memory",
    "posthog",
    "ban_record",
    "database",
    "request",
}

# `report.not_claimed` for a request an earlier run finished.
ERASED_ALREADY = "erased already"


async def erase_account(
    user_id: str,
    *,
    dry_run: bool = True,
    email: str | None = None,
) -> ErasureReport:
    """Erase the account, or with `dry_run` (the default) only count what
    would go. `email` finds the MailerLite subscriber of an account whose rows
    are already gone.

    Raises `AccountErasureBlocked` before deleting anything when a blocker
    needs a decision; a dry run reports blockers instead. An apply whose
    request can't be claimed, or that another run is erasing, runs no step
    (`report.not_claimed`). Returns the report; `report.complete` says whether
    everything is gone."""
    if dry_run:
        return await _erase(user_id, email, dry_run=True)
    lease = await acquire_erasure_lease(user_id)
    if lease is None:
        return ErasureReport(
            user_id=user_id, dry_run=False, not_claimed="another run is erasing it"
        )
    try:
        return await _erase(
            user_id, email, dry_run=False, keepalive=partial(keep_erasure_lease, lease)
        )
    finally:
        await release_erasure_lease(lease)


async def _erase(
    user_id: str,
    email: str | None,
    *,
    dry_run: bool,
    keepalive: KeepAlive | None = None,
) -> ErasureReport:
    subject = await load_subject(user_id, email)
    plan = await plan_transfer(user_id)
    subject.shared_graph_ids = plan.graph_ids
    subject.kept_listing_ids = plan.listing_ids
    blockers = await find_blockers(subject)
    return await _run(subject, _erasure_steps(), blockers, dry_run, keepalive)


async def _run(
    subject: Subject,
    steps: list[tuple[str, Step]],
    blockers: list[Blocker],
    dry_run: bool,
    keepalive: KeepAlive | None = None,
) -> ErasureReport:
    report = ErasureReport(
        user_id=subject.user_id,
        dry_run=dry_run,
        account_found=subject.has_user_row or subject.has_auth_identity,
        blockers=blockers,
    )
    if report.blockers and not dry_run:
        raise AccountErasureBlocked(report)
    if not dry_run and not await _claim(subject, report):
        return report
    for name, step in steps:
        if keepalive is not None and not await keepalive():
            # Another run may be on this account now: leave it to that one.
            report.steps.append(
                StepReport(
                    name=name,
                    status=StepStatus.FAILED,
                    note="lost the erasure lease before this step; the next "
                    "run resumes",
                )
            )
            break
        result = await _run_step(name, step, subject, dry_run)
        if not dry_run:
            result = _stop_on_a_missing_service(result)
        report.steps.append(result)
        if result.status == StepStatus.FAILED and not dry_run:
            break
    logger.info(
        f"Account erasure of {subject.user_id}: dry_run={dry_run}, "
        f"complete={report.complete}"
    )
    return report


async def _claim(subject: Subject, report: ErasureReport) -> bool:
    """Claim the request before any step, so Keep can no longer win. Without
    a claim nothing runs: the owner kept the account in the meantime, or the
    erasure finished on an earlier run."""
    subject.request_id = await claim_request(subject.user_id)
    if subject.request_id is not None:
        return True
    report.not_claimed = (
        ERASED_ALREADY
        if subject.request_completed
        else "kept by its owner, or its 14 days are not over"
    )
    logger.info(f"Account erasure of {subject.user_id}: not claimed")
    return False


def _stop_on_a_missing_service(result: StepReport) -> StepReport:
    """A step skipped for want of its settings fails the apply, so the request
    is never marked done with that service's data left behind, unless the
    stack doesn't use the service."""
    service = _SERVICE_OF_STEP.get(result.name)
    if result.status != StepStatus.SKIPPED or service is None:
        return result
    if service in Settings().config.account_erasure_unused_services:
        return result
    return result.model_copy(
        update={
            "status": StepStatus.FAILED,
            "note": f"{result.note}. Set it up and run again, or delete this "
            f'data by hand and run with "{service}" in '
            "ACCOUNT_ERASURE_UNUSED_SERVICES.",
        }
    )


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
        ("ban_record", keep_ban_record),
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
        trace = (
            "\n" + "".join(traceback.format_exception(e))
            if name in _TRACEBACK_STEPS
            else ""
        )
        logger.error(
            f"Account erasure step {name} failed for {subject.user_id}: "
            f"{type(e).__name__}{trace}"
        )
        return StepReport(
            name=name,
            status=StepStatus.FAILED,
            note=f"{type(e).__name__}; the log has the details. Run again to "
            "retry from here.",
        )
