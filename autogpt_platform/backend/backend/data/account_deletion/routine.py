"""Delete an account and everything it holds (SECRT-2854).

Two phases, with an undo window between them:

- `lock_account` stops the account: no sign-in, no schedules, no runs, no
  subscription. It records when, and `unlock_account` undoes what it can.
- `delete_account` runs once `UNDO_WINDOW` has passed since the lock (or with
  `force`). It repeats the stop steps, then hands the agents other accounts
  still use to the "deleted user" account, removes what lives outside the
  account's rows (marketplace, files, memory, PostHog, MailerLite), and its
  rows last. The rows go last because they are how every earlier step finds
  its data, so a step that fails leaves the account in place for a rerun to
  finish. An ID with no account left is refused unless `force`; with it, a
  rerun of a finished deletion finds nothing and changes nothing.
"""

import logging
from collections.abc import Awaitable, Callable

from backend.data.account_deletion.activity import (
    disable_sign_in,
    stop_executions,
    stop_schedules,
)
from backend.data.account_deletion.database import delete_rows
from backend.data.account_deletion.integrations import release_integrations
from backend.data.account_deletion.lock import lift_lock, record_lock
from backend.data.account_deletion.marketplace import delete_marketplace
from backend.data.account_deletion.memory import delete_memory
from backend.data.account_deletion.models import (
    AccountDeletionBlocked,
    Blocker,
    DeletionReport,
    StepReport,
    StepStatus,
    Subject,
)
from backend.data.account_deletion.preflight import (
    find_blockers,
    invalid_id,
    load_subject,
    system_account,
)
from backend.data.account_deletion.shared_agents import (
    plan_transfer,
    transfer_shared_agents,
)
from backend.data.account_deletion.storage import delete_files
from backend.data.account_deletion.subscriptions import cancel_subscriptions
from backend.data.account_deletion.third_parties import (
    delete_posthog_person,
    forget_mailerlite_subscriber,
)

logger = logging.getLogger(__name__)

Step = Callable[..., Awaitable[StepReport]]


async def lock_account(
    user_id: str, *, dry_run: bool = True, email: str | None = None
) -> DeletionReport:
    """Stop the account and start the undo window. Deletes no data. Raises
    `AccountDeletionBlocked` for the placeholder or an unknown account."""
    subject = await load_subject(user_id, email)
    return await _run(subject, _lock_steps(), _lockable(subject), dry_run)


async def unlock_account(user_id: str, *, dry_run: bool = True) -> DeletionReport:
    """Let a locked account back in: its ban and the marketing opt-out the
    lock set are lifted, and the lock cleared. Subscriptions the lock
    cancelled stay cancelled, and deleted schedules are not restored."""
    subject = await load_subject(user_id)
    steps: list[tuple[str, Step]] = [("unlock", lift_lock)]
    return await _run(subject, steps, _lockable(subject), dry_run)


async def delete_account(
    user_id: str,
    *,
    dry_run: bool = True,
    email: str | None = None,
    force: bool = False,
) -> DeletionReport:
    """Delete the account, or with `dry_run` (the default) only count what
    would go. `email` finds the MailerLite subscriber of an account whose rows
    are already gone. `force` skips the undo window (no lock, or a lock less
    than `UNDO_WINDOW` old), and lets an ID with no account left through, to
    finish what a deletion left; other blockers still stop it.

    Raises `AccountDeletionBlocked` before deleting anything when a blocker
    needs a decision; a dry run reports blockers instead. Returns the report;
    `report.complete` says whether everything is gone."""
    subject = await load_subject(user_id, email)
    plan = await plan_transfer(user_id)
    subject.shared_graph_ids = plan.graph_ids
    subject.kept_listing_ids = plan.listing_ids
    blockers = await find_blockers(subject, force=force)
    return await _run(subject, _deletion_steps(), blockers, dry_run)


async def _run(
    subject: Subject,
    steps: list[tuple[str, Step]],
    blockers: list[Blocker],
    dry_run: bool,
) -> DeletionReport:
    report = DeletionReport(
        user_id=subject.user_id,
        dry_run=dry_run,
        account_found=subject.has_user_row or subject.has_auth_identity,
        blockers=blockers,
    )
    if report.blockers and not dry_run:
        raise AccountDeletionBlocked(report)
    for name, step in steps:
        result = await _run_step(name, step, subject, dry_run)
        report.steps.append(result)
        if result.status == StepStatus.FAILED and not dry_run:
            break
    logger.info(
        f"Account deletion ({', '.join(name for name, _ in steps)}) of "
        f"{subject.user_id}: dry_run={dry_run}, complete={report.complete}"
    )
    return report


def _lockable(subject: Subject) -> list[Blocker]:
    if refused := invalid_id(subject.user_id):
        return refused
    if subject.has_user_row or subject.has_auth_identity:
        return system_account(subject.user_id)
    return [
        Blocker(
            name="account_not_found",
            count=0,
            resolution="No platform account or auth identity has this ID.",
        )
    ]


def _stop_steps() -> list[tuple[str, Step]]:
    return [
        ("sign_in", disable_sign_in),
        ("schedules", stop_schedules),
        ("executions", stop_executions),
        ("subscriptions", cancel_subscriptions),
    ]


def _lock_steps() -> list[tuple[str, Step]]:
    return [*_stop_steps(), ("lock", record_lock)]


def _deletion_steps() -> list[tuple[str, Step]]:
    return [
        *_stop_steps(),
        ("integrations", release_integrations),
        ("shared_agents", transfer_shared_agents),
        ("marketplace", delete_marketplace),
        ("files", delete_files),
        ("memory", delete_memory),
        ("posthog", delete_posthog_person),
        ("mailerlite", forget_mailerlite_subscriber),
        ("database", delete_rows),
    ]


async def _run_step(
    name: str, step: Step, subject: Subject, dry_run: bool
) -> StepReport:
    """One step. A failure is reported by its exception type only: messages
    can carry an email or a path, and the report is printed and logged."""
    try:
        return await step(subject, dry_run=dry_run)
    except Exception as e:
        logger.exception(f"Account deletion step {name} failed for {subject.user_id}")
        return StepReport(
            name=name,
            status=StepStatus.FAILED,
            note=f"{type(e).__name__}; the log has the details. Run again to "
            "retry from here.",
        )
