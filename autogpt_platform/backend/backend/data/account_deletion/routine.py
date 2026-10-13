"""Delete an account and everything it holds (SECRT-2854).

`delete_account` runs idempotent steps in a safe order: first stop the
account (sign-in, schedules, runs, billing, integrations), then remove what
lives outside its rows (marketplace, files, memory, PostHog, MailerLite), and
its rows last. The rows go last because they are how every earlier step finds
its data, so a step that fails leaves the account in place for a rerun to
finish. A rerun of a finished deletion finds nothing and changes nothing.
"""

import logging
from collections.abc import Awaitable, Callable
from functools import partial

from backend.data.account_deletion.activity import (
    disable_sign_in,
    stop_executions,
    stop_schedules,
)
from backend.data.account_deletion.database import delete_rows
from backend.data.account_deletion.integrations import release_integrations
from backend.data.account_deletion.marketplace import delete_marketplace
from backend.data.account_deletion.memory import delete_memory
from backend.data.account_deletion.models import (
    AccountDeletionBlocked,
    DeletionReport,
    StepReport,
    StepStatus,
    Subject,
)
from backend.data.account_deletion.preflight import (
    find_blockers,
    load_subject,
    shared_agent_counts,
)
from backend.data.account_deletion.storage import delete_files
from backend.data.account_deletion.subscriptions import cancel_subscriptions
from backend.data.account_deletion.third_parties import (
    delete_posthog_person,
    forget_mailerlite_subscriber,
)

logger = logging.getLogger(__name__)

Step = Callable[..., Awaitable[StepReport]]


async def delete_account(
    user_id: str,
    *,
    dry_run: bool = True,
    email: str | None = None,
    remove_shared_agents: bool = False,
) -> DeletionReport:
    """Delete the account, or with `dry_run` (the default) only count what
    would go. `email` finds the MailerLite subscriber of an account whose rows
    are already gone. `remove_shared_agents` lets the deletion take this
    account's agents out of other accounts' libraries, which is otherwise a
    blocker.

    Raises `AccountDeletionBlocked` before deleting anything when a blocker
    needs a decision; a dry run reports blockers instead. Returns the report;
    `report.complete` says whether everything is gone."""
    subject = await load_subject(user_id, email)
    shared = await shared_agent_counts(user_id)
    report = DeletionReport(
        user_id=user_id,
        dry_run=dry_run,
        account_found=subject.has_user_row or subject.has_auth_identity,
        blockers=await find_blockers(
            subject, shared, remove_shared_agents=remove_shared_agents
        ),
    )
    if report.blockers and not dry_run:
        raise AccountDeletionBlocked(report)
    marketplace = partial(
        delete_marketplace,
        shared_agent_rows=shared.library_entries + shared.presets,
        remove_shared_agents=remove_shared_agents,
    )
    for name, step in _steps(marketplace):
        result = await _run_step(name, step, subject, dry_run)
        report.steps.append(result)
        if result.status == StepStatus.FAILED and not dry_run:
            break
    logger.info(
        f"Account deletion of {user_id}: dry_run={dry_run}, "
        f"complete={report.complete}"
    )
    return report


def _steps(marketplace: Step) -> list[tuple[str, Step]]:
    return [
        ("sign_in", disable_sign_in),
        ("schedules", stop_schedules),
        ("executions", stop_executions),
        ("subscriptions", cancel_subscriptions),
        ("integrations", release_integrations),
        ("marketplace", marketplace),
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
