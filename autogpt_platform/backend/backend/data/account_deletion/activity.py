"""Stop the account from doing anything while it is being deleted: no new
sign-ins, no scheduled runs, no runs in flight.

The self-service endpoint (SECRT-2877) does these itself before it hands
off; they are no-ops the second time.
"""

import asyncio
import logging
from datetime import datetime, timezone

from prisma.models import AuthSession, AuthUser, User

from backend.data import execution as execution_db
from backend.data.account_deletion.models import StepReport, StepStatus, Subject
from backend.executor.utils import stop_graph_execution
from backend.util.clients import get_scheduler_client

logger = logging.getLogger(__name__)

BAN_REASON = "Account deletion in progress"
OPT_OUT_SOURCE = "account_deletion"
_LIVE_STATUSES = [
    execution_db.ExecutionStatus.INCOMPLETE,
    execution_db.ExecutionStatus.QUEUED,
    execution_db.ExecutionStatus.RUNNING,
]


async def disable_sign_in(subject: Subject, *, dry_run: bool) -> StepReport:
    """Ban the auth identity, which Better Auth checks on every sign-in, and
    end its sessions. An access token already issued stays valid until it
    expires (an hour).

    Also opts the account out of marketing, so a MailerLite write queued by a
    webhook (the subscription cancellation below, say) is dropped instead of
    re-creating the subscriber this deletion forgets."""
    sessions_where = {"userId": subject.user_id}
    if dry_run:
        sessions = await AuthSession.prisma().count(where=sessions_where)
    else:
        if subject.has_auth_identity:
            await AuthUser.prisma().update(
                where={"id": subject.user_id},
                data={"banned": True, "banReason": BAN_REASON, "banExpires": None},
            )
        await User.prisma().update_many(
            where={"id": subject.user_id, "marketingOptOutAt": None},
            data={
                "marketingOptOutAt": datetime.now(timezone.utc),
                "marketingOptOutSource": OPT_OUT_SOURCE,
            },
        )
        sessions = await AuthSession.prisma().delete_many(where=sessions_where)
    return StepReport(
        name="sign_in",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"sessions": sessions},
    )


async def stop_schedules(subject: Subject, *, dry_run: bool) -> StepReport:
    """Agent and AutoPilot schedules, then the per-user system crons (memory
    rebuild, nightly batch, morning briefing). They live in the scheduler's
    job store, which has no foreign key to the account."""
    client = get_scheduler_client()
    schedules = await client.get_execution_schedules(
        user_id=subject.user_id, include_paused=True
    )
    if not dry_run:
        for schedule in schedules:
            await client.delete_schedule(schedule.id, user_id=subject.user_id)
        await client.delete_community_rebuild_schedule(subject.user_id)
        await client.delete_nightly_batch_schedule(subject.user_id)
        await client.remove_morning_briefing_schedule(subject.user_id)
    return StepReport(
        name="schedules",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"schedules": len(schedules)},
        note="system crons are removed on apply, and not counted",
    )


async def stop_executions(subject: Subject, *, dry_run: bool) -> StepReport:
    """Agent runs still queued or running. AutoPilot turns in flight are not
    stopped here: their chat sessions go with the account, and a turn that
    finishes afterwards has nowhere to write."""
    running = await execution_db.get_graph_executions(
        user_id=subject.user_id, statuses=_LIVE_STATUSES
    )
    if not dry_run and running:
        await asyncio.gather(
            *(
                stop_graph_execution(user_id=subject.user_id, graph_exec_id=run.id)
                for run in running
            )
        )
    return StepReport(
        name="executions",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"running": len(running)},
    )
