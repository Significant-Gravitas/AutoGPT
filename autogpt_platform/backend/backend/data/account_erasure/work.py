"""Stop the account's work again before its data goes.

Closing the account already ran `stop_user_work` (SECRT-2877), but a schedule
or trigger whose removal failed then is not retried, and neither the
scheduler nor the webhook ingress checks for a closed account, so the erasure
reruns it (`account_deletion/service.py` asks for this).

Closing leaves the memory crons (nightly batch, community rebuild, skill
learning) of the account and of each of its experts registered, so they
resume if the owner keeps the account; `is_user_refused` stops them doing
anything meanwhile. The erasure removes them.
"""

from backend.api.features.account_deletion.scheduled_work import stop_user_work
from backend.copilot.graphiti.scope import MemoryScope
from backend.data import execution as execution_db
from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.util.clients import get_scheduler_client

_UNFINISHED = [
    execution_db.ExecutionStatus.INCOMPLETE,
    execution_db.ExecutionStatus.QUEUED,
    execution_db.ExecutionStatus.RUNNING,
]


async def stop_work(subject: Subject, *, dry_run: bool) -> StepReport:
    """A dry run counts the schedules and unfinished runs left; an apply
    reruns `stop_user_work`, which never raises and logs what it could not
    stop."""
    if not dry_run:
        await stop_user_work(subject.user_id)
        # stop_user_work never raises: a scheduler it couldn't reach is only
        # logged. Listing again turns that into a failed step to retry.
        left = await get_scheduler_client().get_execution_schedules(
            user_id=subject.user_id, include_paused=True
        )
        if left:
            raise RuntimeError(f"{len(left)} schedules outlived stop_user_work")
        removed = await remove_memory_crons(subject)
        return StepReport(
            name="stop_work", status=StepStatus.DONE, counts={"memory_crons": removed}
        )
    schedules = await get_scheduler_client().get_execution_schedules(
        user_id=subject.user_id, include_paused=True
    )
    runs = await execution_db.get_graph_executions(
        user_id=subject.user_id, statuses=_UNFINISHED
    )
    return StepReport(
        name="stop_work",
        status=StepStatus.WOULD_DO,
        counts={"schedules": len(schedules), "unfinished_runs": len(runs)},
        note="the apply reruns stop_user_work (schedules, triggers, chat "
        "links, AutoPilot turns and runs) and removes the memory crons",
    )


async def remove_memory_crons(subject: Subject) -> int:
    """Remove the memory crons of the account and of each expert it owns; how
    many went. A scheduler that can't be reached fails the step, so the
    erasure stops while the expert IDs that name the crons are still known."""
    scopes = [MemoryScope.for_user(subject.user_id)] + [
        MemoryScope.for_expert(subject.user_id, expert_id)
        for expert_id in subject.expert_ids
    ]
    scheduler = get_scheduler_client()
    removed = [await scheduler.remove_scope_memory_jobs(scope=s) for s in scopes]
    return sum(len(ids) for ids in removed)
