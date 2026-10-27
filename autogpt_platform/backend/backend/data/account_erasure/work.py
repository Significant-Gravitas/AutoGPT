"""Stop the account's work again before its data goes.

Closing the account already ran `stop_user_work` (SECRT-2877), but a schedule
or trigger whose removal failed then is not retried, and neither the
scheduler nor the webhook ingress checks for a closed account, so the erasure
reruns it (`account_deletion/service.py` asks for this).
"""

from backend.api.features.account_deletion.scheduled_work import stop_user_work
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
        return StepReport(name="stop_work", status=StepStatus.DONE)
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
        note="the apply reruns stop_user_work: schedules, triggers, chat "
        "links, AutoPilot turns and runs",
    )
