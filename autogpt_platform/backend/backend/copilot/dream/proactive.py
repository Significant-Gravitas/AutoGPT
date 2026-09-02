"""Dream proactive pass: propose one small task per idle expert.

Runs after a dream pass applies (see ``orchestrator.py``). For every hired,
non-archived, non-paused expert with no open DelegatedTask, it opens one
DREAM-created task whose behaviour follows ``Expert.autonomyLevel``:

* ``SUGGEST`` — proposal only. The task is QUEUED with acceptance PENDING
  and the pass never starts any execution for it; it exists so the morning
  briefing and Tasks tab can surface the suggestion for the user to accept.
* ``ASK_FIRST`` / ``AUTONOMOUS`` — a real task (acceptance ACCEPTED),
  bounded by the hard ``ChatConfig.dream_task_budget_cap``: any open
  DREAM-created task whose ``spendTotal`` reaches the cap is failed here.

``originSessionId`` stays null on every task this pass creates, so the
outcome path (``executor.task_outcomes``) never posts into a chat session —
outcomes reach the user through the briefing only.
"""

import logging

import prisma.enums
import prisma.models
from pydantic import BaseModel

from backend.api.features.tasks import tasks_db
from backend.api.features.tasks.models import (
    OPEN_TASK_STATUSES,
    TASK_OUTCOME_MAX_LENGTH,
)
from backend.copilot.config import ChatConfig

logger = logging.getLogger(__name__)

_OPEN_STATUSES = [prisma.enums.DelegatedTaskStatus(s) for s in OPEN_TASK_STATUSES]

# One pass never floods the board: at most this many proposals per run.
MAX_PROACTIVE_TASKS_PER_PASS = 5


class ProactiveTask(BaseModel):
    task_id: str
    expert_id: str
    autonomy: str
    # True for SUGGEST experts — the task is a proposal the user must accept.
    proposal_only: bool


class ProactivePassResult(BaseModel):
    budget_capped_task_count: int = 0
    created: list[ProactiveTask] = []


async def run_proactive_pass(
    user_id: str,
    *,
    dream_pass_id: str,
    config: ChatConfig | None = None,
) -> ProactivePassResult:
    """One sweep: enforce the budget cap, then propose work for idle experts.

    A cap of zero is a real zero-credit limit: open dream tasks are stopped
    and no new work is proposed, so the pass cannot create tasks only to
    fail them on the next sweep."""
    config = config or ChatConfig()
    capped = await enforce_task_budget_caps(user_id, config.dream_task_budget_cap)
    if config.dream_task_budget_cap == 0:
        return ProactivePassResult(budget_capped_task_count=capped)

    created = [
        await _propose_task(user_id, expert, dream_pass_id)
        for expert in (await _idle_experts(user_id))[:MAX_PROACTIVE_TASKS_PER_PASS]
    ]
    return ProactivePassResult(budget_capped_task_count=capped, created=created)


async def enforce_task_budget_caps(user_id: str, cap: int) -> int:
    """Fail every open DREAM-created task whose spend reached *cap*.

    ``spendTotal`` is reconciled when a task's run settles, so this sweep is
    the hard stop that keeps a dream-created task from queueing further paid
    work once its budget is gone. Returns the number of tasks stopped. The
    config floors the cap at zero, which is a real zero-credit limit — any
    spend at all stops the task.
    """
    if cap < 0:
        return 0
    stopped = await prisma.models.DelegatedTask.prisma().update_many(
        where={
            "userId": user_id,
            "createdByType": prisma.enums.TaskCreatedByType.DREAM,
            "status": {"in": _OPEN_STATUSES},
            "spendTotal": {"gte": cap},
        },
        data={
            "status": prisma.enums.DelegatedTaskStatus.FAILED,
            "outcomeSummary": (
                f"Stopped: spend reached the {cap}-credit cap for "
                "dream-created tasks."
            )[:TASK_OUTCOME_MAX_LENGTH],
        },
    )
    if stopped:
        logger.info(
            "Dream proactive: stopped %d over-budget task(s) for user %s",
            stopped,
            user_id[:12],
        )
    return stopped


async def _idle_experts(user_id: str) -> list[prisma.models.Expert]:
    """Hired, non-archived, non-paused experts with no open task."""
    experts = await prisma.models.Expert.prisma().find_many(
        where={
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
            "schedulesPausedAt": None,
        },
        order={"createdAt": "asc"},
    )
    if not experts:
        return []
    open_tasks = await prisma.models.DelegatedTask.prisma().find_many(
        where={
            "userId": user_id,
            "ownerId": {"in": [expert.id for expert in experts]},
            "status": {"in": _OPEN_STATUSES},
        },
        distinct=["ownerId"],
    )
    busy = {task.ownerId for task in open_tasks}
    return [expert for expert in experts if expert.id not in busy]


async def _propose_task(
    user_id: str, expert: prisma.models.Expert, dream_pass_id: str
) -> ProactiveTask:
    proposal_only = expert.autonomyLevel == prisma.enums.ExpertAutonomyLevel.SUGGEST
    task = await tasks_db.create_delegated_task(
        user_id,
        title=_task_title(expert),
        spec=_task_spec(expert, proposal_only=proposal_only),
        owner_id=expert.id,
        origin_session_id=None,
        created_by_type="DREAM",
        created_by_id=dream_pass_id,
    )
    if not proposal_only:
        # ASK_FIRST / AUTONOMOUS tasks are real work, not proposals — mark
        # them accepted so the review flow doesn't hold them for a nod the
        # autonomy level already granted.
        await prisma.models.DelegatedTask.prisma().update_many(
            where={"id": task.id, "userId": user_id},
            data={"acceptance": prisma.enums.DelegatedTaskAcceptance.ACCEPTED},
        )
    return ProactiveTask(
        task_id=task.id,
        expert_id=expert.id,
        autonomy=str(expert.autonomyLevel),
        proposal_only=proposal_only,
    )


def _task_title(expert: prisma.models.Expert) -> str:
    lane = expert.role or expert.name
    return f"Proactive: one small {lane} win"


def _task_spec(expert: prisma.models.Expert, *, proposal_only: bool) -> str:
    lane = expert.role or "your lane"
    lines = [
        f"{expert.name} has no open work. Propose ONE small, concrete "
        f"{lane} task that would help right now, sized to finish in a "
        "single sitting.",
        "Keep it cheap and reversible; skip anything that needs approvals "
        "or external commitments.",
    ]
    if proposal_only:
        lines.append(
            "This is a suggestion only — do not start any work until the "
            "user accepts it."
        )
    return "\n".join(lines)
