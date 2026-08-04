"""Expert schedule/trigger lifecycle and the weekly credit guardrail.

Owns everything that connects an Expert to the scheduler and to her
AgentPresets: install-time schedule creation, detach-on-archive and
reattach-on-revive, budget pause/resume with an audit trail, and the
run-time budget gate every expert-attributed execution passes through.
Split out of ``experts_db`` to keep both files within the size guideline.
"""

import logging
import uuid
from datetime import datetime, timezone

import prisma.models

from backend.api.features.experts.models import ExpertDetachPreview
from backend.copilot import db as chat_db
from backend.data.expert_spend import get_weekly_spend
from backend.data.user import get_user_by_id
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import ExpertRunPausedError
from backend.util.settings import Settings
from backend.util.timezone_utils import get_user_timezone_or_utc

logger = logging.getLogger(__name__)
settings = Settings()

_BUDGET_WARN_FRACTION = 0.8
# Namespace for deterministic post ids: one warning / one pause message per
# expert per ISO week, deduplicated by append_expert_run_message.
_POST_NAMESPACE = uuid.UUID("6f9c1f6e-8f4b-4c6e-9be1-a1b2c3d4e5f6")


def effective_weekly_budget(row: prisma.models.Expert) -> int | None:
    """The budget the guardrail enforces: the expert's own, else the
    platform default. ``None`` = guardrail disabled (value <= 0)."""
    budget = (
        row.weeklyBudget
        if row.weeklyBudget is not None
        else settings.config.expert_weekly_credit_budget_default
    )
    return budget if budget > 0 else None


async def create_workflow_schedule(
    workflow_row_id: str,
    expert_id: str,
    user_id: str,
    cron: str,
    graph_id: str,
    graph_version: int,
    name: str,
    user_timezone: str,
) -> bool:
    """Create the attributed schedule for an expert workflow and record its
    id on the ExpertWorkflow row.

    Failure is non-fatal and expected for agents that need credentials the
    user hasn't connected yet: the cadence stays on the row with a null
    ``scheduleId``, surfacing the workflow as "needs setup" instead of
    silently dropping the roster's intent.
    """
    try:
        schedule = await get_scheduler_client().add_execution_schedule(
            user_id=user_id,
            graph_id=graph_id,
            graph_version=graph_version,
            name=name,
            cron=cron,
            input_data={},
            input_credentials={},
            user_timezone=user_timezone,
            expert_id=expert_id,
        )
    except Exception as e:
        logger.warning(
            f"Schedule for expert #{expert_id} workflow #{workflow_row_id} "
            f"not created (needs setup): {type(e).__name__}: {e}"
        )
        return False
    try:
        await prisma.models.ExpertWorkflow.prisma().update(
            where={"id": workflow_row_id}, data={"scheduleId": schedule.id}
        )
    except Exception as e:
        # Never leave a schedule firing with no row pointing at it: undo the
        # registration, fall back to needs-setup. A failed cleanup is loud so
        # operators can delete the orphan by the logged id.
        logger.warning(
            f"Failed to record schedule #{schedule.id} on workflow "
            f"#{workflow_row_id}; deleting it: {type(e).__name__}: {e}"
        )
        try:
            await get_scheduler_client().delete_schedule(schedule.id, user_id=user_id)
        except Exception as cleanup_error:
            logger.error(
                f"Orphaned schedule #{schedule.id} for expert #{expert_id} "
                f"could not be deleted: {type(cleanup_error).__name__}: "
                f"{cleanup_error}"
            )
        return False
    return True


async def _get_expert_schedules(user_id: str, expert_id: str) -> list:
    schedules = await get_scheduler_client().get_execution_schedules(user_id)
    return [s for s in schedules if s.kind == "graph" and s.expert_id == expert_id]


async def get_detach_preview(user_id: str, expert_id: str) -> ExpertDetachPreview:
    """What archiving this expert would pause — drives the confirm dialog
    so triggers are never silently orphaned or silently kept firing."""
    schedules = await _get_expert_schedules(user_id, expert_id)
    presets = await prisma.models.AgentPreset.prisma().find_many(
        where={
            "expertId": expert_id,
            "userId": user_id,
            "isDeleted": False,
            "isActive": True,
        }
    )
    return ExpertDetachPreview(
        schedule_names=[s.name or s.cron for s in schedules],
        trigger_names=[p.name for p in presets],
    )


async def detach_expert_triggers(user_id: str, expert_id: str) -> None:
    """Deactivate the expert's presets and delete her schedules.

    Called on archive. Preset deactivation is the loud guard against
    orphaned webhook firing; the run-time gate (``enforce_expert_run_budget``
    refusing archived/paused experts) is the backstop for anything missed.
    """
    await prisma.models.AgentPreset.prisma().update_many(
        where={"expertId": expert_id, "userId": user_id, "isDeleted": False},
        data={"isActive": False},
    )
    scheduler = get_scheduler_client()
    for schedule in await _get_expert_schedules(user_id, expert_id):
        try:
            await scheduler.delete_schedule(schedule.id, user_id=user_id)
        except Exception as e:
            logger.warning(
                f"Failed to delete schedule #{schedule.id} while detaching "
                f"expert #{expert_id}: {type(e).__name__}: {e}"
            )
    await prisma.models.ExpertWorkflow.prisma().update_many(
        where={"expertId": expert_id},
        data={"scheduleId": None},
    )


async def reattach_expert_triggers(user_id: str, expert_id: str) -> None:
    """Reverse of ``detach_expert_triggers``, for re-hire revival:
    reactivate presets and recreate schedules from the stored cadence."""
    await prisma.models.AgentPreset.prisma().update_many(
        where={"expertId": expert_id, "userId": user_id, "isDeleted": False},
        data={"isActive": True},
    )
    workflows = await prisma.models.ExpertWorkflow.prisma().find_many(
        where={
            "expertId": expert_id,
            "scheduleCron": {"not": None},
            "scheduleId": None,
        },
        include={"LibraryAgent": True, "StoreListingVersion": True},
    )
    if not workflows:
        return
    user = await get_user_by_id(user_id)
    user_timezone = get_user_timezone_or_utc(user.timezone if user else None)
    for workflow in workflows:
        agent = workflow.LibraryAgent
        if agent is None or not workflow.scheduleCron:
            continue
        listing = workflow.StoreListingVersion
        await create_workflow_schedule(
            workflow_row_id=workflow.id,
            expert_id=expert_id,
            user_id=user_id,
            cron=workflow.scheduleCron,
            graph_id=agent.agentGraphId,
            graph_version=agent.agentGraphVersion,
            name=listing.name if listing else "Expert workflow",
            user_timezone=user_timezone,
        )


async def pause_expert_schedules(user_id: str, expert_id: str, reason: str) -> bool:
    """Pause the expert's scheduled/triggered runs (chat is untouched) and
    log the pause. Returns False when already paused (no double events)."""
    updated = await prisma.models.Expert.prisma().update_many(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "schedulesPausedAt": None,
        },
        data={"schedulesPausedAt": datetime.now(timezone.utc)},
    )
    if updated == 0:
        return False
    await prisma.models.ExpertPauseEvent.prisma().create(
        data={"expertId": expert_id, "reason": reason}
    )
    return True


async def resume_expert_schedules(user_id: str, expert_id: str) -> bool:
    """One-click reversal of a pause; stamps the open pause event."""
    updated = await prisma.models.Expert.prisma().update_many(
        where={"id": expert_id, "ownerUserId": user_id, "isTemplate": False},
        data={"schedulesPausedAt": None},
    )
    if updated == 0:
        return False
    await prisma.models.ExpertPauseEvent.prisma().update_many(
        where={"expertId": expert_id, "clearedAt": None},
        data={"clearedAt": datetime.now(timezone.utc)},
    )
    return True


async def enforce_expert_run_budget(user_id: str, expert_id: str) -> None:
    """Run-start gate for expert-attributed executions (schedules and
    triggers; chat runs never carry an expert_id and are never gated).

    Raises ExpertRunPausedError when the expert is archived, paused, or has
    hit her weekly credit budget — breaching pauses her and posts an
    in-thread message. Approaching the budget posts a once-per-week warning.
    """
    expert = await prisma.models.Expert.prisma().find_first(
        where={"id": expert_id, "ownerUserId": user_id, "isTemplate": False}
    )
    if expert is None:
        return
    if expert.isArchived:
        raise ExpertRunPausedError(
            f"{expert.name} is archived; her schedules do not run.", expert_id
        )
    if expert.schedulesPausedAt is not None:
        raise ExpertRunPausedError(f"{expert.name}'s schedules are paused.", expert_id)
    budget = effective_weekly_budget(expert)
    if budget is None:
        return
    spent = await get_weekly_spend(expert_id)
    if spent >= budget:
        await pause_expert_schedules(
            user_id,
            expert_id,
            reason=f"Weekly credit budget reached ({spent}/{budget})",
        )
        await _post_budget_message(user_id, expert, spent, budget, breached=True)
        raise ExpertRunPausedError(
            f"{expert.name} hit her weekly credit budget ({spent}/{budget}); "
            "schedules are paused until you resume them.",
            expert_id,
        )
    if spent >= int(budget * _BUDGET_WARN_FRACTION):
        await _post_budget_message(user_id, expert, spent, budget, breached=False)


async def _post_budget_message(
    user_id: str,
    expert: prisma.models.Expert,
    spent: int,
    budget: int,
    breached: bool,
) -> None:
    """Post the warning/pause message into the expert's thread. Deduped to
    once per expert per week per kind via the deterministic message id.
    Never raises — a failed post must not affect the run decision."""
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    kind = "pause" if breached else "warn"
    message_id = str(
        uuid.uuid5(_POST_NAMESPACE, f"budget-{kind}:{expert.id}:{year}-W{week:02d}")
    )
    if breached:
        content = (
            f"I've hit my weekly credit budget ({spent} of {budget} credits), "
            "so my scheduled runs and triggers are paused for now. Chat with "
            "me anytime — and you can resume my schedules from the Team page "
            "whenever you're ready."
        )
    else:
        content = (
            f"Heads up — I've used {spent} of my {budget} weekly credits. "
            "If I reach the cap, my scheduled runs pause until you raise it "
            "or resume me from the Team page."
        )
    try:
        await chat_db.append_expert_run_message(
            user_id=user_id,
            expert_id=expert.id,
            content=content,
            message_id=message_id,
        )
    except Exception as e:
        logger.warning(
            f"Failed to post budget {kind} message for expert #{expert.id}: "
            f"{type(e).__name__}: {e}"
        )
