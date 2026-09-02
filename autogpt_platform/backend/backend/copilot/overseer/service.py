"""The overseer pass: a 15-minute per-user sweep over open delegated tasks.

Runs in the scheduler process (Prisma-less), so every read and write goes
through the DatabaseManager RPC. Three checks per pass:

* **Stall** — an open task with no live execution and no update for 15
  minutes gets one retry (a nudge into its working session, or a fresh
  worker session when it never had one); a task that stalls again after the
  retry is closed FAILED, which Home's attention list then surfaces.
  QUEUED counts as stalled too: a receipt opened outside a conversation has
  nothing driving it, so silence there is the failure, not the wait.
* **Staleness** — a WAITING_USER task unanswered for 7 days gets ``staleAt``
  stamped. Never auto-cancelled; Home nags instead.
* **Expert health** — an expert with 3+ FAILED tasks inside 7 days is
  paused through the existing ExpertPauseEvent flow, which Home's paused
  attention card already surfaces.

The daily briefing cards (24h nudges, merge suggestions, hire
recommendations) are composed at briefing-generation time from
``cards.py`` / ``recruiter.py`` — they describe a day, not a 15-minute
window, so they live with the briefing composer rather than this pass.
"""

import logging
from datetime import UTC, datetime, timedelta

from backend.api.features.tasks.models import DelegatedTask
from backend.copilot.executor.utils import schedule_chat_turn
from backend.copilot.pending_message_helpers import queue_user_message
from backend.copilot.task_kickoff import start_task_in_new_session
from backend.util.clients import get_database_manager_async_client
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

STALL_AFTER = timedelta(minutes=15)
STALE_AFTER = timedelta(days=7)
EXPERT_FAILURE_WINDOW = timedelta(days=7)
EXPERT_FAILURE_THRESHOLD = 3

_RETRY_NOTE = (
    "No progress for 15 minutes with no run in flight — the overseer asked "
    "the owner to pick this back up."
)
_FAILED_OUTCOME = (
    "Stalled with no progress after a retry; marked failed by the overseer."
)


def _as_utc(value: datetime) -> datetime:
    """Normalize a possibly naive DB timestamp to timezone-aware UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


async def run_overseer_pass(
    user_id: str, *, now: datetime | None = None
) -> dict[str, int]:
    """One sweep for one user. Returns per-check counters for the job log."""
    summary = {"retried": 0, "failed": 0, "stale": 0, "paused_experts": 0}
    if not await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False):
        return summary
    if not await is_feature_enabled(
        Flag.EXPERT_TASK_MANAGEMENT, user_id, default=False
    ):
        return summary
    now = now or datetime.now(UTC)
    client = get_database_manager_async_client()

    tasks = await client.list_open_tasks(user_id)
    if tasks:
        stalled = await _find_stalled(client, user_id, tasks, now)
        for task in stalled:
            if any(a.kind == "retry" for a in task.amendments):
                await client.close_delegated_task(
                    user_id=user_id,
                    task_id=task.id,
                    succeeded=False,
                    outcome_summary=_FAILED_OUTCOME,
                )
                summary["failed"] += 1
            else:
                await _retry_task(client, user_id, task)
                summary["retried"] += 1

        for task in tasks:
            if (
                task.status == "WAITING_USER"
                and task.stale_at is None
                and _as_utc(task.updated_at) < now - STALE_AFTER
            ):
                if await client.mark_task_stale(user_id, task.id, stale_at=now):
                    summary["stale"] += 1

    counts = await client.count_recent_failed_tasks_by_expert(
        user_id, since=now - EXPERT_FAILURE_WINDOW
    )
    for expert_id, failures in counts.items():
        if failures >= EXPERT_FAILURE_THRESHOLD:
            paused = await client.pause_expert_schedules(
                user_id,
                expert_id,
                reason=f"{failures} failed tasks in the last 7 days",
            )
            if paused:
                summary["paused_experts"] += 1

    return summary


async def _find_stalled(
    client, user_id: str, tasks: list[DelegatedTask], now: datetime
) -> list[DelegatedTask]:
    candidates = [
        task
        for task in tasks
        if _as_utc(task.updated_at) < now - STALL_AFTER and _can_stall(task)
    ]
    if not candidates:
        return []
    running = await client.has_running_executions(
        user_id, [task.id for task in candidates]
    )
    return [task for task in candidates if not running.get(task.id)]


def _can_stall(task: DelegatedTask) -> bool:
    """Whether silence on this task means something went wrong.

    A QUEUED task normally has a turn starting behind it, so silence means
    its kickoff never landed. The exception is the dream pass, which opens
    sessionless tasks on purpose — a proposal sitting untouched is it
    waiting for the user, not a stall, and starting it here would do work
    nobody approved.
    """
    if task.status == "WORKING":
        return True
    return task.status == "QUEUED" and task.created_by_type != "DREAM"


async def _retry_task(client, user_id: str, task: DelegatedTask) -> None:
    """First stall: record the retry on the timeline, then nudge the working
    session back into motion. The amendment write bumps ``updatedAt``, so
    the next stall check measures from the retry — a second silent window
    is what fails the task.

    A task with no session never got a worker at all (its kickoff failed, or
    it was opened outside a conversation), so the retry opens one instead of
    nudging."""
    await client.append_task_amendment(
        user_id, task.id, note=_RETRY_NOTE, by="overseer", kind="retry"
    )
    if task.origin_session_id is None:
        await start_task_in_new_session(
            user_id,
            task_id=task.id,
            title=task.title,
            expert_id=task.owner.id if task.owner else None,
        )
        return
    message = (
        f"[Overseer] Task '{task.title}' (task_id: {task.id}) has stalled — "
        "no progress for 15 minutes and no run in flight. Pick it back up "
        "and finish it, or escalate to the user with escalate_task if you "
        "are blocked."
    )
    try:
        queued = await queue_user_message(
            session_id=task.origin_session_id,
            message=message,
            require_turn_in_flight=True,
        )
        if not queued.turn_in_flight:
            await schedule_chat_turn(
                session_id=task.origin_session_id,
                user_id=user_id,
                message=message,
            )
    except Exception:
        logger.warning(
            "Overseer could not nudge session #%s for task #%s",
            task.origin_session_id,
            task.id,
            exc_info=True,
        )
