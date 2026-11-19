"""Start the tasks whose start time has come (the Scheduler's due-task sweep).

Each pass reads a small batch of due tasks, oldest first, and for each one:

1. Too late (past ``scheduled_task_late_window_seconds`` since it was first
   due): it does not run. It stays TODO, marked missed, for the user to start
   or cancel.
2. The checks an unattended turn passes (``turn_refusal``): a closed or
   banned account's scheduled tasks are cancelled quietly; a missing plan or
   used-up trial holds the task and says why; a used-up daily or weekly
   allowance moves it to the reset.
3. Its owner: an expert no longer on the team hands the task to Otto; a
   paused one holds it.
4. Claim and start: under the chat that asked when that chat is still there
   (background work whose result comes back to it), else in a chat of its own
   (``start_task_run``). Past the running cap it queues; at the hard cap it
   waits for a slot.

A task that fails unexpectedly is put back with a short back-off and the
batch carries on.
"""

import asyncio
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from prisma.enums import TaskActivityKind, TaskActor

from backend.copilot.background_subtasks import launch_background_subtask
from backend.copilot.background_task import SCHEDULED_LAUNCH_PREFIX
from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.limit_tracking import track_turn_refusal
from backend.copilot.model import ChatSession, get_chat_session
from backend.copilot.session_permissions import resolve_session_permissions
from backend.copilot.task_run import start_task_run
from backend.copilot.task_watch_run import event_brief, started_data, started_line
from backend.copilot.tools.start_task import link_worker, worker_prompt
from backend.copilot.tree import TreeRefusal, root_envelope
from backend.copilot.turn_queue import InflightCapExceeded, TurnRefusal, turn_refusal
from backend.data.db_accessors import (
    experts_db,
    task_db,
    task_due_db,
    task_record_db,
    task_run_db,
    task_schedule_db,
)
from backend.data.task import Task, TaskPatch
from backend.data.task_due import DueTask
from backend.data.task_record import ActivityInput
from backend.data.task_rhythm import next_run_at
from backend.data.task_run import TaskRunClaim
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

DUE_BATCH_SIZE = 20
# After an unexpected failure or an outage on our side.
RETRY_AFTER = timedelta(seconds=60)
# While the user has something to fix (a plan, a paused expert).
HOLD_AFTER = timedelta(minutes=10)
# At the hard cap of running plus queued work.
SLOT_WAIT = timedelta(minutes=2)
# Spreads tasks moved to one usage reset over a few minutes.
USAGE_SPREAD_SECONDS = 300

MISSED_TEXT = "Missed its start time"
SKIPPED_TEXT = "Skipped a run: it could not start in time"
WAITING_FOR_SLOT_TEXT = "Waiting for a free slot to start"
_HOLD_TEXT = {
    "no_plan": "Waiting to start: your account needs a plan",
    "codex_access": "Waiting to start: reconnect your ChatGPT account",
    "usage": "Waiting to start: your usage allowance is used up",
}
_EXPERT_PAUSED_TEXT = "Waiting to start: its expert is paused"
_HANDED_TO_OTTO_TEXT = "Handed to Otto: its expert is no longer on your team"

ExpertState = Literal["active", "gone", "paused", "unavailable"]


async def start_due_tasks(now: datetime | None = None) -> None:
    """One pass of the sweep; one indexed query when nothing is due."""
    now = now or datetime.now(UTC)
    for item in await task_due_db().list_due_tasks(now, DUE_BATCH_SIZE):
        try:
            await start_due_task(item, now)
        except Exception:
            logger.exception(f"Could not start due task {item.task.id}")
            await _back_off(item, now)


async def start_due_task(item: DueTask, now: datetime) -> None:
    """Start one due task, or hold, move, miss or cancel it."""
    late_window = timedelta(
        seconds=Settings().config.scheduled_task_late_window_seconds
    )
    watching = bool(item.repeat and item.repeat.watch)
    # An event happened: running late for it beats not running at all.
    late = not watching and now - item.due_at > late_window
    if late and item.repeat:
        await _skip_run(item, now)
        return
    if late:
        await _mark_missed(item)
        return
    origin = await _origin_chat(item)
    provider: CopilotLlmAuthProvider = (
        origin.metadata.llm_auth_provider if origin else "platform"
    )
    if await _refused(item, provider, now):
        return
    expert = await _owner_state(item)
    if expert in ("paused", "unavailable"):
        await _hold_for_expert(item, expert, now)
        return
    if expert == "gone":
        await _hand_to_otto(item)
        if origin is not None and origin.expert_id == item.task.owner_expert_id:
            origin = None
    claim = await task_due_db().claim_due_task(
        item.task.id, item.task.user_id, item.start_at
    )
    if claim is None:
        return  # started by hand, moved or cancelled since it was read
    try:
        await _start(claim, item, origin)
    except InflightCapExceeded:
        await _hold(item, now + SLOT_WAIT, WAITING_FOR_SLOT_TEXT, item.due_at)


async def _start(
    claim: TaskRunClaim, item: DueTask, origin: ChatSession | None
) -> None:
    """Under the chat that asked when it can, else in a chat of the task's own,
    which queues past the running cap."""
    user_id = item.task.user_id
    if origin is not None:
        try:
            await _start_under_chat(claim, item, origin)
            return
        except TreeRefusal as refused:
            logger.info(
                f"Due task {item.task.id} runs in its own chat: {refused.message}"
            )
        except InflightCapExceeded:
            raise
        except Exception:
            # Running late in a chat of its own beats not running at all.
            logger.exception(
                f"Due task {item.task.id} could not start under its chat; "
                "it runs in its own chat"
            )
        reclaimed = await task_due_db().claim_due_task(
            item.task.id, user_id, item.start_at
        )
        if reclaimed is None:
            return
        claim = reclaimed
    await start_task_run(user_id, item.task.id, claim=claim)


async def start_scheduled_now(user_id: str, task_id: str) -> Task | None:
    """Start a task that waits for its start time now, under the chat that
    asked, so its result still comes back there. None when the task is not
    scheduled, that chat is gone or the start under it failed: the caller
    starts it in a chat of its own."""
    item = await task_due_db().get_scheduled_task(task_id, user_id)
    if item is None:
        return None
    origin = await _origin_chat(item)
    if origin is None:
        return None
    claim = await task_due_db().claim_due_task(task_id, user_id, item.start_at)
    if claim is None:
        return None
    try:
        await _start_under_chat(claim, item, origin, note="You started it")
    except InflightCapExceeded:
        raise
    except Exception:
        logger.exception(f"Task {task_id} could not start under its chat")
        return None
    return await task_db().get_task(task_id, user_id)


async def _start_under_chat(
    claim: TaskRunClaim,
    item: DueTask,
    origin: ChatSession,
    note: str = "Started at its scheduled time",
) -> None:
    """Background work under the chat that asked, as ``start_task`` starts it,
    so the result comes back to that chat (``background_delivery``). Any
    failure puts the task back as it was claimed."""
    user_id = item.task.user_id
    turn_id = f"{SCHEDULED_LAUNCH_PREFIX}{item.task.id}:{item.start_at.isoformat()}"
    if item.repeat and claim.task.started_at:
        # Run now and the next scheduled run share a start time: the claim
        # tells each run's work apart.
        turn_id = f"{turn_id}:{claim.task.started_at.isoformat()}"
    try:
        background = await launch_background_subtask(
            user_id,
            origin,
            worker_prompt(claim.task, event_brief(claim)),
            envelope=root_envelope(turn_id, session_id=origin.session_id),
            call_id="start",
            turn_id=turn_id,
            permissions=resolve_session_permissions(origin),
        )
    except BaseException:
        await asyncio.shield(task_run_db().undo_task_run(claim, user_id))
        raise
    await link_worker(claim.task, background.child_session_id, user_id)
    await _note(
        item,
        started_line(claim, note),
        TaskActivityKind.STARTED,
        chat_session_id=background.child_session_id,
        data=started_data(claim),
    )


async def _refused(
    item: DueTask, provider: CopilotLlmAuthProvider, now: datetime
) -> bool:
    """Apply the unattended-turn checks; True when the task may not start now."""
    user_id = item.task.user_id
    try:
        refusal = await turn_refusal(user_id, provider)
    except Exception as e:
        refusal = TurnRefusal(
            reason=f"has an unreadable plan ({type(e).__name__})",
            kind="outage",
            transient=True,
        )
    if refusal is None:
        return False
    logger.info(f"Due task {item.task.id} held: owner {refusal.reason}")
    if refusal.kind == "banned":
        await task_schedule_db().cancel_scheduled_tasks(user_id)
    elif refusal.kind == "outage":
        await _hold(item, now + RETRY_AFTER, None, item.due_at)
    elif _waits_for_reset(refusal):
        await _move_to_reset(item, refusal)
    else:
        await _hold(item, now + HOLD_AFTER, _HOLD_TEXT[refusal.kind], item.due_at)
    return True


def _waits_for_reset(refusal: TurnRefusal) -> bool:
    return (
        refusal.kind == "usage"
        and refusal.window in ("daily", "weekly")
        and refusal.resets_at is not None
    )


async def _move_to_reset(item: DueTask, refusal: TurnRefusal) -> None:
    """A used-up daily or weekly allowance moves the task to its reset, the
    way a one-shot follow-up waits for it; the reset is its new due time."""
    assert refusal.resets_at is not None and refusal.window is not None
    spread = int(hashlib.sha256(item.task.id.encode()).hexdigest(), 16)
    at = refusal.resets_at + timedelta(seconds=spread % USAGE_SPREAD_SECONDS)
    text = f"Moved to when your {refusal.window} usage resets"
    if await _hold(item, at, text, at):
        track_turn_refusal(
            item.task.user_id, refusal, surface="schedule", outcome="queued"
        )


async def _owner_state(item: DueTask) -> ExpertState:
    """Whether the task's expert can take it; Otto always can."""
    expert_id = item.task.owner_expert_id
    if expert_id is None:
        return "active"
    user_id = item.task.user_id
    try:
        expert = await experts_db().get_expert(
            user_id, expert_id, include_workflows=False, include_archived=True
        )
        if expert is None:
            exists = await experts_db().expert_row_exists(user_id, expert_id)
            return "unavailable" if exists else "gone"
    except Exception:
        logger.warning(f"Could not look up the expert of due task {item.task.id}")
        return "unavailable"
    if expert.is_archived:
        return "gone"
    return "paused" if expert.schedules_paused_at is not None else "active"


async def _hold_for_expert(
    item: DueTask, state: Literal["paused", "unavailable"], now: datetime
) -> None:
    if state == "paused":
        await _hold(item, now + HOLD_AFTER, _EXPERT_PAUSED_TEXT, item.due_at)
    else:
        await _hold(item, now + RETRY_AFTER, None, item.due_at)


async def _hand_to_otto(item: DueTask) -> None:
    await task_db().update_task(
        item.task.id, item.task.user_id, TaskPatch(assign_to_otto=True)
    )
    await _note(item, _HANDED_TO_OTTO_TEXT, TaskActivityKind.STATUS)


async def _origin_chat(item: DueTask) -> ChatSession | None:
    """The chat that asked for the task, while it is there and not archived."""
    if not item.origin_session_id:
        return None
    chat = await get_chat_session(item.origin_session_id, item.task.user_id)
    if chat is None or chat.archived_at is not None:
        return None
    return chat


async def _skip_run(item: DueTask, now: datetime) -> None:
    """A repeating task too late for this run waits for its next one; with
    none left in its rhythm it is marked missed like a one-off."""
    assert item.repeat is not None
    next_at = next_run_at(item.repeat, now)
    if next_at is None:
        await _mark_missed(item)
        return
    if await _hold(item, next_at, None, next_at):
        await _note(item, SKIPPED_TEXT, TaskActivityKind.STATUS)


async def _mark_missed(item: DueTask) -> None:
    if await task_due_db().mark_task_missed(
        item.task.id, item.task.user_id, item.start_at, MISSED_TEXT
    ):
        await _note(item, MISSED_TEXT, TaskActivityKind.STATUS)


async def _hold(
    item: DueTask, at: datetime, status_text: str | None, due_at: datetime
) -> bool:
    return await task_due_db().hold_due_task(
        item.task.id, item.task.user_id, item.start_at, at, status_text, due_at
    )


async def _back_off(item: DueTask, now: datetime) -> None:
    """Put a task that failed unexpectedly back, to try again shortly."""
    try:
        await _hold(item, now + RETRY_AFTER, None, item.due_at)
    except Exception:
        logger.exception(f"Could not put due task {item.task.id} back")


async def _note(
    item: DueTask,
    text: str,
    kind: TaskActivityKind,
    chat_session_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """Add a line to the task's activity; the start matters more."""
    try:
        await task_record_db().add_activity(
            item.task.id,
            item.task.user_id,
            ActivityInput(
                kind=kind,
                text=text,
                actor_type=TaskActor.SYSTEM,
                chat_session_id=chat_session_id,
                is_milestone=True,
                data=data or {},
            ),
        )
    except Exception as e:
        logger.warning(f"Could not note activity on task {item.task.id}: {e}")
