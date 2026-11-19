"""Start and stop a task's work from the task itself, with no chat open.

Starting opens a chat for the task's owner (Otto when there is none) and
sends the worker's instructions as its first message, through the same
scheduling path a typed message takes: the user's concurrent-turn cap, the
queue past the running cap, and the executor's allowance and billing checks.
The chat is the task's work chat, so its tool calls land on the task's record
(``task_recorder``), and it is marked as the task's run, so each of its turns
settles the task when it ends (``task_run_settle``).
"""

import asyncio
import logging

from prisma.enums import TaskActivityKind, TaskActor, TaskStatus

from backend.copilot import stream_registry, task_recorder, turn_queue
from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    get_inflight_turn_limit,
)
from backend.copilot.background_delivery import cancel_background_subtasks
from backend.copilot.executor.utils import enqueue_cancel_task, schedule_chat_turn
from backend.copilot.model import ChatSession, create_chat_session, delete_chat_session
from backend.copilot.session_permissions import resolve_session_permissions
from backend.copilot.task_watch_run import started_data, started_line
from backend.copilot.tools.start_task import BRIEF_MAX_CHARS, worker_prompt_text
from backend.data.db_accessors import (
    background_task_db,
    task_db,
    task_record_db,
    task_run_db,
    task_schedule_db,
)
from backend.data.task import OPEN_STATUSES, Task, TaskPatch
from backend.data.task_record import ActivityInput
from backend.data.task_run import TaskRunClaim
from backend.data.task_runs import run_end_data
from backend.data.task_watch_event import what_happened
from backend.util.exceptions import ConflictError, ExpertNotFoundError, NotFoundError

logger = logging.getLogger(__name__)

# Marks the first message of a task's run chat, for the record and analytics.
TASK_RUN_KEY = "task_run_id"


async def start_task_run(
    user_id: str,
    task_id: str,
    instructions: str | None = None,
    *,
    organization_id: str | None = None,
    team_id: str | None = None,
    impersonated: bool = False,
    claim: TaskRunClaim | None = None,
) -> Task:
    """Start a new run of the task in a chat of its own. Raises NotFoundError
    for a task the user does not own, ConflictError while it is being worked
    on, and the turn-cap errors of a normal message when the user is at
    their limit; on any failure the task is left as it was. ``claim`` is a
    claim already taken (the due-task sweep's), used instead of a new one."""
    claim = claim or await task_run_db().claim_task_run(task_id, user_id)
    instructions = (instructions or "").strip() or None
    session: ChatSession | None = None
    try:
        session = await _open_work_chat(
            user_id, claim.task, organization_id=organization_id, team_id=team_id
        )
        task = await _make_work_chat(claim.task, session.session_id, user_id)
        await _send_first_message(
            session, user_id, run_prompt(task, instructions, claim), impersonated
        )
    except BaseException:
        await asyncio.shield(_undo(claim, user_id, session))
        raise
    await _note_start(claim, user_id, session.session_id, instructions)
    return task


async def cancel_task_run(
    user_id: str, task_id: str, this_run_only: bool = False
) -> Task:
    """Stop the work running for the task and mark it cancelled; a repeating
    task stops repeating. With ``this_run_only`` a repeating task stops only
    the run going now and waits for its next one. A task that is already
    settled is returned as it is."""
    task = await task_db().get_task(task_id, user_id)
    if task is None:
        raise NotFoundError(f"Task {task_id} not found")
    if task.status not in OPEN_STATUSES:
        return task
    if this_run_only and task.repeat and task.status != TaskStatus.TODO:
        return await _stop_this_run(user_id, task)
    not_started = task.status == TaskStatus.TODO
    # Cancelled first, so the stopped turn's end finds nothing left to settle;
    # a task waiting for its start time loses it (``update_task``).
    task = await task_db().update_task(
        task_id, user_id, TaskPatch(status=TaskStatus.CANCELLED)
    )
    if task.repeat:
        await task_schedule_db().clear_task_repeat(task_id, user_id)
        task = await task_db().get_task(task_id, user_id) or task
    if task.work_session_id:
        await _stop_work(user_id, task.work_session_id)
    await _note(
        task_id,
        user_id,
        ActivityInput(
            kind=TaskActivityKind.STATUS,
            text="You cancelled it" if not_started else "You stopped it",
            actor_type=TaskActor.USER,
            chat_session_id=task.work_session_id,
            is_milestone=True,
            data={} if not_started else run_end_data("cancelled"),
        ),
    )
    return task


async def _stop_this_run(user_id: str, task: Task) -> Task:
    """Back to waiting first, so the stopped turn's end finds nothing to
    settle; then stop the run's work."""
    waiting = await task_schedule_db().end_repeat_run(task.id, user_id)
    if waiting is None:
        raise ConflictError(f"'{task.title}' changed meanwhile. Try again.")
    if task.work_session_id:
        await _stop_work(user_id, task.work_session_id)
    await _note(
        task.id,
        user_id,
        ActivityInput(
            kind=TaskActivityKind.STATUS,
            text="You stopped this run",
            actor_type=TaskActor.USER,
            chat_session_id=task.work_session_id,
            is_milestone=True,
            data=run_end_data("cancelled"),
        ),
    )
    return waiting


def run_prompt(
    task: Task, instructions: str | None, claim: TaskRunClaim | None = None
) -> str:
    """The worker's instructions: the task's brief, what the user asked to
    change this time and, for a watching task, the event it runs for."""
    brief = run_brief(task, instructions)
    if claim is not None and claim.event is not None:
        brief = f"{brief}\n\n{what_happened(claim.event)}"
    return worker_prompt_text(task.title, task.id, brief, task.done_when)


# Nobody waits on a run started from the task's page, so a worker that is
# stuck says so in its last words, which become the task's failure reason.
BLOCKED_RULE = (
    "If you cannot do this task (something it needs is not connected, or the "
    "data does not exist), do not call finish_task: end with one or two plain "
    "sentences saying what blocks you and what the user would need to do."
)


def run_brief(task: Task, instructions: str | None) -> str:
    """The goal unless the description already says it, the description (or
    the title when there is neither), the user's note, and what to do when
    blocked. Only the description is cut to fit, so the rest always arrives
    whole."""
    goal = _goal_line(task)
    note = f"The user asked for this change: {instructions}" if instructions else ""
    fixed = [part for part in (goal, note, BLOCKED_RULE) if part]
    room = BRIEF_MAX_CHARS - sum(len(part) + 2 for part in fixed)
    body = (task.description or ("" if goal else task.title)).strip()
    body = body[: max(0, room)].rstrip()
    return "\n\n".join(part for part in (goal, body, note, BLOCKED_RULE) if part)


def _goal_line(task: Task) -> str:
    goal = " ".join((task.goal or "").split())
    if not goal:
        return ""
    description = " ".join((task.description or "").split())
    if goal.casefold() in description.casefold():
        return ""
    return f"Goal: {goal}"


async def _open_work_chat(
    user_id: str,
    task: Task,
    *,
    organization_id: str | None,
    team_id: str | None,
) -> ChatSession:
    """A chat for the task's owner; Otto's when the expert is gone."""
    tenancy = (
        (task.organization_id, task.team_id)
        if task.organization_id
        else (organization_id, team_id)
    )
    try:
        return await _create_chat(user_id, task.id, task.owner_expert_id, tenancy)
    except ExpertNotFoundError:
        if task.owner_expert_id is None:
            raise
    return await _create_chat(user_id, task.id, None, tenancy)


async def _create_chat(
    user_id: str,
    task_id: str,
    expert_id: str | None,
    tenancy: tuple[str | None, str | None],
) -> ChatSession:
    return await create_chat_session(
        user_id,
        dry_run=False,
        organization_id=tenancy[0],
        team_id=tenancy[1],
        # Its first message is the task's brief, which Otto may have drafted
        # from untrusted data: no person drives this chat.
        origin="automation",
        expert_id=expert_id,
        work_task_id=task_id,
    )


async def _make_work_chat(task: Task, session_id: str, user_id: str) -> Task:
    """Link the chat to the task and name it the chat doing the work."""
    await task_db().attach_task_session(task.id, session_id, user_id)
    task = await task_db().update_task(
        task.id, user_id, TaskPatch(work_session_id=session_id)
    )
    task_recorder.forget(session_id)
    return task


async def _send_first_message(
    session: ChatSession, user_id: str, prompt: str, impersonated: bool
) -> None:
    """Start the chat's first turn as a sent message would, or queue it when
    the user is at the running cap; past the hard cap this raises."""
    metadata = {TASK_RUN_KEY: session.metadata.work_task_id}
    permissions = resolve_session_permissions(session)
    try:
        turn_id = await schedule_chat_turn(
            session_id=session.session_id,
            user_id=user_id,
            message=prompt,
            message_metadata=metadata,
            dedupe_content=False,
            expert_id=session.expert_id,
            session_origin=session.metadata.origin,
            organization_id=session.organization_id,
            team_id=session.team_id,
            llm_auth_provider=session.metadata.llm_auth_provider,
            llm_credential_id=session.metadata.llm_credential_id,
            permissions=permissions,
            impersonated=impersonated,
        )
    except ConcurrentTurnLimitError:
        await turn_queue.try_enqueue_turn(
            user_id=user_id,
            inflight_cap=get_inflight_turn_limit(),
            session_id=session.session_id,
            message=prompt,
            message_metadata=metadata,
            llm_auth_provider=session.metadata.llm_auth_provider,
            llm_credential_id=session.metadata.llm_credential_id,
            permissions=(
                permissions.model_dump(exclude_none=True) if permissions else None
            ),
            impersonated=impersonated,
        )
        return
    if turn_id is None:
        raise RuntimeError("The task's chat is already busy.")


async def _stop_work(user_id: str, session_id: str) -> None:
    """Stop the work chat's turn, queued or running, and the background work
    it started. Best-effort: the task is already cancelled."""
    try:
        await cancel_background_subtasks(user_id, session_id)
        if await background_task_db().cancel_queued_background_followup(
            user_id, session_id
        ):
            return
        if await turn_queue.cancel_queued_turn(user_id=user_id, session_id=session_id):
            return
        active, _ = await stream_registry.get_active_session(session_id, user_id)
        if active is not None:
            await enqueue_cancel_task(session_id, active.turn_id)
    except Exception:
        logger.exception(f"Could not stop the work in chat {session_id}")


async def _undo(claim: TaskRunClaim, user_id: str, session: ChatSession | None) -> None:
    try:
        await task_run_db().undo_task_run(claim, user_id)
        if session is not None:
            await delete_chat_session(session.session_id, user_id)
    except Exception:
        logger.exception(f"Could not undo the run of task {claim.task.id}")


async def _note_start(
    claim: TaskRunClaim, user_id: str, session_id: str, instructions: str | None
) -> None:
    started = claim.before.status == TaskStatus.TODO
    if claim.scheduled:
        text = "Started at its scheduled time"
    elif claim.retry:
        text = f"You asked for another try (attempt {claim.task.attempt})"
    else:
        text = "You started it"
    data = started_data(claim)
    if instructions:
        data["instructions"] = instructions
    await _note(
        claim.task.id,
        user_id,
        ActivityInput(
            kind=TaskActivityKind.STARTED if started else TaskActivityKind.RETRIED,
            text=started_line(claim, text),
            actor_type=TaskActor.SYSTEM if claim.scheduled else TaskActor.USER,
            chat_session_id=session_id,
            is_milestone=True,
            data=data,
        ),
    )


async def _note(task_id: str, user_id: str, entry: ActivityInput) -> None:
    """Add a line to the task's activity; the work matters more than the note."""
    try:
        await task_record_db().add_activity(task_id, user_id, entry)
    except Exception as e:
        logger.warning(f"Could not note activity on task {task_id}: {e}")
