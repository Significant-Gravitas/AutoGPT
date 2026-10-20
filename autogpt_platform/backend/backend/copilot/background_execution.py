"""Worker-side identity, deadline, and completion fences for background turns."""

import logging
from datetime import UTC, datetime

from backend.copilot import stream_registry
from backend.copilot.background_task import (
    BACKGROUND_STAGE_KEY,
    BACKGROUND_TASK_KEY,
    BACKGROUND_TIMEOUT_MESSAGE,
    BackgroundStage,
)
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.model import invalidate_session_cache
from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.copilot.tree import readmit_turn
from backend.copilot.turn_queue import dispatch_next_for_user
from backend.data.db_accessors import background_task_db

logger = logging.getLogger(__name__)


async def claim_background_execution(entry: CoPilotExecutionEntry) -> bool:
    metadata = entry.message_metadata or {}
    task_id = metadata.get(BACKGROUND_TASK_KEY)
    if not task_id:
        return True
    stage = metadata.get(BACKGROUND_STAGE_KEY)
    if (
        not entry.user_id
        or stage not in ("child", "followup")
        or entry.envelope is None
    ):
        raise ValueError("The background subtask's saved limits are unavailable.")
    claimed = await background_task_db().claim_background_turn(
        task_id, entry.user_id, entry.session_id, entry.turn_id, stage
    )
    if not claimed:
        await _close_unstarted_background_turn(entry, task_id, stage)
        return False
    try:
        if background_timeout(entry) == 0:
            raise TimeoutError(BACKGROUND_TIMEOUT_MESSAGE)
        await readmit_turn(entry.envelope, user_id=entry.user_id)
    except Exception as error:
        message = str(error) or type(error).__name__
        try:
            await finish_background_execution(entry, message)
        except Exception:
            logger.exception("Could not persist background admission failure")
        finally:
            await stream_registry.mark_session_completed(
                entry.session_id, error_message=message, turn_id=entry.turn_id
            )
        raise
    return True


async def _close_unstarted_background_turn(
    entry: CoPilotExecutionEntry, task_id: str, stage: BackgroundStage
) -> None:
    if not entry.user_id:
        return
    task = await background_task_db().get_background_task(task_id, entry.user_id)
    if task is None:
        active = await stream_registry.get_session(entry.session_id)
        if active and (active.user_id, active.turn_id) == (
            entry.user_id,
            entry.turn_id,
        ):
            if await stream_registry.mark_session_completed(
                entry.session_id,
                error_message=CANCELLED_MESSAGE,
                turn_id=entry.turn_id,
            ):
                return
        if await background_task_db().release_deleted_background_turn(
            task_id, entry.user_id, entry.session_id, entry.turn_id
        ):
            await invalidate_session_cache(entry.session_id)
            await dispatch_next_for_user(entry.user_id)
        return
    if stage == "child":
        if (entry.session_id, entry.turn_id) != (
            task.child_session_id,
            task.child_turn_id,
        ) or task.started_at is not None:
            return
        error = {
            "cancelled": CANCELLED_MESSAGE,
            "timed_out": BACKGROUND_TIMEOUT_MESSAGE,
        }.get(task.outcome or "")
    else:
        if (entry.session_id, entry.turn_id) != (
            task.parent_session_id,
            task.wake_id,
        ) or task.followup_started_at is not None:
            return
        error = (
            CANCELLED_MESSAGE
            if task.wake_suppressed or task.followup_state == "suppressed"
            else None
        )
    if error:
        await stream_registry.mark_session_completed(
            entry.session_id, error_message=error, turn_id=entry.turn_id
        )


def background_timeout(entry: CoPilotExecutionEntry) -> float | None:
    if not (entry.message_metadata or {}).get(BACKGROUND_TASK_KEY):
        return None
    if entry.envelope is None or entry.envelope.deadline_at is None:
        raise ValueError("The background subtask's deadline is unavailable.")
    return max(0, (entry.envelope.deadline_at - datetime.now(UTC)).total_seconds())


async def finish_background_execution(
    entry: CoPilotExecutionEntry, error: str | None
) -> None:
    if (
        entry.user_id
        and error == CANCELLED_MESSAGE
        and (entry.message_metadata or {}).get(BACKGROUND_STAGE_KEY) != "child"
    ):
        # A cancelled parent turn also suppresses its children's later wakes.
        await background_task_db().cancel_background_tasks(
            entry.user_id, entry.session_id, entry.turn_id
        )
    if entry.user_id and (entry.message_metadata or {}).get(BACKGROUND_TASK_KEY):
        await background_task_db().finish_background_turn(
            entry.user_id, entry.session_id, entry.turn_id, error
        )
