"""Worker-side identity, deadline, and completion fences for background turns."""

from datetime import UTC, datetime

from backend.copilot.background_task import BACKGROUND_STAGE_KEY, BACKGROUND_TASK_KEY
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.tree import readmit_turn
from backend.data.db_accessors import background_task_db


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
    if claimed:
        try:
            await readmit_turn(entry.envelope, user_id=entry.user_id)
        except Exception as error:
            await finish_background_execution(entry, str(error))
            raise
    return claimed


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
        and error == "Operation cancelled"
        and (entry.message_metadata or {}).get(BACKGROUND_STAGE_KEY) != "child"
    ):
        await background_task_db().cancel_background_tasks(
            entry.user_id, entry.session_id, entry.turn_id
        )
    if entry.user_id and (entry.message_metadata or {}).get(BACKGROUND_TASK_KEY):
        await background_task_db().finish_background_turn(
            entry.user_id, entry.session_id, entry.turn_id, error
        )
