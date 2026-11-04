"""Live "current step" for a running task turn.

Dispatch writes ``work session -> (user, task)`` to Redis when a turn
starts, so the per-tool-call hook in ``BaseTool.execute`` costs one Redis
read and never touches the database outside a work chat. Work chats are not
all automation sessions: a task the user wrote for Otto runs in an
interactive one, so it can propose hires.
"""

import asyncio
import json
import logging
from datetime import datetime

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_KEY = "tasks:work_session:{}"
_TTL_SECONDS = 6 * 3600
_pending: set[asyncio.Task] = set()

_STEP_LABELS = {
    "web_search": "Searching the web",
    "web_fetch": "Reading a page",
    "bash_exec": "Running code",
    "run_agent": "Running a workflow",
    "run_capability": "Using a tool",
    "find_capability": "Looking for a tool",
    "write_workspace_file": "Writing a file",
    "read_workspace_file": "Reading a file",
    "delegate_to_expert": "Asking a teammate",
    "update_task_status": "Updating the task",
    "comment_on_task": "Commenting",
    "request_input": "Asking you",
    "link_artifact": "Attaching a result",
    "update_task": "Updating the task",
    "add_evidence": "Saving proof",
    "ask_user": "Asking you",
    "create_task": "Creating a sub-task",
}


def step_label(tool_name: str) -> str:
    return _STEP_LABELS.get(tool_name, tool_name.replace("_", " ").capitalize())


async def remember_work_session(
    session_id: str, user_id: str, task_id: str, started_at: datetime
) -> None:
    try:
        redis = await get_redis_async()
        await redis.set(
            _KEY.format(session_id),
            json.dumps(
                {
                    "user_id": user_id,
                    "task_id": task_id,
                    "started_at": started_at.isoformat(),
                }
            ),
            ex=_TTL_SECONDS,
        )
    except Exception:
        logger.warning("Could not remember work session %s", session_id)


def note_tool_call(session_id: str, tool_name: str) -> None:
    """Fire-and-forget: show *tool_name* as the task's current step."""
    if not session_id:
        return
    task = asyncio.create_task(_note(session_id, tool_name))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def _note(session_id: str, tool_name: str) -> None:
    from backend.api.features.tasks.access import tasks_store
    from backend.api.features.tasks.events import publish_task_event

    try:
        redis = await get_redis_async()
        raw = await redis.get(_KEY.format(session_id))
        if not raw:
            return
        ref = json.loads(raw)
        store = tasks_store()
        started = ref.get("started_at")
        await store.set_live(
            ref["task_id"],
            True,
            step_label(tool_name),
            datetime.fromisoformat(started) if started else None,
        )
        task = await store.get_task(ref["user_id"], ref["task_id"])
        await publish_task_event(
            ref["user_id"], "task_updated", task, task_id=ref["task_id"]
        )
    except Exception:
        logger.debug("Live step update failed for %s", session_id, exc_info=True)
