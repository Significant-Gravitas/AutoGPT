"""``start_task`` with ``on_event``: the task waits and runs once each time
the event happens ("when a PR is opened on acme/web, review it").

It is a repeating task whose next start is set by the event, not a cron: the
webhook is set up now (``task_watch_setup``), the task waits in TODO with no
start time, and the ingress makes it due when a matching event arrives, so the
due-task sweep starts it under the chat that asked and its result comes back
here. Nothing runs when it is saved.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from backend.blocks import get_block
from backend.blocks._base import AnyBlockSchema
from backend.copilot import turn_end
from backend.copilot.context import is_unattended_turn
from backend.copilot.model import ChatSession
from backend.data.db_accessors import task_record_db, task_schedule_db, task_watch_db
from backend.data.model import CredentialsMetaInput
from backend.data.task import Task
from backend.data.task_schedule_meta import TaskRepeat, TaskWatch

from .event_triggers import EventTrigger, resolve_event_trigger
from .expert_scope import annotate_expert_grants
from .helpers import resolve_block_credentials, unattended_missing_credentials_error
from .models import (
    ErrorResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)
from .task_models import TaskInfo, TaskStartedResponse
from .utils import build_missing_credentials_from_field_info

logger = logging.getLogger(__name__)


async def start_on_event(
    user_id: str,
    session: ChatSession,
    on_event: dict[str, Any],
    duplicate: Task | None,
    create: Callable[[str], Awaitable[Task]],
) -> ToolResponseBase:
    """Save a task that watches for ``on_event`` and wrap the turn up; or
    the card to connect the account the event comes from."""
    if duplicate is not None:
        return _error(
            f"'{duplicate.title}' is already a task from this chat; no second "
            "one was made. Tell the user it is set up.",
            session,
        )
    if await task_record_db().work_tasks(session.session_id, user_id):
        return _error(
            "A task that starts on an event is for a task asked for in a chat, "
            "not for a part of a task.",
            session,
        )
    trigger = resolve_event_trigger(
        str(on_event.get("provider") or ""),
        str(on_event.get("event") or ""),
        _config(on_event),
    )
    block = get_block(trigger.block_id)
    assert block is not None
    matched, missing = await resolve_block_credentials(
        user_id, block, {}, session.expert_id, session_id=session.session_id
    )
    if missing:
        return await _connect_card(user_id, session, block, matched)
    watch = await task_watch_db().setup_task_watch(
        user_id,
        block.id,
        _with_credentials(trigger, matched),
        trigger.line,
        session.organization_id,
        session.team_id,
    )
    task = await _save(user_id, session, watch, create)
    turn_end.request_wrap_up(session.session_id)
    return TaskStartedResponse(
        message=_watching_message(task, watch),
        task=TaskInfo.from_task(task),
        in_background=True,
        session_id=session.session_id,
    )


async def _save(
    user_id: str,
    session: ChatSession,
    watch: TaskWatch,
    create: Callable[[str], Awaitable[Task]],
) -> Task:
    """The task, waiting with its watch; the webhook is let go of when the
    task cannot be saved."""
    try:
        task = await create(f"Watching: {watch.line}")
        return await task_schedule_db().set_task_start(
            task.id,
            user_id,
            None,
            session.session_id,
            repeat=TaskRepeat(label=watch.line, watch=watch),
        )
    except BaseException:
        await task_watch_db().release_watch_webhook(user_id, watch)
        raise


def _config(on_event: dict[str, Any]) -> dict[str, Any]:
    config = on_event.get("config")
    return dict(config) if isinstance(config, dict) else {}


def _with_credentials(
    trigger: EventTrigger, matched: dict[str, CredentialsMetaInput]
) -> dict[str, Any]:
    credentials = {
        field: meta.model_dump(mode="json") for field, meta in matched.items()
    }
    return {**trigger.trigger_config, **credentials}


async def _connect_card(
    user_id: str,
    session: ChatSession,
    block: AnyBlockSchema,
    matched: dict[str, CredentialsMetaInput],
) -> ToolResponseBase:
    """The same card run_block shows for an account that is not connected."""
    entries = build_missing_credentials_from_field_info(
        block.input_schema.get_credentials_fields_info(), set(matched)
    )
    if is_unattended_turn():
        return await unattended_missing_credentials_error(
            "Starting on this event",
            entries,
            session.session_id,
            user_id,
            session.expert_id,
        )
    entries = await annotate_expert_grants(user_id, session.expert_id, entries)
    return SetupRequirementsResponse(
        message=(
            "The account these events come from is not connected. Connect it "
            "in the card below, then call start_task again with the same "
            "on_event; nothing was saved yet."
        ),
        session_id=session.session_id,
        setup_info=SetupInfo(
            agent_id=block.id,
            agent_name=block.name,
            user_readiness=UserReadiness(
                has_all_credentials=False,
                missing_credentials=entries,
                ready_to_run=False,
            ),
            requirements={
                "credentials": list(entries.values()),
                "inputs": [],
                "execution_modes": ["webhook"],
            },
        ),
    )


def _watching_message(task: Task, watch: TaskWatch) -> str:
    url_note = (
        " Events arrive when something calls its webhook URL. The URL is in "
        "the task's details, which open when the user clicks the card: do "
        "not print it and do not explain where to paste it unless asked."
        if watch.url
        else ""
    )
    return (
        f"'{task.title}' now watches: {watch.line}. Nothing runs now: each "
        "time it happens the task runs once in the background, its result "
        "comes back to this chat, and it goes back to watching until the user "
        f"pauses or stops it on the task.{url_note} Reply with one short "
        "sentence: that you are watching and will post here each time it "
        "happens. No description of the result, no setup steps. Do none of "
        "the task here."
    )


def _error(message: str, session: ChatSession) -> ErrorResponse:
    return ErrorResponse(
        message=message, error="start_task_failed", session_id=session.session_id
    )
