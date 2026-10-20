"""Accept a bounded background subtask without waiting for its model work."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from backend.copilot.active_turns import get_inflight_turn_limit
from backend.copilot.background_task import (
    TASK_SECONDS,
    BackgroundContext,
    BackgroundTask,
    operation_id,
    task_metadata,
)
from backend.copilot.context import (
    get_current_envelope,
    get_current_permissions,
    get_current_turn_id,
)
from backend.copilot.credential_selection import (
    CredentialPin,
    selected_credentials,
    turn_credential_pins,
)
from backend.copilot.llm_attribution import is_turn_impersonated
from backend.copilot.model import (
    ChatSession,
    _get_session_lock,
    create_chat_session,
    delete_chat_session,
    invalidate_session_cache,
)
from backend.copilot.tool_call_context import current_tool_call_id
from backend.copilot.tree import (
    SpawnRequest,
    TreeRefusal,
    admit_turn,
    derive_child_envelope,
    release_turn,
)
from backend.copilot.trial_cost_context import capture_cost_context
from backend.copilot.turn_queue import count_inflight_turns, dispatch_next_for_user
from backend.data.db_accessors import background_task_db
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)


async def start_background_subtask(
    user_id: str,
    parent: ChatSession,
    prompt: str,
) -> BackgroundTask:
    if not await is_feature_enabled(Flag.COPILOT_BACKGROUND_SUBTASKS, user_id):
        raise TreeRefusal("Background subtasks are not enabled for this account.")
    envelope, call_id, turn_id = (
        get_current_envelope(),
        current_tool_call_id(),
        get_current_turn_id(),
    )
    if envelope is None or not call_id or not turn_id:
        raise TreeRefusal(
            "This subtask's limits could not be preserved. Start it again."
        )
    task_id = operation_id(f"{parent.session_id}:{turn_id}", call_id)
    existing = await background_task_db().get_background_task(task_id, user_id)
    if existing:
        return existing
    if await count_inflight_turns(user_id) >= get_inflight_turn_limit():
        raise TreeRefusal(
            "Too many tasks are running or queued. Try again when one finishes."
        )
    child_envelope = derive_child_envelope(
        envelope,
        SpawnRequest(shares_memory=True, max_seconds=TASK_SECONDS),
        spawner_permissions=get_current_permissions(),
    )
    await admit_turn(child_envelope, user_id=user_id)
    child: ChatSession | None = None
    try:
        child = await create_chat_session(
            user_id=user_id,
            dry_run=parent.dry_run,
            organization_id=parent.organization_id,
            team_id=parent.team_id,
            expert_id=parent.expert_id,
            origin="automation",
            delegated_by_session_id=parent.session_id,
            delegated_by_expert_id=parent.expert_id,
            llm_auth_provider=parent.metadata.llm_auth_provider,
            llm_credential_id=parent.metadata.llm_credential_id,
        )
        task = BackgroundTask(
            id=task_id,
            user_id=user_id,
            parent_session_id=parent.session_id,
            child_session_id=child.session_id,
            launch_turn_id=turn_id,
            child_turn_id=operation_id(task_id, "child"),
            deadline=child_envelope.deadline_at
            or datetime.now(UTC) + timedelta(seconds=TASK_SECONDS),
            context=BackgroundContext(
                envelope=envelope,
                child_envelope=child_envelope,
                permissions=get_current_permissions(),
                cost_context=await capture_cost_context(user_id),
                organization_id=parent.organization_id,
                team_id=parent.team_id,
                expert_id=parent.expert_id,
                llm_auth_provider=parent.metadata.llm_auth_provider,
                llm_credential_id=parent.metadata.llm_credential_id,
                credential_pins={
                    key: CredentialPin(id=value)
                    for key, value in (
                        await selected_credentials(parent.session_id)
                    ).items()
                }
                | dict(turn_credential_pins()),
                impersonated=is_turn_impersonated(),
            ),
        )
        async with _get_session_lock(child.session_id) as locked:
            if not locked:
                raise RuntimeError("Could not queue this subtask. Try again.")
            await background_task_db().create_background_task(
                task, prompt, task_metadata(task, "child", child_envelope)
            )
    except BaseException as error:
        # An ambiguous commit is recovered from its durable operation identity.
        committed = await asyncio.shield(
            background_task_db().get_background_task(task_id, user_id)
        )
        if committed is None or (
            child and committed.child_session_id != child.session_id
        ):
            await release_turn(child_envelope)
            if child:
                await delete_chat_session(child.session_id, user_id)
        if committed is not None:
            if isinstance(error, asyncio.CancelledError):
                await asyncio.shield(
                    background_task_db().cancel_background_tasks(
                        user_id, parent.session_id, turn_id
                    )
                )
            else:
                return committed
        raise
    await invalidate_session_cache(child.session_id)
    try:
        await dispatch_next_for_user(user_id)
    except Exception:
        logger.exception(f"Background subtask {task.id} awaits dispatcher recovery")
    return task
