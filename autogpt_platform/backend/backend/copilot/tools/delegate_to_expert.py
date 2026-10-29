"""Hand a task to a *different* hired expert on the user's team.

``run_sub_session`` spawns a sub that is always the caller's own scope — same
expert identity, same memory namespace, same budget. That makes it a context
isolator, not a hand-off: an expert cannot ask a teammate with different
skills, workflows, or integrations to do a piece of work.

This tool closes that gap. It creates a sub ``ChatSession`` bound to the
*target* expert and runs one turn on the same ``copilot_executor`` queue that
backs every other copilot turn, so the delegated work:

- speaks and acts as the target expert (``<expert_identity>`` is rebuilt from
  the target's Soul by the normal per-turn path),
- reads and writes the *target's* memory namespace
  (the session's ``expert_id`` selects its ``MemoryScope``),
- attributes the agent runs it starts to the *target's* weekly budget (only
  graph executions accrue weekly spend; the delegated conversation's own LLM
  cost does not), which is why a paused/archived teammate is refused here,
- inherits the caller's ``dry_run`` and LLM routing, and can only ever narrow
  the caller's tool permissions (``merged_with_parent``).

Provenance lives in the sub's session metadata rather than a new column:
``delegated_by_expert_id`` records who asked, and ``delegated_by_session_id``
doubles as the poll capability — ``get_sub_session_result`` accepts an
out-of-scope sub only when it names the caller's session there. Walking that
same chain upwards bounds how far a task may be passed on and catches an
expert handing work back to one already waiting on it
(:func:`expert_delegation.chain_refusal`, shared with ``handoff_to_expert``).
"""

import logging
import time
from typing import Any

from backend.api.features.experts.models import Expert
from backend.copilot.active_turns import spawn_limit_message
from backend.copilot.budget_signal import build_spawn_state_note
from backend.copilot.context import get_current_permissions
from backend.copilot.gate import child_autopilot_mode
from backend.copilot.model import (
    ChatSession,
    child_session_origin,
    create_chat_session,
    delete_chat_session,
    get_chat_session,
)
from backend.copilot.sdk.session_waiter import (
    TurnNotDispatched,
    run_copilot_turn_via_queue,
)
from backend.copilot.tree import SpawnRequest
from backend.data.db_accessors import experts_db

from .base import BaseTool
from .expert_delegation import (
    chain_refusal,
    resolve_target_expert,
    safe_caller_name,
    sent_from_metadata,
    unknown_target_message,
)
from .models import DelegatedExpertInfo, ErrorResponse, ToolResponseBase
from .run_sub_session import (
    MAX_SUB_SESSION_WAIT_SECONDS,
    apply_delegated_expert,
    discard_unused_sub_session,
    list_sub_workspace_files,
    response_from_outcome,
)

logger = logging.getLogger(__name__)


class DelegateToExpertTool(BaseTool):
    """Delegate a task to another hired expert and wait for their answer."""

    @property
    def name(self) -> str:
        return "delegate_to_expert"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Hand a task to a DIFFERENT expert on the user's team when it "
            "needs their skills, workflows, or integrations rather than "
            "yours. The teammate answers in their own voice, memory, and "
            "budget. Use run_sub_session instead for isolating your own "
            f"work. Waits up to wait_for_result sec (max "
            f"{MAX_SUB_SESSION_WAIT_SECONDS}); if not done, returns "
            "status=running + sub_session_id — poll via "
            "tool:get_sub_session_result."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {
                    "type": "string",
                    "description": (
                        "Teammate to delegate to: their expert id from "
                        "<team_context>, or their exact name if you don't "
                        "have the id. Must not be you."
                    ),
                    "entity": "expert_or_name",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The task, written for the teammate. Include the "
                        "context they need — they cannot see this "
                        "conversation."
                    ),
                    "title": "Task",
                },
                "system_context": {
                    "type": "string",
                    "description": "Optional context prepended to the prompt.",
                    "default": "",
                    "title": "Background",
                },
                "delegated_session_id": {
                    "type": "string",
                    "description": (
                        "Continue a prior delegation to this teammate; empty = new."
                    ),
                    "default": "",
                    "title": "Continues",
                    "entity": "chat_session",
                },
                "wait_for_result": {
                    "type": "integer",
                    "description": (
                        "Seconds to wait inline. 0 = return immediately. "
                        f"Clamped to {MAX_SUB_SESSION_WAIT_SECONDS}."
                    ),
                    "default": 60,
                    "title": "Waits up to",
                    "format": "seconds",
                },
            },
            "required": ["expert_id", "prompt"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        expert_id: str = "",
        prompt: str = "",
        system_context: str = "",
        delegated_session_id: str = "",
        wait_for_result: int = 60,
        **kwargs,
    ) -> ToolResponseBase:
        target_id = expert_id.strip()
        if not target_id:
            return self._error("expert_id is required", session)
        if not prompt.strip():
            return self._error("prompt is required", session)
        if user_id is None:
            return self._error("Authentication required", session)
        if target_id == session.expert_id:
            return self._error(
                "You are that expert — do the work yourself, or use "
                "run_sub_session to isolate it in a fresh context.",
                session,
            )

        target = await self._load_delegate_target(user_id, target_id, session)
        if isinstance(target, ErrorResponse):
            return target
        if target.id == session.expert_id:
            # A name reference can resolve back to the caller even though the
            # raw-id self check above passed.
            return self._error(
                "You are that expert — do the work yourself, or use "
                "run_sub_session to isolate it in a fresh context.",
                session,
            )

        refusal = await chain_refusal(user_id, session, target)
        if refusal is not None:
            return self._error(refusal, session)

        return await start_delegation(
            user_id=user_id,
            session=session,
            target=target,
            prompt=prompt,
            system_context=system_context,
            delegated_session_id=delegated_session_id.strip(),
            wait_for_result=wait_for_result,
        )

    def _error(self, message: str, session: ChatSession) -> ErrorResponse:
        return _error(message, session)

    async def _load_delegate_target(
        self, user_id: str, target_id: str, session: ChatSession
    ) -> Expert | ErrorResponse:
        """Resolve the teammate, refusing anyone who can't safely take work."""
        try:
            target = await resolve_target_expert(user_id, target_id)
        except Exception as e:
            logger.warning(f"Delegate target lookup failed for {target_id}: {e}")
            return self._error(
                "Could not reach that expert right now. Try again.", session
            )
        if target is None or target.is_archived:
            return self._error(
                await unknown_target_message(user_id, target_id, session.expert_id),
                session,
            )
        if (paused := paused_refusal(target)) is not None:
            return self._error(paused, session)
        return target


def paused_refusal(target: Expert) -> str | None:
    """Why a paused teammate can't take delegated work, or ``None``.

    Shared with ``hire_handoff``: re-hiring a template returns the existing
    hire unchanged, so a "new" hire can be one the budget guardrail paused.
    """
    if target.schedules_paused_at is None:
        return None
    return (
        f"{target.name} is paused (budget guardrail or archive) and "
        "cannot take delegated work until the user resumes them."
    )


async def start_delegation(
    *,
    user_id: str,
    session: ChatSession,
    target: Expert,
    prompt: str,
    system_context: str = "",
    delegated_session_id: str = "",
    wait_for_result: int = 60,
) -> ToolResponseBase:
    """Run *prompt* as *target* in a thread delegated from *session*.

    The part of ``delegate_to_expert`` after its checks. The caller has
    already resolved *target* and cleared ``chain_refusal``.
    """
    inner_session_id = await _resolve_session(
        user_id=user_id,
        session=session,
        target=target,
        delegated_session_id=delegated_session_id,
    )
    if isinstance(inner_session_id, ErrorResponse):
        return inner_session_id
    return await run_delegated_turn(
        user_id=user_id,
        session=session,
        target=target,
        inner_session_id=inner_session_id,
        prompt=prompt,
        system_context=system_context,
        opened_here=not delegated_session_id,
        wait_for_result=wait_for_result,
    )


async def run_delegated_turn(
    *,
    user_id: str,
    session: ChatSession,
    target: Expert,
    inner_session_id: str,
    prompt: str,
    system_context: str = "",
    opened_here: bool,
    wait_for_result: int,
) -> ToolResponseBase:
    """Queue *prompt* on a delegation thread and report what happened.

    Shared with ``hire_handoff``, which opens the thread itself first
    (``open_delegation_thread``) so it always knows where a task went, so a
    task handed to a new hire runs exactly like any other delegation: same
    provenance, permissions, mode and result contract. ``opened_here``
    drops a thread this call opened when its turn never reached it
    (``TurnNotDispatched`` is re-raised after).
    """
    caller = await _caller_name(user_id, session.expert_id)
    started_at = time.monotonic()
    try:
        outcome, result = await run_copilot_turn_via_queue(
            session_id=inner_session_id,
            user_id=user_id,
            message=_handoff_message(caller, system_context, prompt),
            timeout=max(0, min(wait_for_result, MAX_SUB_SESSION_WAIT_SECONDS)),
            permissions=get_current_permissions(),
            tool_call_id=(
                f"delegate:{session.session_id}" if session.session_id else "delegate"
            ),
            tool_name="delegate_to_expert",
            # A teammate keeps their own teammates; depth still bounds the chain.
            spawn=SpawnRequest(may_spawn=True),
            allow_queue=False,
            message_metadata=sent_from_metadata(session, caller),
        )
    except TurnNotDispatched:
        # Nothing reached the queue, so a thread opened for this task is
        # empty. Any other error, or a cancellation (which can land inside
        # the publish), may have left the task in it, so the thread stays.
        if opened_here:
            await _discard_thread(inner_session_id, user_id)
        raise
    elapsed = time.monotonic() - started_at
    discarded = opened_here and await discard_unused_sub_session(
        inner_session_id, user_id, outcome
    )
    if discarded:
        # The thread is gone; handing back its id would send the model to
        # poll a delegation that no longer exists.
        return _error(
            (
                spawn_limit_message()
                if outcome == "rejected_concurrent_turn_cap"
                else result.refusal or f"{target.name} could not start this task."
            ),
            session,
        )
    workspace_files = (
        await list_sub_workspace_files(user_id, inner_session_id)
        if outcome == "completed"
        else None
    )
    delegated = response_from_outcome(
        outcome=outcome,
        result=result,
        inner_session_id=inner_session_id,
        parent_session_id=session.session_id,
        elapsed=elapsed,
        workspace_files=workspace_files,
        actor=target.name,
    )
    delegated.message += await _spawn_note()
    return apply_delegated_expert(
        delegated,
        DelegatedExpertInfo(
            id=target.id,
            name=target.name,
            role=target.role,
            avatar_url=target.avatar_url,
            color=target.color,
        ),
    )


async def _spawn_note() -> str:
    """The tree's state for the result, or nothing: it is read after the task
    is queued, so failing here must not report that task as failed."""
    try:
        return await build_spawn_state_note()
    except Exception:
        logger.warning("Could not read the spawn state note", exc_info=True)
        return ""


async def _discard_thread(inner_session_id: str, user_id: str) -> None:
    """Remove a delegation thread nothing was sent to. Best-effort, like
    ``discard_unused_sub_session``: an empty thread is the lesser harm."""
    try:
        await delete_chat_session(inner_session_id, user_id)
    except Exception:
        logger.warning("Failed to remove an unused delegation thread", exc_info=True)


def _error(message: str, session: ChatSession) -> ErrorResponse:
    return ErrorResponse(message=message, session_id=session.session_id).from_outside()


async def _resolve_session(
    *,
    user_id: str,
    session: ChatSession,
    target: Expert,
    delegated_session_id: str,
) -> str | ErrorResponse:
    """Reuse a prior delegation thread with this teammate, or open one.

    Resuming is restricted to threads this session itself delegated, so a
    session can never read or steer another scope's conversation by
    guessing an id.
    """
    if not delegated_session_id:
        return await open_delegation_thread(user_id, session, target)

    prior = await get_chat_session(delegated_session_id)
    if (
        prior is None
        or prior.user_id != user_id
        or prior.expert_id != target.id
        or prior.metadata.delegated_by_session_id != session.session_id
        or prior.metadata.handed_off_from_expert_id is not None
    ):
        return _error(
            f"delegated_session_id {delegated_session_id} is not a "
            f"delegation you started with {target.name}. Leave it empty "
            "to open a fresh one.",
            session,
        )
    if (
        prior.metadata.llm_auth_provider != session.metadata.llm_auth_provider
        or prior.metadata.llm_credential_id != session.metadata.llm_credential_id
    ):
        return _error(
            f"That delegation thread with {target.name} runs on a "
            "different model connection than this chat does, so it "
            "cannot be resumed from here. Leave delegated_session_id "
            "empty to open a fresh one.",
            session,
        )
    return delegated_session_id


async def open_delegation_thread(
    user_id: str, session: ChatSession, target: Expert
) -> str:
    """Open a fresh thread for *target*, delegated from *session*."""
    new_session = await create_chat_session(
        user_id,
        dry_run=session.dry_run,
        llm_auth_provider=session.metadata.llm_auth_provider,
        llm_credential_id=session.metadata.llm_credential_id,
        expert_id=target.id,
        delegated_by_expert_id=session.expert_id,
        delegated_by_session_id=session.session_id,
        origin=child_session_origin(session.metadata),
        # Decided here, from the expert row already in hand, so the
        # thread cannot pick up a looser default written in between.
        inherited_autopilot_mode=child_autopilot_mode(session, target.autopilot_mode),
    )
    return new_session.session_id


async def _caller_name(user_id: str, caller_expert_id: str | None) -> str:
    """Who to introduce the hand-off as. Plain sessions are Otto."""
    if caller_expert_id is None:
        return "Otto"
    try:
        caller = await experts_db().get_expert(
            user_id, caller_expert_id, include_workflows=False
        )
    except Exception as e:
        logger.warning(f"Delegating expert lookup failed: {e}")
        return "a teammate"
    return caller.name if caller else "a teammate"


def _handoff_message(caller: str, system_context: str, prompt: str) -> str:
    """Frame the task so the teammate knows a colleague — not the user — asked.

    Without this the delegated prompt reads as the user speaking, and the
    teammate would address them directly and ask follow-up questions nobody
    is there to answer.
    """
    name = safe_caller_name(caller)
    preamble = (
        f"[Delegated task from {name}, a teammate on this user's team — not "
        "the user. They cannot see your thread, so report the outcome in your "
        "final message. If the task needs something only the user can "
        "provide, say what is missing instead of guessing.]"
    )
    if system_context.strip():
        preamble += f"\n\n[Context: {system_context.strip()}]"
    return f"{preamble}\n\n{prompt}"
