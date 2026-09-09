"""Hand a task *over* to another expert — a transfer, not a delegation.

``delegate_to_expert`` borrows a teammate and waits for their answer, so the
caller still owns the task. A handoff gives it away: the receiving expert owns
the outcome and reports to the user, and this tool returns as soon as the work
is queued rather than waiting on a result.

The mechanics are the same queue-backed sub-session — target scope, memory and
budget — so the guards are too: never yourself, never an archived or
budget-paused teammate, and never past the shared chain bound
(:func:`expert_delegation.chain_refusal`, since a handoff hop costs the same
session-plus-turn as a delegated one and writes the same provenance to walk).
Provenance records the delegation fields (so the sub is attributable) plus
``handed_off_from_expert_id``, which tells the receiver the task is now theirs.

The *response* is deliberately not delegation's. A handoff is terminal for the
caller: it gets no result and — by design, see
``get_sub_session_result._in_caller_scope`` — cannot poll the sub it gave
away. So this tool builds its own ``status="transferred"`` response rather
than reuse ``response_from_outcome``, whose queued/running wording would send
the model to a poll that answers "no such sub-session".
"""

import logging
from typing import Any

from backend.api.features.experts.models import Expert
from backend.copilot.active_turns import running_turn_limit_message
from backend.copilot.context import get_current_permissions
from backend.copilot.model import (
    ChatSession,
    child_session_origin,
    create_chat_session,
    delete_chat_session,
)
from backend.copilot.sdk.session_waiter import (
    SessionOutcome,
    run_copilot_turn_via_queue,
)
from backend.data.db_accessors import experts_db

from .base import BaseTool
from .expert_delegation import (
    chain_refusal,
    resolve_target_expert,
    safe_caller_name,
    unknown_target_message,
)
from .models import (
    DelegatedExpertInfo,
    ErrorResponse,
    SubSessionStatusResponse,
    ToolResponseBase,
)
from .run_sub_session import _sub_session_link, apply_delegated_expert

logger = logging.getLogger(__name__)

# Outcomes meaning the task now sits in the target's own session — dispatched,
# already picked up, or appended to a turn in flight there. Anything else (the
# concurrent-turn cap, a failed dispatch) means nothing moved.
_OWNERSHIP_TAKEN: frozenset[SessionOutcome] = frozenset(
    {"running", "queued", "completed"}
)


class _HandoffRefused(Exception):
    """The target cannot own this task; carries the model-facing reason.

    Typed control flow so the caller checks one exception type instead of
    narrowing a returned union by ``isinstance``.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class HandoffToExpertTool(BaseTool):
    """Transfer ownership of a task to another expert on the team."""

    @property
    def name(self) -> str:
        return "handoff_to_expert"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Hand a task OVER to a different expert when it belongs to them, "
            "not you. They own it and report to the user themselves — you "
            "get no result back. Use delegate_to_expert when you need their "
            "answer to finish your own work."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expert_id": {
                    "type": "string",
                    "description": (
                        "Teammate taking it over, from <team_context>. Not you."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The task, written for them. They cannot see this "
                        "conversation or ask you follow-ups."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": "Optional background; what you already tried.",
                    "default": "",
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
        context: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return self._error("Authentication required", session)
        target_id = expert_id.strip()
        refusal = _request_refusal(target_id, prompt, session.expert_id)
        if refusal is not None:
            return self._error(refusal, session)
        try:
            target = await self._resolve_target(user_id, target_id, session.expert_id)
        except _HandoffRefused as refused:
            return self._error(refused.message, session)
        if target.id == session.expert_id:
            # A name reference can resolve back to the caller even though the
            # raw-id self check in _request_refusal passed.
            return self._error(
                "You are that expert — the task is already yours. Do it, or "
                "use run_sub_session to isolate it in a fresh context.",
                session,
            )
        chain = await chain_refusal(user_id, session, target)
        if chain is not None:
            return self._error(chain, session)
        return await self._transfer(user_id, session, target, prompt, context)

    async def _transfer(
        self,
        user_id: str,
        session: ChatSession,
        target: Expert,
        prompt: str,
        context: str,
    ) -> ToolResponseBase:
        """Open the receiving thread, queue the task, and hand ownership over."""
        inner = await create_chat_session(
            user_id,
            dry_run=session.dry_run,
            llm_auth_provider=session.metadata.llm_auth_provider,
            llm_credential_id=session.metadata.llm_credential_id,
            expert_id=target.id,
            delegated_by_expert_id=session.expert_id,
            delegated_by_session_id=session.session_id,
            handed_off_from_expert_id=session.expert_id,
            origin=child_session_origin(session.metadata),
        )
        outcome = await self._queue_task(
            user_id, session, inner.session_id, prompt, context
        )
        if outcome not in _OWNERSHIP_TAKEN:
            # Nothing moved (that is what the outcome means), so the thread we
            # opened for the target holds no work — leaving it behind would
            # show them an empty handoff that never happened. Best-effort:
            # a failed cleanup must not turn a refusal into an error.
            try:
                await delete_chat_session(inner.session_id, user_id)
            except Exception:
                logger.warning(
                    "Failed to clean up unused handoff session %s",
                    inner.session_id,
                    exc_info=True,
                )
            return self._error(_refused_transfer_message(target.name, outcome), session)
        return apply_delegated_expert(
            _transferred_response(
                inner_session_id=inner.session_id,
                parent_session_id=session.session_id,
                target_name=target.name,
            ),
            # Identity for the ToolChain card, so it names the new owner.
            DelegatedExpertInfo(
                id=target.id,
                name=target.name,
                role=target.role,
                avatar_url=target.avatar_url,
                color=target.color,
            ),
        )

    async def _queue_task(
        self,
        user_id: str,
        session: ChatSession,
        inner_session_id: str,
        prompt: str,
        context: str,
    ) -> SessionOutcome:
        """Push the framed task onto the target's session. Never waits."""
        caller = await self._caller_name(user_id, session.expert_id)
        outcome, _result = await run_copilot_turn_via_queue(
            session_id=inner_session_id,
            user_id=user_id,
            message=_transfer_message(caller, context, prompt),
            timeout=0,
            permissions=get_current_permissions(),
            tool_call_id=(
                f"handoff:{session.session_id}" if session.session_id else "handoff"
            ),
            tool_name="handoff_to_expert",
        )
        return outcome

    def _error(self, message: str, session: ChatSession) -> ErrorResponse:
        return ErrorResponse(message=message, session_id=session.session_id)

    async def _resolve_target(
        self, user_id: str, target_id: str, caller_expert_id: str | None
    ) -> Expert:
        """Resolve the teammate, refusing anyone who can't safely own work."""
        try:
            target = await resolve_target_expert(user_id, target_id)
        except Exception as e:
            logger.warning(f"Handoff target lookup failed for {target_id}: {e}")
            raise _HandoffRefused(
                "Could not reach that expert right now. Try again."
            ) from e
        if target is None or target.is_archived:
            raise _HandoffRefused(
                await unknown_target_message(user_id, target_id, caller_expert_id)
            )
        if target.schedules_paused_at is not None:
            raise _HandoffRefused(
                f"{target.name} is paused (budget guardrail or archive) and "
                "cannot take the task until the user resumes them."
            )
        return target

    async def _caller_name(self, user_id: str, caller_expert_id: str | None) -> str:
        if caller_expert_id is None:
            return "AutoPilot"
        try:
            caller = await experts_db().get_expert(
                user_id, caller_expert_id, include_workflows=False
            )
        except Exception as e:
            logger.warning(f"Handing-off expert lookup failed: {e}")
            return "a teammate"
        return caller.name if caller else "a teammate"


def _request_refusal(
    target_id: str, prompt: str, caller_expert_id: str | None
) -> str | None:
    """Why this handoff can't even be attempted, or ``None`` if it can."""
    if not target_id:
        return "expert_id is required"
    if not prompt.strip():
        return "prompt is required"
    if caller_expert_id is None:
        # The ``experts`` tool group already hides and refuses this tool for a
        # plain Autopilot session, so this is defence in depth — but the
        # failure it prevents is silent rather than loud: ``_transfer`` would
        # persist ``handed_off_from_expert_id`` as JSON null while still
        # setting ``delegated_by_session_id``, and the Home pending-question
        # predicate (``copilot/db.py``) reads the first with ``->>``, whose
        # NULL then fails the ``IS NOT NULL`` re-admit arm. The receiving
        # expert's question would vanish from Home with nothing logged.
        return (
            "Only an expert can hand a task over. Delegate it with "
            "delegate_to_expert, or tell the user which expert should own it."
        )
    if target_id == caller_expert_id:
        return (
            "You are that expert — the task is already yours. Do it, or "
            "use run_sub_session to isolate it in a fresh context."
        )
    return None


def _transferred_response(
    *,
    inner_session_id: str,
    parent_session_id: str | None,
    target_name: str,
) -> SubSessionStatusResponse:
    """The terminal handoff contract: ownership moved, nothing to poll.

    No poll instruction on purpose — ``_in_caller_scope`` denies the
    handing-off session any read on the sub, so pointing the model at
    ``get_sub_session_result`` earns it a false "no sub-session with id X"
    and the user never learns the receiving thread exists. ``elapsed_seconds``
    stays unset for the same reason: this tool waits for nothing.
    """
    link = _sub_session_link(inner_session_id)
    return SubSessionStatusResponse(
        message=(
            f"{target_name} owns this now and will report to the user "
            f"directly.{f' Follow along at {link}.' if link else ''}"
        ),
        session_id=parent_session_id,
        status="transferred",
        sub_session_id=inner_session_id,
        sub_autopilot_session_id=inner_session_id,
        sub_autopilot_session_link=link,
    )


def _refused_transfer_message(target_name: str, outcome: SessionOutcome) -> str:
    """Say plainly that nothing moved, so the model doesn't announce a handoff
    that never happened and then drop the task."""
    if outcome == "rejected_concurrent_turn_cap":
        return (
            f"The handoff to {target_name} did not happen — the task is still "
            f"yours. {running_turn_limit_message()}"
        )
    return (
        f"The handoff to {target_name} did not happen — their session could not "
        "take the task, so it is still yours. Do it yourself or try again."
    )


def _transfer_message(caller: str, context: str, prompt: str) -> str:
    """Frame the task as transferred, not borrowed.

    Delegation's preamble asks the receiver to report back to the teammate who
    called. A handoff has no one to report to, so the receiver is told to own
    the task and speak to the user directly.
    """
    name = safe_caller_name(caller)
    preamble = (
        f"[Task handed to you by {name}, a teammate on this user's team. It "
        "is yours now: they are not waiting on a report and cannot answer "
        "follow-ups. Take it to completion and tell the user the outcome "
        "yourself. If something only the user can provide is missing, ask "
        "them.]"
    )
    if context.strip():
        preamble += f"\n\n[Context: {context.strip()}]"
    return f"{preamble}\n\n{prompt}"
