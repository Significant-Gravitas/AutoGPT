"""Hand a task *over* to another expert — a transfer, not a delegation.

``delegate_to_expert`` borrows a teammate and waits for their answer, so the
caller still owns the task. A handoff gives it away: the receiving expert owns
the outcome and reports to the user, and this tool returns as soon as the work
is queued rather than waiting on a result.

The mechanics are the same queue-backed sub-session — target scope, target
memory, target budget — so the guards are the same too: never yourself, never
an archived or budget-paused teammate. Provenance records both the delegation
fields (so the sub is attributable) and ``handed_off_from_expert_id``, which
is what tells the receiver the task is now theirs.
"""

import logging
import time
from typing import Any

from backend.api.features.experts.models import Expert
from backend.copilot.context import get_current_permissions
from backend.copilot.model import ChatSession, create_chat_session
from backend.copilot.sdk.session_waiter import run_copilot_turn_via_queue
from backend.data.db_accessors import experts_db

from .base import BaseTool
from .models import DelegatedExpertInfo, ErrorResponse, ToolResponseBase
from .run_sub_session import apply_delegated_expert, response_from_outcome

logger = logging.getLogger(__name__)

# Caller names are user-authored; keep them to a single short line so a
# crafted name can't forge extra framing inside the handoff preamble.
_CALLER_NAME_LIMIT = 80


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
        target_id = expert_id.strip()
        if not target_id:
            return self._error("expert_id is required", session)
        if not prompt.strip():
            return self._error("prompt is required", session)
        if user_id is None:
            return self._error("Authentication required", session)
        if target_id == session.expert_id:
            return self._error(
                "You are that expert — the task is already yours. Do it, or "
                "use run_sub_session to isolate it in a fresh context.",
                session,
            )

        target = await self._load_target(user_id, target_id, session)
        if isinstance(target, ErrorResponse):
            return target

        inner = await create_chat_session(
            user_id,
            dry_run=session.dry_run,
            llm_auth_provider=session.metadata.llm_auth_provider,
            llm_credential_id=session.metadata.llm_credential_id,
            expert_id=target.id,
            delegated_by_expert_id=session.expert_id,
            delegated_by_session_id=session.session_id,
            handed_off_from_expert_id=session.expert_id,
        )

        caller = await self._caller_name(user_id, session.expert_id)
        started_at = time.monotonic()
        outcome, result = await run_copilot_turn_via_queue(
            session_id=inner.session_id,
            user_id=user_id,
            message=_transfer_message(caller, context, prompt),
            timeout=0,
            permissions=get_current_permissions(),
            tool_call_id=(
                f"handoff:{session.session_id}" if session.session_id else "handoff"
            ),
            tool_name="handoff_to_expert",
        )
        return apply_delegated_expert(
            response_from_outcome(
                outcome=outcome,
                result=result,
                inner_session_id=inner.session_id,
                parent_session_id=session.session_id,
                elapsed=time.monotonic() - started_at,
            ),
            DelegatedExpertInfo(
                id=target.id,
                name=target.name,
                role=target.role,
                avatar_url=target.avatar_url,
                color=target.color,
            ),
        )

    def _error(self, message: str, session: ChatSession) -> ErrorResponse:
        return ErrorResponse(message=message, session_id=session.session_id)

    async def _load_target(
        self, user_id: str, target_id: str, session: ChatSession
    ) -> Expert | ErrorResponse:
        """Resolve the teammate, refusing anyone who can't safely own work."""
        try:
            target = await experts_db().get_expert(
                user_id, target_id, include_workflows=False
            )
        except Exception as e:
            logger.warning(f"Handoff target lookup failed for {target_id}: {e}")
            return self._error(
                "Could not reach that expert right now. Try again.", session
            )
        if target is None or target.is_archived:
            return self._error(
                f"No active expert with id {target_id} on this team. Pick one "
                "from <team_context>.",
                session,
            )
        if target.schedules_paused_at is not None:
            return self._error(
                f"{target.name} is paused (budget guardrail or archive) and "
                "cannot take the task until the user resumes them.",
                session,
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


def _transfer_message(caller: str, context: str, prompt: str) -> str:
    """Frame the task as transferred, not borrowed.

    The delegation preamble asks the receiver to report back to the teammate
    who called. A handoff has no one to report to — the caller has moved on —
    so the receiver is told to own the task and speak to the user directly.
    """
    name = " ".join(caller.split())[:_CALLER_NAME_LIMIT] or "a teammate"
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
