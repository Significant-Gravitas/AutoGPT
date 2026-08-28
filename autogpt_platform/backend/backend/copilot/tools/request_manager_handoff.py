import logging
from typing import Any, Literal, cast

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
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import ErrorResponse, ManagerHandoffRequestedResponse, ToolResponseBase
from .run_sub_session import _sub_session_link

logger = logging.getLogger(__name__)

_ACCEPTED: frozenset[SessionOutcome] = frozenset({"queued", "running", "completed"})


class RequestManagerHandoffTool(BaseTool):
    @property
    def name(self) -> str:
        return "request_manager_handoff"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Ask AutoPilot to take ownership of an out-of-scope direct request "
            "and route it to the right teammate. Use this instead of asking the "
            "founder how to coordinate the team. During delegated work, report "
            "blocked_manager instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Outcome that needs a different owner.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why it is outside this expert's role.",
                },
                "attempts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Useful work already attempted.",
                },
                "recommended_expert": {
                    "type": "string",
                    "description": "Optional suggested teammate name or role.",
                },
            },
            "required": ["task", "reason"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        task: str = "",
        reason: str = "",
        attempts: list[str] | None = None,
        recommended_expert: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if not await _enabled(user_id):
            return self._error("Manager routing is not available.", session)
        if user_id is None:
            return self._error("Authentication required.", session)
        if session.expert_id is None:
            return self._error(
                "This is already an AutoPilot session. Route the work directly.",
                session,
            )
        if (
            session.metadata.delegated_by_session_id is not None
            and session.metadata.handed_off_from_expert_id is None
        ):
            return self._error(
                "This work already reports to AutoPilot. Use "
                "report_delegated_result(status='blocked_manager') with what "
                "you tried and a recommended fallback.",
                session,
            )
        task = task.strip()[:2_000]
        reason = reason.strip()[:1_000]
        if not task or not reason:
            return self._error("Task and routing reason are required.", session)

        manager: ChatSession | None = None
        try:
            expert = await experts_db().get_expert(
                user_id, session.expert_id, include_workflows=False
            )
            expert_name = expert.name if expert else "an expert"
            manager = await create_chat_session(
                user_id,
                dry_run=session.dry_run,
                organization_id=session.organization_id,
                team_id=session.team_id,
                origin=child_session_origin(session.metadata),
                llm_auth_provider=session.metadata.llm_auth_provider,
                llm_credential_id=session.metadata.llm_credential_id,
                delegated_by_expert_id=session.expert_id,
                delegated_by_session_id=session.session_id,
                handed_off_from_expert_id=session.expert_id,
            )
            outcome, _ = await run_copilot_turn_via_queue(
                session_id=manager.session_id,
                user_id=user_id,
                message=_manager_message(
                    expert_name=expert_name,
                    task=task,
                    reason=reason,
                    attempts=attempts or [],
                    recommended_expert=recommended_expert,
                ),
                timeout=0,
                permissions=get_current_permissions(),
                tool_call_id=f"manager-handoff:{session.session_id}",
                tool_name=self.name,
            )
        except Exception:
            logger.warning("Could not request manager routing", exc_info=True)
            if manager is not None:
                try:
                    await delete_chat_session(manager.session_id, user_id)
                except Exception:
                    logger.warning(
                        "Could not clean up failed manager handoff", exc_info=True
                    )
            return self._error("AutoPilot could not take the routing request.", session)

        if outcome not in _ACCEPTED:
            try:
                await delete_chat_session(manager.session_id, user_id)
            except Exception:
                logger.warning("Could not clean up manager handoff", exc_info=True)
            return self._error(
                "AutoPilot could not take the routing request. Keep ownership "
                "and continue with the safest useful fallback.",
                session,
            )

        status = cast(Literal["queued", "running", "completed"], outcome)
        link = _sub_session_link(manager.session_id)
        return ManagerHandoffRequestedResponse(
            message=(
                "AutoPilot now owns routing this request and will involve the "
                "right teammate. The founder does not need to coordinate it."
            ),
            session_id=session.session_id,
            status=status,
            manager_session_id=manager.session_id,
            manager_session_link=link,
        )

    @staticmethod
    def _error(message: str, session: ChatSession) -> ErrorResponse:
        return ErrorResponse(message=message, session_id=session.session_id)


async def _enabled(user_id: str | None) -> bool:
    if user_id is None:
        return False
    try:
        return await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False)
    except Exception:
        logger.warning("Could not resolve manager routing flag", exc_info=True)
        return False


def _manager_message(
    *,
    expert_name: str,
    task: str,
    reason: str,
    attempts: list[str],
    recommended_expert: str,
) -> str:
    attempted = "; ".join(item.strip()[:500] for item in attempts[:10] if item.strip())
    recommendation = recommended_expert.strip()[:200] or "No recommendation"
    return (
        f"[Manager routing request from {expert_name}. You are AutoPilot and now "
        "own the outcome. Read the active shared project context, resolve the "
        "correct owner, and hand the task to the best teammate with all relevant "
        "decisions and artifacts. Answer from existing context when possible. Ask "
        "the founder only if a genuine founder-held decision remains.]\n\n"
        f"Task: {task}\n"
        f"Why it needs routing: {reason}\n"
        f"What was already tried: {attempted or 'Nothing useful yet'}\n"
        f"Suggested owner: {recommendation}"
    )
