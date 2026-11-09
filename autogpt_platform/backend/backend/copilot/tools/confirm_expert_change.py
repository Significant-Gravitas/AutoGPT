"""Apply a previewed hire or raise by its one-time confirmation_id.

Step 2 of the confirm-gated team-change flow. One confirm tool serves both
kinds: it takes nothing but the id, so the applied change is exactly what the
user saw. The id is single-use and bound to the Otto session that
produced it. A hire offered with its task also starts that task when the id
confirmed is the "hire and start" one (see ``hire_handoff``).
"""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.data.db_accessors import expert_kickoff_db
from backend.data.redis_client import get_redis_async

from .base import BaseTool
from .expert_proposal import (
    apply_proposal,
    autopilot_session_guard,
    load_bound_proposal,
)
from .hire_handoff import hand_off_to_new_hire, with_hire_only
from .models import ErrorResponse, ExpertChangeAppliedResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_PROPOSAL_FIELDS = (
    "template_id",
    "name",
    "role",
    "tagline",
    "about",
    "boundaries",
    "voice_preferences",
    "weekly_budget",
)


class ConfirmExpertChangeTool(BaseTool):
    """Create the expert previewed by hire_expert or raise_expert."""

    @property
    def name(self) -> str:
        return "confirm_expert_change"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Apply a hire_expert or raise_expert proposal after the user has "
            "approved it. Takes only the confirmation_id and creates exactly "
            "the previewed expert; the id is single-use."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "confirmation_id": {
                    "type": "string",
                    "description": "The id returned by hire_expert/raise_expert.",
                    "entity": "team_change",
                },
            },
            "required": ["confirmation_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        confirmation_id: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if error := autopilot_session_guard(user_id, session):
            return error
        assert user_id is not None

        if any(field in kwargs for field in _PROPOSAL_FIELDS):
            return ErrorResponse(
                message=(
                    "confirm_expert_change creates exactly the previewed "
                    "expert and does not accept new values. Call tool:hire_expert "
                    "or tool:raise_expert to propose something different."
                ),
                session_id=session_id,
            )
        if not confirmation_id:
            return ErrorResponse(
                message=(
                    "A confirmation_id from hire_expert or raise_expert is required."
                ),
                session_id=session_id,
            )

        proposal = await load_bound_proposal(
            await get_redis_async(),
            confirmation_id,
            user_id,
            session,
        )
        if isinstance(proposal, ErrorResponse):
            return proposal
        applied = await apply_proposal(user_id, session_id, proposal)
        task = proposal.handoff
        if (
            isinstance(applied, ExpertChangeAppliedResponse)
            and applied.kind == "hire"
            and not (task is not None and proposal.hand_off)
        ):
            # No page follows a chat hire into the new expert's thread, so
            # the server sends its kickoff. A hand-off opens that thread with
            # the task instead.
            await _send_kickoff(user_id, applied.expert.id, session)
        if task is None or not isinstance(applied, ExpertChangeAppliedResponse):
            return applied
        if not proposal.hand_off:
            return with_hire_only(applied, task)
        return await hand_off_to_new_hire(user_id, session, applied, task)


async def _send_kickoff(user_id: str, expert_id: str, session: ChatSession) -> None:
    """Best-effort: the hire is already applied, so a failed RPC to the
    database manager (restarting, timed out) must not turn it into an error
    Otto reports as a failed hire. The expert then introduces itself the
    first time it is opened, as before."""
    try:
        await expert_kickoff_db().start_expert_kickoff(
            user_id,
            expert_id,
            organization_id=session.organization_id,
            team_id=session.team_id,
        )
    except Exception:
        logger.warning(
            f"Kickoff for chat-hired expert #{expert_id} could not be sent",
            exc_info=True,
        )
