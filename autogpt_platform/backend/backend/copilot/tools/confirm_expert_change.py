"""Apply a previewed hire or raise by its one-time confirmation_id.

Step 2 of the confirm-gated team-change flow. One confirm tool serves both
kinds: it takes nothing but the id, so the applied change is exactly what the
user saw. The id is single-use and bound to the Autopilot session that
produced it.
"""

from typing import Any

from backend.copilot.model import ChatSession
from backend.data.redis_client import get_redis_async

from .base import BaseTool
from .expert_proposal import (
    apply_proposal,
    autopilot_session_guard,
    load_bound_proposal,
)
from .models import ErrorResponse, ToolResponseBase

_PROPOSAL_FIELDS = (
    "template_id",
    "name",
    "role",
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
                    "expert and does not accept new values. Call hire_expert "
                    "or raise_expert to propose something different."
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
        return await apply_proposal(user_id, session_id, proposal)
