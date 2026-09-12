"""Read the user's current expert roster (never writes).

The ``<team_context>`` roster block is injected only into a session's first
user message, so any team change after that — a hire confirmed three turns
ago, an expert archived from the Team page — is invisible to the model. The
observed failure mode is an expensive guess: the model re-proposes an expert
who already exists, or "checks" existence by delegating a ping. This tool
gives it a cheap, authoritative read instead.
"""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db

from .base import BaseTool
from .models import ErrorResponse, TeamExpertInfo, TeamRosterResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class ListTeamTool(BaseTool):
    """List the experts currently on the user's team."""

    @property
    def name(self) -> str:
        return "list_team"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "List the experts currently on the user's team (name, expert_id, "
            "role, paused state). The roster in <team_context> is a snapshot "
            "from the start of this chat — call this before hiring, raising, "
            "or delegating whenever the team may have changed since, or when "
            "an expert lookup failed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        if user_id is None:
            return ErrorResponse(
                message="Authentication required", session_id=session.session_id
            )
        try:
            experts = await experts_db().list_experts(user_id, with_metrics=False)
        except Exception as e:
            logger.warning(f"list_team roster lookup failed: {e}")
            return ErrorResponse(
                message="Could not load the team right now. Try again.",
                session_id=session.session_id,
            )
        active = [e for e in experts if not e.is_archived]
        if not active:
            return TeamRosterResponse(
                message=(
                    "The team is empty — no experts exist yet. hire_expert or "
                    "raise_expert (with user approval) is how one joins."
                ),
                session_id=session.session_id,
            )
        lines = "; ".join(
            f"{e.name} — {e.role} (expert_id: {e.id})"
            + (" [paused]" if e.schedules_paused_at is not None else "")
            for e in active
        )
        return TeamRosterResponse(
            message=(
                f"{len(active)} expert{'s' if len(active) != 1 else ''} on the "
                f"team: {lines}. Use these expert_ids with delegate_to_expert; "
                "never re-raise an expert who is already listed here."
            ),
            session_id=session.session_id,
            experts=[
                TeamExpertInfo(
                    id=e.id,
                    name=e.name,
                    role=e.role,
                    color=e.color,
                    avatar_url=e.avatar_url,
                    is_paused=e.schedules_paused_at is not None,
                )
                for e in active
            ],
        )
