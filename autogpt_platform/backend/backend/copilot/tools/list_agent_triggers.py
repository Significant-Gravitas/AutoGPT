"""Tool for listing all triggers (trigger agents + webhook presets) for an agent."""

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel

from backend.api.features.library.db import (
    get_library_agent,
    list_presets,
    list_trigger_agents,
)
from backend.copilot.model import ChatSession
from backend.util.exceptions import NotFoundError

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)


class AgentTriggerInfo(BaseModel):
    """Single trigger configured for a parent library agent.

    Two kinds: ``agent`` for a hidden trigger agent that runs on a
    schedule and invokes the parent (Fetch→Compare→Store→Sink pattern),
    and ``webhook`` for a webhook-triggered preset that fires the
    parent on incoming HTTP events. ``id`` identifies the trigger for
    follow-up actions: library_agent_id for ``agent`` (use to delete or
    edit the trigger agent), preset_id for ``webhook``.
    """

    kind: Literal["agent", "webhook"]
    id: str
    name: str
    description: str = ""

    # Agent-trigger only.
    graph_id: str | None = None
    is_scheduled: bool | None = None
    next_scheduled_run: str | None = None

    # Webhook-trigger only.
    is_active: bool | None = None
    webhook_id: str | None = None


class AgentTriggerListResponse(ToolResponseBase):
    """Response listing all triggers (agent + webhook) for a parent agent."""

    type: ResponseType = ResponseType.AGENT_TRIGGER_LIST
    triggers: list[AgentTriggerInfo]


class ListAgentTriggersTool(BaseTool):
    """List all triggers configured for a parent library agent.

    Returns both kinds of triggers: hidden trigger agents (scheduled
    fetch→compare→store→sink graphs that invoke the parent) and webhook
    presets (HTTP events that fire the parent directly). Use this
    before deleting a trigger or to show the user what's already set up.
    """

    @property
    def name(self) -> str:
        return "list_agent_triggers"

    @property
    def description(self) -> str:
        return (
            "List triggers (agents + webhook presets) for a library agent. "
            "Use before delete or to show what's set up."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "library_agent_id": {
                    "type": "string",
                    "description": "Parent library agent ID.",
                },
            },
            "required": ["library_agent_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )

        library_agent_id: str | None = kwargs.get("library_agent_id")
        if not library_agent_id:
            return ErrorResponse(
                message="library_agent_id is required.",
                error="missing_argument",
                session_id=session_id,
            )

        try:
            parent = await get_library_agent(id=library_agent_id, user_id=user_id)
        except NotFoundError as e:
            return ErrorResponse(
                message=f"Library agent not found: {e}",
                error="library_agent_not_found",
                session_id=session_id,
            )

        # Pass parent_graph_id so list_trigger_agents skips its own
        # get_library_agent call (we already loaded the parent above).
        # Both queries can run concurrently since they don't depend
        # on each other once we have the parent's graph_id.
        trigger_agents, preset_response = await asyncio.gather(
            list_trigger_agents(
                user_id=user_id,
                library_agent_id=library_agent_id,
                parent_graph_id=parent.graph_id,
            ),
            list_presets(
                user_id=user_id,
                page=1,
                page_size=100,
                graph_id=parent.graph_id,
            ),
        )
        webhook_presets = [
            p for p in preset_response.presets if p.webhook_id is not None
        ]

        triggers: list[AgentTriggerInfo] = [
            AgentTriggerInfo(
                kind="agent",
                id=ta.id,
                name=ta.name,
                description=ta.description or "",
                graph_id=ta.graph_id,
                is_scheduled=ta.is_scheduled,
                next_scheduled_run=ta.next_scheduled_run,
            )
            for ta in trigger_agents
        ] + [
            AgentTriggerInfo(
                kind="webhook",
                id=p.id,
                name=p.name,
                description=p.description or "",
                is_active=p.is_active,
                webhook_id=p.webhook_id,
            )
            for p in webhook_presets
        ]

        message = (
            f"Found {len(triggers)} trigger(s) for this agent."
            if triggers
            else "No triggers configured."
        )
        return AgentTriggerListResponse(
            message=message,
            triggers=triggers,
            session_id=session_id,
        )
