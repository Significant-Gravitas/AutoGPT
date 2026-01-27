"""EditAgentTool - Edits existing agents using natural language."""

import logging
from typing import Any

from backend.api.features.chat.model import ChatSession

from .agent_generator import (
    AgentGeneratorNotConfiguredError,
    generate_agent_patch,
    get_agent_as_json,
    save_agent_to_library,
)
from .base import BaseTool
from .models import (
    AgentPreviewResponse,
    AgentSavedResponse,
    ClarificationNeededResponse,
    ClarifyingQuestion,
    ErrorResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class EditAgentTool(BaseTool):
    """Tool for editing existing agents using natural language."""

    @property
    def name(self) -> str:
        return "edit_agent"

    @property
    def description(self) -> str:
        return (
            "Edit an existing agent from the user's library using natural language. "
            "Generates updates to the agent while preserving unchanged parts."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_long_running(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": (
                        "The ID of the agent to edit. "
                        "Can be a graph ID or library agent ID."
                    ),
                },
                "changes": {
                    "type": "string",
                    "description": (
                        "Natural language description of what changes to make. "
                        "Be specific about what to add, remove, or modify."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Additional context or answers to previous clarifying questions."
                    ),
                },
                "save": {
                    "type": "boolean",
                    "description": (
                        "Whether to save the changes. "
                        "Default is true. Set to false for preview only."
                    ),
                    "default": True,
                },
            },
            "required": ["agent_id", "changes"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Execute the edit_agent tool.

        Flow:
        1. Fetch the current agent
        2. Generate updated agent (external service handles fixing and validation)
        3. Preview or save based on the save parameter
        """
        agent_id = kwargs.get("agent_id", "").strip()
        changes = kwargs.get("changes", "").strip()
        context = kwargs.get("context", "")
        save = kwargs.get("save", True)
        session_id = session.session_id if session else None

        if not agent_id:
            return ErrorResponse(
                message="Please provide the agent ID to edit.",
                error="Missing agent_id parameter",
                session_id=session_id,
            )

        if not changes:
            return ErrorResponse(
                message="Please describe what changes you want to make.",
                error="Missing changes parameter",
                session_id=session_id,
            )

        # Step 1: Fetch current agent
        current_agent = await get_agent_as_json(agent_id, user_id)

        if current_agent is None:
            return ErrorResponse(
                message=f"Could not find agent with ID '{agent_id}' in your library.",
                error="agent_not_found",
                session_id=session_id,
            )

        # Build the update request with context
        update_request = changes
        if context:
            update_request = f"{changes}\n\nAdditional context:\n{context}"

        # Step 2: Generate updated agent (external service handles fixing and validation)
        try:
            result = await generate_agent_patch(update_request, current_agent)
        except AgentGeneratorNotConfiguredError:
            return ErrorResponse(
                message=(
                    "Agent editing is not available. "
                    "The Agent Generator service is not configured."
                ),
                error="service_not_configured",
                session_id=session_id,
            )

        if result is None:
            return ErrorResponse(
                message="Failed to generate changes. The agent generation service may be unavailable or timed out. Please try again.",
                error="update_generation_failed",
                details={"agent_id": agent_id, "changes": changes[:100]},
                session_id=session_id,
            )

        # Check if LLM returned clarifying questions
        if result.get("type") == "clarifying_questions":
            questions = result.get("questions", [])
            return ClarificationNeededResponse(
                message=(
                    "I need some more information about the changes. "
                    "Please answer the following questions:"
                ),
                questions=[
                    ClarifyingQuestion(
                        question=q.get("question", ""),
                        keyword=q.get("keyword", ""),
                        example=q.get("example"),
                    )
                    for q in questions
                ],
                session_id=session_id,
            )

        # Result is the updated agent JSON
        updated_agent = result

        agent_name = updated_agent.get("name", "Updated Agent")
        agent_description = updated_agent.get("description", "")
        node_count = len(updated_agent.get("nodes", []))
        link_count = len(updated_agent.get("links", []))

        # Step 3: Preview or save
        if not save:
            return AgentPreviewResponse(
                message=(
                    f"I've updated the agent. "
                    f"The agent now has {node_count} blocks. "
                    f"Review it and call edit_agent with save=true to save the changes."
                ),
                agent_json=updated_agent,
                agent_name=agent_name,
                description=agent_description,
                node_count=node_count,
                link_count=link_count,
                session_id=session_id,
            )

        # Save to library (creates a new version)
        if not user_id:
            return ErrorResponse(
                message="You must be logged in to save agents.",
                error="auth_required",
                session_id=session_id,
            )

        try:
            created_graph, library_agent = await save_agent_to_library(
                updated_agent, user_id, is_update=True
            )

            return AgentSavedResponse(
                message=f"Updated agent '{created_graph.name}' has been saved to your library!",
                agent_id=created_graph.id,
                agent_name=created_graph.name,
                library_agent_id=library_agent.id,
                library_agent_link=f"/library/{library_agent.id}",
                agent_page_link=f"/build?flowID={created_graph.id}",
                session_id=session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to save the updated agent: {str(e)}",
                error="save_failed",
                details={"exception": str(e)},
                session_id=session_id,
            )
