"""CreateAgentTool - Creates agents from natural language descriptions."""

import logging
from typing import Any

from backend.api.features.chat.model import ChatSession

from .agent_generator import (
    AgentGeneratorNotConfiguredError,
    decompose_goal,
    generate_agent,
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


class CreateAgentTool(BaseTool):
    """Tool for creating agents from natural language descriptions."""

    @property
    def name(self) -> str:
        return "create_agent"

    @property
    def description(self) -> str:
        return (
            "Create a new agent workflow from a natural language description. "
            "First generates a preview, then saves to library if save=true."
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
                "description": {
                    "type": "string",
                    "description": (
                        "Natural language description of what the agent should do. "
                        "Be specific about inputs, outputs, and the workflow steps."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Additional context or answers to previous clarifying questions. "
                        "Include any preferences or constraints mentioned by the user."
                    ),
                },
                "save": {
                    "type": "boolean",
                    "description": (
                        "Whether to save the agent to the user's library. "
                        "Default is true. Set to false for preview only."
                    ),
                    "default": True,
                },
            },
            "required": ["description"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Execute the create_agent tool.

        Flow:
        1. Decompose the description into steps (may return clarifying questions)
        2. Generate agent JSON (external service handles fixing and validation)
        3. Preview or save based on the save parameter
        """
        description = kwargs.get("description", "").strip()
        context = kwargs.get("context", "")
        save = kwargs.get("save", True)
        session_id = session.session_id if session else None

        if not description:
            return ErrorResponse(
                message="Please provide a description of what the agent should do.",
                error="Missing description parameter",
                session_id=session_id,
            )

        # Step 1: Decompose goal into steps
        try:
            decomposition_result = await decompose_goal(description, context)
        except AgentGeneratorNotConfiguredError:
            return ErrorResponse(
                message=(
                    "Agent generation is not available. "
                    "The Agent Generator service is not configured."
                ),
                error="service_not_configured",
                session_id=session_id,
            )

        if decomposition_result is None:
            return ErrorResponse(
                message="Failed to analyze the goal. The agent generation service may be unavailable or timed out. Please try again.",
                error="decomposition_failed",
                details={
                    "description": description[:100]
                },  # Include context for debugging
                session_id=session_id,
            )

        # Check if LLM returned clarifying questions
        if decomposition_result.get("type") == "clarifying_questions":
            questions = decomposition_result.get("questions", [])
            return ClarificationNeededResponse(
                message=(
                    "I need some more information to create this agent. "
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

        # Check for unachievable/vague goals
        if decomposition_result.get("type") == "unachievable_goal":
            suggested = decomposition_result.get("suggested_goal", "")
            reason = decomposition_result.get("reason", "")
            return ErrorResponse(
                message=(
                    f"This goal cannot be accomplished with the available blocks. "
                    f"{reason} "
                    f"Suggestion: {suggested}"
                ),
                error="unachievable_goal",
                details={"suggested_goal": suggested, "reason": reason},
                session_id=session_id,
            )

        if decomposition_result.get("type") == "vague_goal":
            suggested = decomposition_result.get("suggested_goal", "")
            return ErrorResponse(
                message=(
                    f"The goal is too vague to create a specific workflow. "
                    f"Suggestion: {suggested}"
                ),
                error="vague_goal",
                details={"suggested_goal": suggested},
                session_id=session_id,
            )

        # Step 2: Generate agent JSON (external service handles fixing and validation)
        try:
            agent_json = await generate_agent(decomposition_result)
        except AgentGeneratorNotConfiguredError:
            return ErrorResponse(
                message=(
                    "Agent generation is not available. "
                    "The Agent Generator service is not configured."
                ),
                error="service_not_configured",
                session_id=session_id,
            )

        if agent_json is None:
            return ErrorResponse(
                message="Failed to generate the agent. The agent generation service may be unavailable or timed out. Please try again.",
                error="generation_failed",
                details={
                    "description": description[:100]
                },  # Include context for debugging
                session_id=session_id,
            )

        agent_name = agent_json.get("name", "Generated Agent")
        agent_description = agent_json.get("description", "")
        node_count = len(agent_json.get("nodes", []))
        link_count = len(agent_json.get("links", []))

        # Step 3: Preview or save
        if not save:
            return AgentPreviewResponse(
                message=(
                    f"I've generated an agent called '{agent_name}' with {node_count} blocks. "
                    f"Review it and call create_agent with save=true to save it to your library."
                ),
                agent_json=agent_json,
                agent_name=agent_name,
                description=agent_description,
                node_count=node_count,
                link_count=link_count,
                session_id=session_id,
            )

        # Save to library
        if not user_id:
            return ErrorResponse(
                message="You must be logged in to save agents.",
                error="auth_required",
                session_id=session_id,
            )

        try:
            created_graph, library_agent = await save_agent_to_library(
                agent_json, user_id
            )

            return AgentSavedResponse(
                message=f"Agent '{created_graph.name}' has been saved to your library!",
                agent_id=created_graph.id,
                agent_name=created_graph.name,
                library_agent_id=library_agent.id,
                library_agent_link=f"/library/{library_agent.id}",
                agent_page_link=f"/build?flowID={created_graph.id}",
                session_id=session_id,
            )
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to save the agent: {str(e)}",
                error="save_failed",
                details={"exception": str(e)},
                session_id=session_id,
            )
