"""CreateAgentTool - Creates agents from natural language descriptions."""

import logging
from typing import Any

from backend.api.features.chat.model import ChatSession

from .agent_generator import (
    AgentGeneratorNotConfiguredError,
    decompose_goal,
    enrich_library_agents_from_steps,
    generate_agent,
    get_all_relevant_agents_for_generation,
    get_user_message_for_error,
    save_agent_to_library,
)
from .base import BaseTool
from .models import (
    AgentPreviewResponse,
    AgentSavedResponse,
    AsyncProcessingResponse,
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

        # Extract async processing params (passed by long-running tool handler)
        operation_id = kwargs.get("_operation_id")
        task_id = kwargs.get("_task_id")

        if not description:
            return ErrorResponse(
                message="Please provide a description of what the agent should do.",
                error="Missing description parameter",
                session_id=session_id,
            )

        library_agents = None
        if user_id:
            try:
                library_agents = await get_all_relevant_agents_for_generation(
                    user_id=user_id,
                    search_query=description,
                    include_marketplace=True,
                )
                logger.debug(
                    f"Found {len(library_agents)} relevant agents for sub-agent composition"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch library agents: {e}")

        try:
            decomposition_result = await decompose_goal(
                description, context, library_agents
            )
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
                message="Failed to analyze the goal. The agent generation service may be unavailable. Please try again.",
                error="decomposition_failed",
                details={"description": description[:100]},
                session_id=session_id,
            )

        if decomposition_result.get("type") == "error":
            error_msg = decomposition_result.get("error", "Unknown error")
            error_type = decomposition_result.get("error_type", "unknown")
            user_message = get_user_message_for_error(
                error_type,
                operation="analyze the goal",
                llm_parse_message="The AI had trouble understanding this request. Please try rephrasing your goal.",
            )
            return ErrorResponse(
                message=user_message,
                error=f"decomposition_failed:{error_type}",
                details={
                    "description": description[:100],
                    "service_error": error_msg,
                    "error_type": error_type,
                },
                session_id=session_id,
            )

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

        if user_id and library_agents is not None:
            try:
                library_agents = await enrich_library_agents_from_steps(
                    user_id=user_id,
                    decomposition_result=decomposition_result,
                    existing_agents=library_agents,
                    include_marketplace=True,
                )
                logger.debug(
                    f"After enrichment: {len(library_agents)} total agents for sub-agent composition"
                )
            except Exception as e:
                logger.warning(f"Failed to enrich library agents from steps: {e}")

        try:
            agent_json = await generate_agent(
                decomposition_result,
                library_agents,
                operation_id=operation_id,
                task_id=task_id,
            )
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
                message="Failed to generate the agent. The agent generation service may be unavailable. Please try again.",
                error="generation_failed",
                details={"description": description[:100]},
                session_id=session_id,
            )

        if isinstance(agent_json, dict) and agent_json.get("type") == "error":
            error_msg = agent_json.get("error", "Unknown error")
            error_type = agent_json.get("error_type", "unknown")
            user_message = get_user_message_for_error(
                error_type,
                operation="generate the agent",
                llm_parse_message="The AI had trouble generating the agent. Please try again or simplify your goal.",
                validation_message=(
                    "I wasn't able to create a valid agent for this request. "
                    "The generated workflow had some structural issues. "
                    "Please try simplifying your goal or breaking it into smaller steps."
                ),
                error_details=error_msg,
            )
            return ErrorResponse(
                message=user_message,
                error=f"generation_failed:{error_type}",
                details={
                    "description": description[:100],
                    "service_error": error_msg,
                    "error_type": error_type,
                },
                session_id=session_id,
            )

        # Check if Agent Generator accepted for async processing
        if agent_json.get("status") == "accepted":
            logger.info(
                f"Agent generation delegated to async processing "
                f"(operation_id={operation_id}, task_id={task_id})"
            )
            return AsyncProcessingResponse(
                message="Agent generation started. You'll be notified when it's complete.",
                operation_id=operation_id,
                task_id=task_id,
                session_id=session_id,
            )

        agent_name = agent_json.get("name", "Generated Agent")
        agent_description = agent_json.get("description", "")
        node_count = len(agent_json.get("nodes", []))
        link_count = len(agent_json.get("links", []))

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
                library_agent_link=f"/library/agents/{library_agent.id}",
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
