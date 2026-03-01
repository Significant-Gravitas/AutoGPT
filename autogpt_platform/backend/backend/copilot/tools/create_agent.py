"""CreateAgentTool - Creates agents from natural language or pre-built JSON."""

import logging
import uuid
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator import (
    AgentGeneratorNotConfiguredError,
    decompose_goal,
    enrich_library_agents_from_steps,
    generate_agent,
    get_user_message_for_error,
    save_agent_to_library,
)
from .agent_generator.pipeline import fetch_library_agents, fix_validate_and_save
from .base import BaseTool
from .models import (
    AgentPreviewResponse,
    AgentSavedResponse,
    ClarificationNeededResponse,
    ClarifyingQuestion,
    ErrorResponse,
    SuggestedGoalResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class CreateAgentTool(BaseTool):
    """Tool for creating agents from natural language or pre-built JSON."""

    @property
    def name(self) -> str:
        return "create_agent"

    @property
    def description(self) -> str:
        return (
            "Create a new agent workflow. Supports two modes:\n\n"
            "1. **Local mode** (preferred): Pass `agent_json` directly — "
            "the JSON you generated using block schemas from get_blocks_for_goal. "
            "The tool validates, auto-fixes, and saves.\n\n"
            "2. **External mode** (fallback): Pass `description` only — "
            "delegates to the external Agent Generator service for decomposition "
            "and generation.\n\n"
            "IMPORTANT: Before calling this tool, search for relevant existing agents "
            "using find_library_agent that could be used as building blocks. "
            "Pass their IDs in the library_agent_ids parameter."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_json": {
                    "type": "object",
                    "description": (
                        "Pre-built agent JSON to validate and save. "
                        "Must contain 'nodes' and 'links' arrays, and optionally "
                        "'name' and 'description'. When provided, skips external "
                        "service and uses local validate+fix+save flow."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Natural language description of what the agent should do. "
                        "Used when agent_json is not provided (external service mode)."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Additional context or answers to previous clarifying questions."
                    ),
                },
                "library_agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of library agent IDs to use as building blocks."
                    ),
                },
                "save": {
                    "type": "boolean",
                    "description": (
                        "Whether to save the agent. Default is true. "
                        "Set to false for preview only."
                    ),
                    "default": True,
                },
            },
            "required": [],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        agent_json = kwargs.get("agent_json")
        session_id = session.session_id if session else None

        if isinstance(agent_json, dict):
            return await self._execute_local(user_id, session_id, agent_json, kwargs)
        return await self._execute_external(user_id, session_id, kwargs)

    async def _execute_local(
        self,
        user_id: str | None,
        session_id: str | None,
        agent_json: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> ToolResponseBase:
        """Local mode: validate, fix, and save pre-built agent JSON."""
        save = kwargs.get("save", True)
        library_agent_ids = kwargs.get("library_agent_ids", [])

        nodes = agent_json.get("nodes", [])

        if not nodes:
            return ErrorResponse(
                message="The agent JSON has no nodes. An agent needs at least one block.",
                error="empty_agent",
                session_id=session_id,
            )

        # Ensure top-level fields
        if "id" not in agent_json:
            agent_json["id"] = str(uuid.uuid4())
        if "version" not in agent_json:
            agent_json["version"] = 1
        if "is_active" not in agent_json:
            agent_json["is_active"] = True

        # Fetch library agents for AgentExecutorBlock validation
        library_agents = await fetch_library_agents(user_id, library_agent_ids)

        return await fix_validate_and_save(
            agent_json,
            user_id=user_id,
            session_id=session_id,
            save=save,
            is_update=False,
            default_name="Generated Agent",
            library_agents=library_agents,
        )

    async def _execute_external(
        self,
        user_id: str | None,
        session_id: str | None,
        kwargs: dict[str, Any],
    ) -> ToolResponseBase:
        """External mode: decompose + generate via external service."""
        description = kwargs.get("description", "").strip()
        context = kwargs.get("context", "")
        library_agent_ids = kwargs.get("library_agent_ids", [])
        save = kwargs.get("save", True)

        if not description:
            return ErrorResponse(
                message="Please provide a description of what the agent should do, or pass agent_json directly.",
                error="Missing description parameter",
                session_id=session_id,
            )

        # Fetch library agents by IDs if provided
        library_agents = None
        if user_id and library_agent_ids:
            try:
                from .agent_generator import get_library_agents_by_ids

                library_agents = await get_library_agents_by_ids(
                    user_id=user_id,
                    agent_ids=library_agent_ids,
                )
            except Exception as e:
                logger.warning(f"Failed to fetch library agents by IDs: {e}")

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
            return SuggestedGoalResponse(
                message=(
                    f"This goal cannot be accomplished with the available blocks. {reason}"
                ),
                suggested_goal=suggested,
                reason=reason,
                original_goal=description,
                goal_type="unachievable",
                session_id=session_id,
            )

        if decomposition_result.get("type") == "vague_goal":
            suggested = decomposition_result.get("suggested_goal", "")
            reason = decomposition_result.get(
                "reason", "The goal needs more specific details"
            )
            return SuggestedGoalResponse(
                message="The goal is too vague to create a specific workflow.",
                suggested_goal=suggested,
                reason=reason,
                original_goal=description,
                goal_type="vague",
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
            except Exception as e:
                logger.warning(f"Failed to enrich library agents from steps: {e}")

        try:
            agent_json = await generate_agent(
                decomposition_result,
                library_agents,
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
