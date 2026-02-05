"""EditAgentTool - Edits existing agents using natural language."""

import logging
from typing import Any

from backend.api.features.chat.model import ChatSession

from .agent_generator import (
    AgentGeneratorNotConfiguredError,
    generate_agent_patch,
    get_agent_as_json,
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

        # Extract async processing params (passed by long-running tool handler)
        operation_id = kwargs.get("_operation_id")
        task_id = kwargs.get("_task_id")

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

        current_agent = await get_agent_as_json(agent_id, user_id)

        if current_agent is None:
            return ErrorResponse(
                message=f"Could not find agent with ID '{agent_id}' in your library.",
                error="agent_not_found",
                session_id=session_id,
            )

        library_agents = None
        if user_id:
            try:
                graph_id = current_agent.get("id")
                library_agents = await get_all_relevant_agents_for_generation(
                    user_id=user_id,
                    search_query=changes,
                    exclude_graph_id=graph_id,
                    include_marketplace=True,
                )
                logger.debug(
                    f"Found {len(library_agents)} relevant agents for sub-agent composition"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch library agents: {e}")

        update_request = changes
        if context:
            update_request = f"{changes}\n\nAdditional context:\n{context}"

        try:
            result = await generate_agent_patch(
                update_request,
                current_agent,
                library_agents,
                operation_id=operation_id,
                task_id=task_id,
            )
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

        # Check if Agent Generator accepted for async processing
        if result.get("status") == "accepted":
            logger.info(
                f"Agent edit delegated to async processing "
                f"(operation_id={operation_id}, task_id={task_id})"
            )
            return AsyncProcessingResponse(
                message="Agent edit started. You'll be notified when it's complete.",
                operation_id=operation_id,
                task_id=task_id,
                session_id=session_id,
            )

        # Check if the result is an error from the external service
        if isinstance(result, dict) and result.get("type") == "error":
            error_msg = result.get("error", "Unknown error")
            error_type = result.get("error_type", "unknown")
            user_message = get_user_message_for_error(
                error_type,
                operation="generate the changes",
                llm_parse_message="The AI had trouble generating the changes. Please try again or simplify your request.",
                validation_message="The generated changes failed validation. Please try rephrasing your request.",
                error_details=error_msg,
            )
            return ErrorResponse(
                message=user_message,
                error=f"update_generation_failed:{error_type}",
                details={
                    "agent_id": agent_id,
                    "changes": changes[:100],
                    "service_error": error_msg,
                    "error_type": error_type,
                },
                session_id=session_id,
            )

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

        updated_agent = result

        agent_name = updated_agent.get("name", "Updated Agent")
        agent_description = updated_agent.get("description", "")
        node_count = len(updated_agent.get("nodes", []))
        link_count = len(updated_agent.get("links", []))

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
                library_agent_link=f"/library/agents/{library_agent.id}",
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
