"""EditAgentTool - Edits existing agents using natural language or pre-built JSON."""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator import (
    AgentGeneratorNotConfiguredError,
    generate_agent_patch,
    get_agent_as_json,
    get_user_message_for_error,
    save_agent_to_library,
)
from .agent_generator.pipeline import fix_validate_and_save
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
    """Tool for editing existing agents using natural language or pre-built JSON."""

    @property
    def name(self) -> str:
        return "edit_agent"

    @property
    def description(self) -> str:
        return (
            "Edit an existing agent. Supports two modes:\n\n"
            "1. **Local mode** (preferred): Pass `agent_json` with the complete "
            "updated agent JSON you generated. The tool validates, auto-fixes, "
            "and saves.\n\n"
            "2. **External mode** (fallback): Pass `changes` as natural language. "
            "Delegates to the external Agent Generator service.\n\n"
            "IMPORTANT: Before calling this tool, if the changes involve adding new "
            "functionality, search for relevant existing agents using find_library_agent "
            "that could be used as building blocks."
        )

    @property
    def requires_auth(self) -> bool:
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
                "agent_json": {
                    "type": "object",
                    "description": (
                        "Complete updated agent JSON to validate and save. "
                        "When provided, skips external service and uses local "
                        "validate+fix+save flow. Must contain 'nodes' and 'links'."
                    ),
                },
                "changes": {
                    "type": "string",
                    "description": (
                        "Natural language description of what changes to make. "
                        "Used when agent_json is not provided (external mode)."
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
                        "List of library agent IDs to use as building blocks for the changes."
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
            "required": ["agent_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        agent_id = kwargs.get("agent_id", "").strip()
        agent_json = kwargs.get("agent_json")
        session_id = session.session_id if session else None

        if not agent_id:
            return ErrorResponse(
                message="Please provide the agent ID to edit.",
                error="Missing agent_id parameter",
                session_id=session_id,
            )

        if isinstance(agent_json, dict):
            return await self._execute_local(
                user_id, session_id, agent_id, agent_json, kwargs
            )
        return await self._execute_external(user_id, session_id, agent_id, kwargs)

    async def _execute_local(
        self,
        user_id: str | None,
        session_id: str | None,
        agent_id: str,
        agent_json: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> ToolResponseBase:
        """Local mode: validate, fix, and save pre-built updated agent JSON."""
        save = kwargs.get("save", True)

        nodes = agent_json.get("nodes", [])

        if not nodes:
            return ErrorResponse(
                message="The agent JSON has no nodes.",
                error="empty_agent",
                session_id=session_id,
            )

        # Preserve original agent's ID
        current_agent = await get_agent_as_json(agent_id, user_id)
        if current_agent is None:
            return ErrorResponse(
                message=f"Could not find agent with ID '{agent_id}' in your library.",
                error="agent_not_found",
                session_id=session_id,
            )

        agent_json.setdefault("id", current_agent.get("id", agent_id))
        agent_json.setdefault("version", current_agent.get("version", 1))
        agent_json.setdefault("is_active", True)

        return await fix_validate_and_save(
            agent_json,
            user_id=user_id,
            session_id=session_id,
            save=save,
            is_update=True,
            default_name="Updated Agent",
        )

    async def _execute_external(
        self,
        user_id: str | None,
        session_id: str | None,
        agent_id: str,
        kwargs: dict[str, Any],
    ) -> ToolResponseBase:
        """External mode: generate patch via external Agent Generator service."""
        changes = kwargs.get("changes", "").strip()
        context = kwargs.get("context", "")
        library_agent_ids = kwargs.get("library_agent_ids", [])
        save = kwargs.get("save", True)

        if not changes:
            return ErrorResponse(
                message="Please describe what changes you want to make, or pass agent_json directly.",
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

        # Fetch library agents by IDs if provided
        library_agents = None
        if user_id and library_agent_ids:
            try:
                from .agent_generator import get_library_agents_by_ids

                graph_id = current_agent.get("id")
                filtered_ids = [id for id in library_agent_ids if id != graph_id]
                library_agents = await get_library_agents_by_ids(
                    user_id=user_id,
                    agent_ids=filtered_ids,
                )
            except Exception as e:
                logger.warning(f"Failed to fetch library agents by IDs: {e}")

        update_request = changes
        if context:
            update_request = f"{changes}\n\nAdditional context:\n{context}"

        try:
            result = await generate_agent_patch(
                update_request,
                current_agent,
                library_agents,
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
                message="Failed to generate changes. Please try again.",
                error="update_generation_failed",
                details={"agent_id": agent_id, "changes": changes[:100]},
                session_id=session_id,
            )

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
                    f"Review it and call edit_agent with save=true to save."
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
                message=f"Updated agent '{created_graph.name}' has been saved!",
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
