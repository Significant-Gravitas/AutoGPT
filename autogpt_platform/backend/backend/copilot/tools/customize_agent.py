"""CustomizeAgentTool - Customizes marketplace/template agents."""

import logging
from typing import Any

from backend.api.features.store.exceptions import AgentNotFoundError
from backend.copilot.model import ChatSession
from backend.data.db_accessors import store_db as get_store_db

from .agent_generator import (
    AgentGeneratorNotConfiguredError,
    customize_template,
    get_user_message_for_error,
    graph_to_json,
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
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class CustomizeAgentTool(BaseTool):
    """Tool for customizing marketplace/template agents."""

    @property
    def name(self) -> str:
        return "customize_agent"

    @property
    def description(self) -> str:
        return (
            "Customize a marketplace or template agent. Supports two modes:\n\n"
            "1. **Local mode** (preferred): Pass `agent_json` with the complete "
            "customized agent JSON. The tool validates, auto-fixes, and saves.\n\n"
            "2. **External mode** (fallback): Pass `modifications` as natural language. "
            "Delegates to the external Agent Generator service.\n\n"
            "Use agent_id in format 'creator/slug' to specify the marketplace agent."
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
                        "The marketplace agent ID in format 'creator/slug' "
                        "(e.g., 'autogpt/newsletter-writer')."
                    ),
                },
                "agent_json": {
                    "type": "object",
                    "description": (
                        "Complete customized agent JSON to validate and save. "
                        "When provided, skips external service."
                    ),
                },
                "modifications": {
                    "type": "string",
                    "description": (
                        "Natural language description of how to customize the agent. "
                        "Used when agent_json is not provided (external mode)."
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
                        "Whether to save the customized agent. Default is true."
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
                message="Please provide the marketplace agent ID (e.g., 'creator/agent-name').",
                error="missing_agent_id",
                session_id=session_id,
            )

        if isinstance(agent_json, dict):
            return await self._execute_local(user_id, session_id, agent_json, kwargs)
        return await self._execute_external(user_id, session_id, agent_id, kwargs)

    async def _execute_local(
        self,
        user_id: str | None,
        session_id: str | None,
        agent_json: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> ToolResponseBase:
        """Local mode: validate, fix, and save pre-built customized agent JSON."""
        save = kwargs.get("save", True)
        library_agent_ids = kwargs.get("library_agent_ids", [])

        nodes = agent_json.get("nodes", [])

        if not nodes:
            return ErrorResponse(
                message="The agent JSON has no nodes.",
                error="empty_agent",
                session_id=session_id,
            )

        agent_json.setdefault("is_active", True)

        # Fetch library agents for AgentExecutorBlock validation
        library_agents = await fetch_library_agents(user_id, library_agent_ids)

        return await fix_validate_and_save(
            agent_json,
            user_id=user_id,
            session_id=session_id,
            save=save,
            is_update=False,
            default_name="Customized Agent",
            library_agents=library_agents,
        )

    async def _execute_external(
        self,
        user_id: str | None,
        session_id: str | None,
        agent_id: str,
        kwargs: dict[str, Any],
    ) -> ToolResponseBase:
        """External mode: customize via external Agent Generator service."""
        modifications = kwargs.get("modifications", "").strip()
        context = kwargs.get("context", "")
        save = kwargs.get("save", True)

        if not modifications:
            return ErrorResponse(
                message="Please describe how you want to customize this agent, or pass agent_json directly.",
                error="missing_modifications",
                session_id=session_id,
            )

        # Parse agent_id in format "creator/slug"
        parts = [p.strip() for p in agent_id.split("/")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return ErrorResponse(
                message=(
                    f"Invalid agent ID format: '{agent_id}'. "
                    "Expected format is 'creator/agent-name'."
                ),
                error="invalid_agent_id_format",
                session_id=session_id,
            )

        creator_username, agent_slug = parts
        store_db = get_store_db()

        try:
            agent_details = await store_db.get_store_agent_details(
                username=creator_username, agent_name=agent_slug
            )
        except AgentNotFoundError:
            return ErrorResponse(
                message=f"Could not find marketplace agent '{agent_id}'.",
                error="agent_not_found",
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error fetching marketplace agent {agent_id}: {e}")
            return ErrorResponse(
                message="Failed to fetch the marketplace agent. Please try again.",
                error="fetch_error",
                session_id=session_id,
            )

        if not agent_details.store_listing_version_id:
            return ErrorResponse(
                message=f"The agent '{agent_id}' does not have an available version.",
                error="no_version_available",
                session_id=session_id,
            )

        try:
            graph = await store_db.get_agent(agent_details.store_listing_version_id)
            template_agent = graph_to_json(graph)
        except Exception as e:
            logger.error(f"Error fetching agent graph for {agent_id}: {e}")
            return ErrorResponse(
                message="Failed to fetch the agent configuration.",
                error="graph_fetch_error",
                session_id=session_id,
            )

        try:
            result = await customize_template(
                template_agent=template_agent,
                modification_request=modifications,
                context=context,
            )
        except AgentGeneratorNotConfiguredError:
            return ErrorResponse(
                message=(
                    "Agent customization is not available. "
                    "The Agent Generator service is not configured."
                ),
                error="service_not_configured",
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error calling customize_template for {agent_id}: {e}")
            return ErrorResponse(
                message="Failed to customize the agent. Please try again.",
                error="customization_service_error",
                session_id=session_id,
            )

        if result is None:
            return ErrorResponse(
                message="Failed to customize the agent. Please try again.",
                error="customization_failed",
                session_id=session_id,
            )

        if isinstance(result, dict) and result.get("type") == "error":
            error_msg = result.get("error", "Unknown error")
            error_type = result.get("error_type", "unknown")
            user_message = get_user_message_for_error(
                error_type,
                operation="customize the agent",
                llm_parse_message="The AI had trouble customizing the agent. Please try again.",
                validation_message="The customized agent failed validation.",
                error_details=error_msg,
            )
            return ErrorResponse(
                message=user_message,
                error=f"customization_failed:{error_type}",
                session_id=session_id,
            )

        if isinstance(result, dict) and result.get("type") == "clarifying_questions":
            questions = result.get("questions") or []
            if not isinstance(questions, list):
                questions = []
            return ClarificationNeededResponse(
                message="I need some more information to customize this agent.",
                questions=[
                    ClarifyingQuestion(
                        question=q.get("question", ""),
                        keyword=q.get("keyword", ""),
                        example=q.get("example"),
                    )
                    for q in questions
                    if isinstance(q, dict)
                ],
                session_id=session_id,
            )

        if not isinstance(result, dict):
            return ErrorResponse(
                message="Failed to customize the agent due to an unexpected response.",
                error="unexpected_response_type",
                session_id=session_id,
            )

        customized_agent = result
        agent_name = customized_agent.get(
            "name", f"Customized {agent_details.agent_name}"
        )
        agent_description = customized_agent.get("description", "")
        nodes = customized_agent.get("nodes")
        links = customized_agent.get("links")
        node_count = len(nodes) if isinstance(nodes, list) else 0
        link_count = len(links) if isinstance(links, list) else 0

        if not save:
            return AgentPreviewResponse(
                message=(
                    f"I've customized '{agent_details.agent_name}'. "
                    f"It has {node_count} blocks. "
                    f"Call customize_agent with save=true to save."
                ),
                agent_json=customized_agent,
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
                customized_agent, user_id, is_update=False
            )
            return AgentSavedResponse(
                message=(
                    f"Customized agent '{created_graph.name}' "
                    f"(based on '{agent_details.agent_name}') has been saved!"
                ),
                agent_id=created_graph.id,
                agent_name=created_graph.name,
                library_agent_id=library_agent.id,
                library_agent_link=f"/library/agents/{library_agent.id}",
                agent_page_link=f"/build?flowID={created_graph.id}",
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Failed to save customized agent: {e}")
            return ErrorResponse(
                message=f"Failed to save the customized agent: {str(e)}",
                error="save_failed",
                details={"exception": str(e)},
                session_id=session_id,
            )
