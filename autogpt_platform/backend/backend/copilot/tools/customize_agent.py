"""CustomizeAgentTool - Customizes marketplace/template agents using natural language."""

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
    """Tool for customizing marketplace/template agents using natural language."""

    @property
    def name(self) -> str:
        return "customize_agent"

    @property
    def description(self) -> str:
        return (
            "Customize a marketplace or template agent using natural language. "
            "Takes an existing agent from the marketplace and modifies it based on "
            "the user's requirements before adding to their library."
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
                        "(e.g., 'autogpt/newsletter-writer'). "
                        "Get this from find_agent results."
                    ),
                },
                "modifications": {
                    "type": "string",
                    "description": (
                        "Natural language description of how to customize the agent. "
                        "Be specific about what changes you want to make."
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
                        "Whether to save the customized agent to the user's library. "
                        "Default is true. Set to false for preview only."
                    ),
                    "default": True,
                },
            },
            "required": ["agent_id", "modifications"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Execute the customize_agent tool.

        Flow:
        1. Parse the agent ID to get creator/slug
        2. Fetch the template agent from the marketplace
        3. Call customize_template with the modification request
        4. Preview or save based on the save parameter
        """
        agent_id = kwargs.get("agent_id", "").strip()
        modifications = kwargs.get("modifications", "").strip()
        context = kwargs.get("context", "")
        save = kwargs.get("save", True)
        session_id = session.session_id if session else None

        if not agent_id:
            return ErrorResponse(
                message="Please provide the marketplace agent ID (e.g., 'creator/agent-name').",
                error="missing_agent_id",
                session_id=session_id,
            )

        if not modifications:
            return ErrorResponse(
                message="Please describe how you want to customize this agent.",
                error="missing_modifications",
                session_id=session_id,
            )

        # Parse agent_id in format "creator/slug"
        parts = [p.strip() for p in agent_id.split("/")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return ErrorResponse(
                message=(
                    f"Invalid agent ID format: '{agent_id}'. "
                    "Expected format is 'creator/agent-name' "
                    "(e.g., 'autogpt/newsletter-writer')."
                ),
                error="invalid_agent_id_format",
                session_id=session_id,
            )

        creator_username, agent_slug = parts

        store_db = get_store_db()

        # Fetch the marketplace agent details
        try:
            agent_details = await store_db.get_store_agent_details(
                username=creator_username, agent_name=agent_slug
            )
        except AgentNotFoundError:
            return ErrorResponse(
                message=(
                    f"Could not find marketplace agent '{agent_id}'. "
                    "Please check the agent ID and try again."
                ),
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
                message=(
                    f"The agent '{agent_id}' does not have an available version. "
                    "Please try a different agent."
                ),
                error="no_version_available",
                session_id=session_id,
            )

        # Get the full agent graph
        try:
            graph = await store_db.get_agent(agent_details.store_listing_version_id)
            template_agent = graph_to_json(graph)
        except Exception as e:
            logger.error(f"Error fetching agent graph for {agent_id}: {e}")
            return ErrorResponse(
                message="Failed to fetch the agent configuration. Please try again.",
                error="graph_fetch_error",
                session_id=session_id,
            )

        # Call customize_template
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
                message=(
                    "Failed to customize the agent due to a service error. "
                    "Please try again."
                ),
                error="customization_service_error",
                session_id=session_id,
            )

        if result is None:
            return ErrorResponse(
                message=(
                    "Failed to customize the agent. "
                    "The agent generation service may be unavailable or timed out. "
                    "Please try again."
                ),
                error="customization_failed",
                session_id=session_id,
            )

        # Handle error response
        if isinstance(result, dict) and result.get("type") == "error":
            error_msg = result.get("error", "Unknown error")
            error_type = result.get("error_type", "unknown")
            user_message = get_user_message_for_error(
                error_type,
                operation="customize the agent",
                llm_parse_message=(
                    "The AI had trouble customizing the agent. "
                    "Please try again or simplify your request."
                ),
                validation_message=(
                    "The customized agent failed validation. "
                    "Please try rephrasing your request."
                ),
                error_details=error_msg,
            )
            return ErrorResponse(
                message=user_message,
                error=f"customization_failed:{error_type}",
                session_id=session_id,
            )

        # Handle clarifying questions
        if isinstance(result, dict) and result.get("type") == "clarifying_questions":
            questions = result.get("questions") or []
            if not isinstance(questions, list):
                logger.error(
                    f"Unexpected clarifying questions format: {type(questions)}"
                )
                questions = []
            return ClarificationNeededResponse(
                message=(
                    "I need some more information to customize this agent. "
                    "Please answer the following questions:"
                ),
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

        # Result should be the customized agent JSON
        if not isinstance(result, dict):
            logger.error(f"Unexpected customize_template response type: {type(result)}")
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
                    f"I've customized the agent '{agent_details.agent_name}'. "
                    f"The customized agent has {node_count} blocks. "
                    f"Review it and call customize_agent with save=true to save it."
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

        # Save to user's library
        try:
            created_graph, library_agent = await save_agent_to_library(
                customized_agent, user_id, is_update=False
            )

            return AgentSavedResponse(
                message=(
                    f"Customized agent '{created_graph.name}' "
                    f"(based on '{agent_details.agent_name}') "
                    f"has been saved to your library!"
                ),
                agent_id=created_graph.id,
                agent_name=created_graph.name,
                library_agent_id=library_agent.id,
                library_agent_link=f"/library/agents/{library_agent.id}",
                agent_page_link=f"/build?flowID={created_graph.id}",
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error saving customized agent: {e}")
            return ErrorResponse(
                message="Failed to save the customized agent. Please try again.",
                error="save_failed",
                session_id=session_id,
            )
