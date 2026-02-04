"""CustomizeAgentTool - Customizes marketplace/template agents using natural language."""

import logging
from typing import Any

from pydantic import BaseModel, field_validator

from backend.api.features.chat.model import ChatSession
from backend.api.features.store import db as store_db
from backend.api.features.store.exceptions import AgentNotFoundError

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


class CustomizeAgentInput(BaseModel):
    """Input parameters for the customize_agent tool."""

    agent_id: str = ""
    modifications: str = ""
    context: str = ""
    save: bool = True

    @field_validator("agent_id", "modifications", "context", mode="before")
    @classmethod
    def strip_strings(cls, v: Any) -> str:
        """Strip whitespace from string fields."""
        if isinstance(v, str):
            return v.strip()
        return v if v is not None else ""


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
        **kwargs: Any,
    ) -> ToolResponseBase:
        """Execute the customize_agent tool.

        Flow:
        1. Parse the agent ID to get creator/slug
        2. Fetch the template agent from the marketplace
        3. Call customize_template with the modification request
        4. Preview or save based on the save parameter
        """
        params = CustomizeAgentInput(**kwargs)
        session_id = session.session_id if session else None

        if not params.agent_id:
            return ErrorResponse(
                message="Please provide the marketplace agent ID (e.g., 'creator/agent-name').",
                error="missing_agent_id",
                session_id=session_id,
            )

        if not params.modifications:
            return ErrorResponse(
                message="Please describe how you want to customize this agent.",
                error="missing_modifications",
                session_id=session_id,
            )

        # Parse agent_id in format "creator/slug"
        parts = params.agent_id.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return ErrorResponse(
                message=(
                    f"Invalid agent ID format: '{params.agent_id}'. "
                    "Expected format is 'creator/agent-name' "
                    "(e.g., 'autogpt/newsletter-writer')."
                ),
                error="invalid_agent_id_format",
                session_id=session_id,
            )

        creator_username, agent_slug = parts

        # Fetch the marketplace agent details
        try:
            agent_details = await store_db.get_store_agent_details(
                username=creator_username, agent_name=agent_slug
            )
        except AgentNotFoundError:
            return ErrorResponse(
                message=(
                    f"Could not find marketplace agent '{params.agent_id}'. "
                    "Please check the agent ID and try again."
                ),
                error="agent_not_found",
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error fetching marketplace agent {params.agent_id}: {e}")
            return ErrorResponse(
                message="Failed to fetch the marketplace agent. Please try again.",
                error="fetch_error",
                session_id=session_id,
            )

        if not agent_details.store_listing_version_id:
            return ErrorResponse(
                message=(
                    f"The agent '{params.agent_id}' does not have an available version. "
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
            logger.error(f"Error fetching agent graph for {params.agent_id}: {e}")
            return ErrorResponse(
                message="Failed to fetch the agent configuration. Please try again.",
                error="graph_fetch_error",
                session_id=session_id,
            )

        # Call customize_template
        try:
            result = await customize_template(
                template_agent=template_agent,
                modification_request=params.modifications,
                context=params.context,
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
            logger.error(f"Error calling customize_template for {params.agent_id}: {e}")
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

        # Handle response using match/case for cleaner pattern matching
        return await self._handle_customization_result(
            result=result,
            params=params,
            agent_details=agent_details,
            user_id=user_id,
            session_id=session_id,
        )

    async def _handle_customization_result(
        self,
        result: dict[str, Any],
        params: CustomizeAgentInput,
        agent_details: Any,
        user_id: str | None,
        session_id: str | None,
    ) -> ToolResponseBase:
        """Handle the result from customize_template using pattern matching."""
        # Ensure result is a dict
        if not isinstance(result, dict):
            logger.error(f"Unexpected customize_template response type: {type(result)}")
            return ErrorResponse(
                message="Failed to customize the agent due to an unexpected response.",
                error="unexpected_response_type",
                session_id=session_id,
            )

        result_type = result.get("type")

        match result_type:
            case "error":
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

            case "clarifying_questions":
                questions_data = result.get("questions") or []
                if not isinstance(questions_data, list):
                    logger.error(
                        f"Unexpected clarifying questions format: {type(questions_data)}"
                    )
                    questions_data = []

                questions = [
                    ClarifyingQuestion(
                        question=q.get("question", "") if isinstance(q, dict) else "",
                        keyword=q.get("keyword", "") if isinstance(q, dict) else "",
                        example=q.get("example") if isinstance(q, dict) else None,
                    )
                    for q in questions_data
                    if isinstance(q, dict)
                ]

                return ClarificationNeededResponse(
                    message=(
                        "I need some more information to customize this agent. "
                        "Please answer the following questions:"
                    ),
                    questions=questions,
                    session_id=session_id,
                )

            case _:
                # Default case: result is the customized agent JSON
                return await self._save_or_preview_agent(
                    customized_agent=result,
                    params=params,
                    agent_details=agent_details,
                    user_id=user_id,
                    session_id=session_id,
                )

    async def _save_or_preview_agent(
        self,
        customized_agent: dict[str, Any],
        params: CustomizeAgentInput,
        agent_details: Any,
        user_id: str | None,
        session_id: str | None,
    ) -> ToolResponseBase:
        """Save or preview the customized agent based on params.save."""
        agent_name = customized_agent.get(
            "name", f"Customized {agent_details.agent_name}"
        )
        agent_description = customized_agent.get("description", "")
        nodes = customized_agent.get("nodes")
        links = customized_agent.get("links")
        node_count = len(nodes) if isinstance(nodes, list) else 0
        link_count = len(links) if isinstance(links, list) else 0

        if not params.save:
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
