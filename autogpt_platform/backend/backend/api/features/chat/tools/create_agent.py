"""CreateAgentTool - Creates agents from natural language descriptions."""

import logging
from typing import Any

from langfuse import observe

from backend.api.features.chat.model import ChatSession

from .agent_generator import (
    apply_all_fixes,
    decompose_goal,
    generate_agent,
    get_blocks_info,
    save_agent_to_library,
    validate_agent,
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

# Maximum retries for agent generation with validation feedback
MAX_GENERATION_RETRIES = 2


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

    @observe(as_type="tool", name="create_agent")
    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Execute the create_agent tool.

        Flow:
        1. Decompose the description into steps (may return clarifying questions)
        2. Generate agent JSON from the steps
        3. Apply fixes to correct common LLM errors
        4. Preview or save based on the save parameter
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
        except ValueError as e:
            # Handle missing API key or configuration errors
            return ErrorResponse(
                message=f"Agent generation is not configured: {str(e)}",
                error="configuration_error",
                session_id=session_id,
            )

        if decomposition_result is None:
            return ErrorResponse(
                message="Failed to analyze the goal. Please try rephrasing.",
                error="Decomposition failed",
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

        # Step 2: Generate agent JSON with retry on validation failure
        blocks_info = get_blocks_info()
        agent_json = None
        validation_errors = None

        for attempt in range(MAX_GENERATION_RETRIES + 1):
            # Generate agent (include validation errors from previous attempt)
            if attempt == 0:
                agent_json = await generate_agent(decomposition_result)
            else:
                # Retry with validation error feedback
                logger.info(
                    f"Retry {attempt}/{MAX_GENERATION_RETRIES} with validation feedback"
                )
                retry_instructions = {
                    **decomposition_result,
                    "previous_errors": validation_errors,
                    "retry_instructions": (
                        "The previous generation had validation errors. "
                        "Please fix these issues in the new generation:\n"
                        f"{validation_errors}"
                    ),
                }
                agent_json = await generate_agent(retry_instructions)

            if agent_json is None:
                if attempt == MAX_GENERATION_RETRIES:
                    return ErrorResponse(
                        message="Failed to generate the agent. Please try again.",
                        error="Generation failed",
                        session_id=session_id,
                    )
                continue

            # Step 3: Apply fixes to correct common errors
            agent_json = apply_all_fixes(agent_json, blocks_info)

            # Step 4: Validate the agent
            is_valid, validation_errors = validate_agent(agent_json, blocks_info)

            if is_valid:
                logger.info(f"Agent generated successfully on attempt {attempt + 1}")
                break

            logger.warning(
                f"Validation failed on attempt {attempt + 1}: {validation_errors}"
            )

            if attempt == MAX_GENERATION_RETRIES:
                # Return error with validation details
                return ErrorResponse(
                    message=(
                        f"Generated agent has validation errors after {MAX_GENERATION_RETRIES + 1} attempts. "
                        f"Please try rephrasing your request or simplify the workflow."
                    ),
                    error="validation_failed",
                    details={"validation_errors": validation_errors},
                    session_id=session_id,
                )

        agent_name = agent_json.get("name", "Generated Agent")
        agent_description = agent_json.get("description", "")
        node_count = len(agent_json.get("nodes", []))
        link_count = len(agent_json.get("links", []))

        # Step 4: Preview or save
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
