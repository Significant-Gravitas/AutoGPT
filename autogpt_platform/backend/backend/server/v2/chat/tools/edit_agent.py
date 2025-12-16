"""EditAgentTool - Edits existing agents using natural language."""

from typing import Any

from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.tools.agent_generator import (
    apply_agent_patch,
    apply_all_fixes,
    generate_agent_patch,
    get_agent_as_json,
    save_agent_to_library,
)
from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    AgentPreviewResponse,
    AgentSavedResponse,
    ClarificationNeededResponse,
    ClarifyingQuestion,
    ErrorResponse,
    ToolResponseBase,
)


class EditAgentTool(BaseTool):
    """Tool for editing existing agents using natural language."""

    @property
    def name(self) -> str:
        return "edit_agent"

    @property
    def description(self) -> str:
        return (
            "Edit an existing agent from the user's library using natural language. "
            "Generates a patch to update the agent while preserving unchanged parts."
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
        2. Generate a patch based on the requested changes
        3. Apply the patch to create an updated agent
        4. Preview or save based on the save parameter
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

        # Step 2: Generate patch
        try:
            patch_result = await generate_agent_patch(update_request, current_agent)
        except ValueError as e:
            # Handle missing API key or configuration errors
            return ErrorResponse(
                message=f"Agent generation is not configured: {str(e)}",
                error="configuration_error",
                session_id=session_id,
            )

        if patch_result is None:
            return ErrorResponse(
                message="Failed to generate changes. Please try rephrasing.",
                error="Patch generation failed",
                session_id=session_id,
            )

        # Check if LLM returned clarifying questions
        if patch_result.get("type") == "clarifying_questions":
            questions = patch_result.get("questions", [])
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

        # Step 3: Apply patch
        try:
            updated_agent = apply_agent_patch(current_agent, patch_result)
            updated_agent = apply_all_fixes(updated_agent)
        except Exception as e:
            return ErrorResponse(
                message=f"Failed to apply changes: {str(e)}",
                error="patch_apply_failed",
                details={"exception": str(e)},
                session_id=session_id,
            )

        agent_name = updated_agent.get("name", "Updated Agent")
        agent_description = updated_agent.get("description", "")
        node_count = len(updated_agent.get("nodes", []))
        link_count = len(updated_agent.get("links", []))
        intent = patch_result.get("intent", "Applied requested changes")

        # Step 4: Preview or save
        if not save:
            return AgentPreviewResponse(
                message=(
                    f"I've updated the agent. Changes: {intent}. "
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
                updated_agent, user_id
            )

            return AgentSavedResponse(
                message=(
                    f"Updated agent '{created_graph.name}' has been saved to your library! "
                    f"Changes: {intent}"
                ),
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
