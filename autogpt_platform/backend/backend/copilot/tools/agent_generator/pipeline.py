"""Shared fix → validate → preview/save pipeline for agent tools."""

import logging
from typing import Any, cast

from backend.copilot.tools.models import (
    AgentPreviewResponse,
    AgentSavedResponse,
    ErrorResponse,
    ToolResponseBase,
)

from .blocks import get_blocks_as_dicts
from .fixer import AgentFixer
from .validator import AgentValidator

logger = logging.getLogger(__name__)


async def fetch_library_agents(
    user_id: str | None,
    library_agent_ids: list[str],
) -> list[dict[str, Any]] | None:
    """Fetch library agents by IDs for AgentExecutorBlock validation.

    Returns None if no IDs provided or user is not authenticated.
    """
    if not user_id or not library_agent_ids:
        return None
    try:
        from .core import get_library_agents_by_ids

        agents = await get_library_agents_by_ids(
            user_id=user_id,
            agent_ids=library_agent_ids,
        )
        return cast(list[dict[str, Any]], agents)
    except Exception as e:
        logger.warning(f"Failed to fetch library agents by IDs: {e}")
        return None


async def fix_validate_and_save(
    agent_json: dict[str, Any],
    *,
    user_id: str | None,
    session_id: str | None,
    save: bool = True,
    is_update: bool = False,
    default_name: str = "Agent",
    preview_message: str | None = None,
    save_message: str | None = None,
    library_agents: list[dict[str, Any]] | None = None,
) -> ToolResponseBase:
    """Shared pipeline: auto-fix → validate → preview or save.

    Args:
        agent_json: The agent JSON dict (must already have id/version/is_active set).
        user_id: The authenticated user's ID.
        session_id: The chat session ID.
        save: Whether to save or just preview.
        is_update: Whether this is an update to an existing agent.
        default_name: Fallback name if agent_json has none.
        preview_message: Custom preview message (optional).
        save_message: Custom save success message (optional).
        library_agents: Library agents for AgentExecutorBlock validation/fixing.

    Returns:
        An appropriate ToolResponseBase subclass.
    """
    blocks = get_blocks_as_dicts()

    # Auto-fix
    try:
        fixer = AgentFixer()
        agent_json = await fixer.apply_all_fixes(agent_json, blocks, library_agents)
        fixes = fixer.get_fixes_applied()
        if fixes:
            logger.info(f"Applied {len(fixes)} auto-fixes to agent JSON")
    except Exception as e:
        logger.warning(f"Auto-fix failed: {e}")

    # Validate
    try:
        validator = AgentValidator()
        is_valid, _ = validator.validate(agent_json, blocks, library_agents)
        if not is_valid:
            errors = validator.errors
            return ErrorResponse(
                message=(
                    f"The agent has {len(errors)} validation error(s):\n"
                    + "\n".join(f"- {e}" for e in errors[:5])
                ),
                error="validation_failed",
                details={"errors": errors},
                session_id=session_id,
            )
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}", exc_info=True)
        return ErrorResponse(
            message="Failed to validate the agent. Please try again.",
            error="validation_exception",
            details={"exception": str(e)},
            session_id=session_id,
        )

    agent_name = agent_json.get("name", default_name)
    agent_description = agent_json.get("description", "")
    node_count = len(agent_json.get("nodes", []))
    link_count = len(agent_json.get("links", []))

    if not save:
        return AgentPreviewResponse(
            message=(
                preview_message
                or f"Agent '{agent_name}' with {node_count} blocks is ready."
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

    # Lazy import to avoid circular dependency
    from .core import save_agent_to_library

    try:
        created_graph, library_agent = await save_agent_to_library(
            agent_json, user_id, is_update=is_update
        )
        return AgentSavedResponse(
            message=(save_message or f"Agent '{created_graph.name}' has been saved!"),
            agent_id=created_graph.id,
            agent_name=created_graph.name,
            library_agent_id=library_agent.id,
            library_agent_link=f"/library/agents/{library_agent.id}",
            agent_page_link=f"/build?flowID={created_graph.id}",
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Failed to save agent: {e}", exc_info=True)
        return ErrorResponse(
            message=f"Failed to save the agent: {str(e)}",
            error="save_failed",
            details={"exception": str(e)},
            session_id=session_id,
        )
